import torch
import torchvision.models as models
from collections import OrderedDict 
from segmentation_models_pytorch.unet.model import Unet
from segmentation_models_pytorch.base.heads import ClassificationHead
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base.model import ClassificationModel
from segmentation_models_pytorch.losses.dice import DiceLoss

from typing import Optional, List, Union
from torch.nn import ModuleList
from utils import Task, Organ
from methods.layer_ensembles import LayerEnsembles

def get_model_for_task(task, organ, layer_ensembles, target_shape, encoder_weights=None):
    # TODO check the number of classes for cardiac (segmentation: 4, classification: ?)
    num_classes = 2 if organ == Organ.BREAST else 4  
    if task == Task.SEGMENTATION:
        architecture = Unet(
            encoder_name="resnet18",
            encoder_weights=encoder_weights,
            decoder_channels=(1024, 512, 256, 128, 64),
            decoder_attention_type='scse',
            in_channels=1,
            classes=num_classes,
        )
        if not layer_ensembles:
            return architecture, []
        all_layers = dict([*architecture.named_modules()])
        intermediate_layers = []
        for name, layer in all_layers.items():
            # Change .relu to any other component e.g., .bn or .conv the '.' is to include only sub-modules (exclude stem)
            if '.relu' in name:
                intermediate_layers.append(name)
        # Init LayerEnsembles with the names of the intermediate layers to use as outputs
        model = LayerEnsembles(architecture, intermediate_layers)
        # Dummy input to get the output shapes of the layers
        x = torch.zeros(target_shape)
        output = model(x)
        out_channels = []
        scale_factors = []
        for _, val in output.items():
            out_channels.append(val.shape[1])
            scale_factors.append(target_shape[-1] // val.shape[-1])
        # Set the output heads with the number of channels of the output layers
        model.set_output_heads(in_channels=out_channels, scale_factors=scale_factors, task=Task.SEGMENTATION, classes=2)
        return model, intermediate_layers
    elif task == Task.CLASSIFICATION:
        architecture = models.resnet18(weights=None, num_classes=2)
        if not layer_ensembles:
            return architecture, []
        architecture.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # TODO Avoid repetition here!!! This is the same as in all other tasks
        # Useful to get the names of the layers to use in the layer ensembles
        all_layers = dict([*architecture.named_modules()])
        intermediate_layers = []
        for name, layer in all_layers.items():
            # Change .relu to any other component e.g., .bn or .conv the '.' is to include only sub-modules (exclude stem)
            if '.relu' in name:
                intermediate_layers.append(name)
        # Init LayerEnsembles with the names of the intermediate layers to use as outputs
        model = LayerEnsembles(architecture, intermediate_layers)
        # Dummy input to get the output shapes of the layers
        x = torch.randn(1, 1, 128, 128)
        output = model(x)
        out_channels = []
        for _, val in output.items():
            out_channels.append(val.shape[1])
        # Set the output heads with the number of channels of the output layers
        model.set_output_heads(in_channels=out_channels, task=Task.CLASSIFICATION, classes=2)
        return model, intermediate_layers
    else:
        raise ValueError('Unknown task: {}'.format(task))


def get_criterion_for_task(task, classes):
    if task == Task.SEGMENTATION:
        return DiceLoss(
            mode='multiclass',
            classes=classes,
            log_loss=False,
            from_logits=True,
            smooth=0.0000001,
            ignore_index=None,
        )
    elif task == Task.CLASSIFICATION:
        return torch.nn.CrossEntropyLoss()
    else:
        raise ValueError('Unknown task: {}'.format(task))


class EncoderModel(ClassificationModel):
    """Used for classification task. Builds an encoder architecture from any torchvision models
    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features 
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and 
            other pretrained weights (see table with available weights for each encoder_name)
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
            Default is **None**
        classes (int): A number of classes
        pooling (str): One of "max", "avg". Default is "avg"
        dropout (float): Dropout factor in [0, 1)
        activation (str): An activation function to apply "sigmoid"/"softmax" (could be **None** to return logits)
        layer_ensembles (bool): If True, the model will contain output heads after each block in the encoder
    Returns:
        ``torch.nn.Module``: Encoder architecture
    """

    def __init__(
        self,
        encoder_name: str = "resnet18",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 3,
        activation: Optional[Union[str, callable]] = None,
        classes: int = 1,
        pooling: str = "avg",
        dropout: Optional[float] = None,
        layer_ensembles: Optional[bool] = True,
    ):
        super().__init__()

        self.layer_ensembles = layer_ensembles

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.inter_activations = []  # TODO this is not working, during the forward pass there are extra tensors (len is 17 instead of 9 for some reason)
        
        if layer_ensembles:
            self.all_pre_outputs, prevs = self.find_all_relus(self.encoder)
            out_channels = [prev.num_features for prev in prevs] + [self.encoder.out_channels[-1]]  # add classification head to the last layer too

            # Register hooks for all_pre_outputs
            for i, pre_output in enumerate(self.all_pre_outputs):
                # self.inter_activations[pre_output] = []
                pre_output.register_forward_hook(self.get_inter_activations())
            
            self.classification_heads = ModuleList([
                ClassificationHead(
                    in_channels=out_channel,
                    classes=classes,
                    pooling=pooling,
                    dropout=dropout,
                    activation=activation,
                # ) for in_channels in self.encoder.out_channels[1:]
                ) for out_channel in out_channels 
            ])
        else:
            self.classification_heads = ClassificationHead(
                in_channels=self.encoder.out_channels[-1],
                classes=classes,
                pooling=pooling,
                dropout=dropout,
                activation=activation,
            )
        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def get_inter_activations(self):
        def hook(model, input, output):
            self.inter_activations.append(output)
        return hook

    def find_all_relus(self, model):
        relus = []
        prevs = []
        prev = None
        for module in model.modules():
            if isinstance(module, torch.nn.ReLU):
                relus.append(module)
                prevs.append(prev)
            prev = module
        return relus, prevs

class SingleOutputModel(torch.nn.Module):
    def __init__(self, model, index):
        super().__init__()
        self.model = model
        self.index = index
    def forward(self, x):
        return self.model(x)[self.index]