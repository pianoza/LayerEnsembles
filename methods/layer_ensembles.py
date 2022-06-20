from tkinter import E
import torch
from torch import nn, Tensor
from torchvision.models import resnet18
from utils import Task, Organ

from typing import Dict, Iterable, Callable

class ArgMax(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
    def forward(self, x):
        return torch.argmax(x, dim=self.dim)

class Activation(nn.Module):
    def __init__(self, name, **params):
        super().__init__()
        if name is None or name == 'identity':
            self.activation = nn.Identity(**params)
        elif name == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif name == 'softmax2d':
            self.activation = nn.Softmax(dim=1, **params)
        elif name == 'softmax':
            self.activation = nn.Softmax(**params)
        elif name == 'logsoftmax':
            self.activation = nn.LogSoftmax(**params)
        elif name == 'tanh':
            self.activation = nn.Tanh()
        elif name == 'argmax':
            self.activation = ArgMax(**params)
        elif name == 'argmax2d':
            self.activation = ArgMax(dim=1, **params)
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError('Activation should be callable/sigmoid/softmax/logsoftmax/tanh/None; got {}'.format(name))

    def forward(self, x):
        return self.activation(x)

class ClassificationHead(nn.Sequential):
    def __init__(self, in_channels, classes, pooling='avg', activation=None, dropout=None):
        if pooling not in ("max", "avg"):
            raise ValueError("Pooling should be one of ('max', 'avg'), got {}.".format(pooling))
        pool = nn.AdaptiveAvgPool2d(1) if pooling == 'avg' else nn.AdaptiveMaxPool2d(1)
        flat = nn.Flatten()
        drop = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        linr = nn.Linear(in_channels, classes)
        act = Activation(activation)
        super().__init__(pool, flat, drop, linr, act)

class LayerEnsembles(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))
        self._layer_ensemble_active = False

    def set_output_heads(self, in_channels: Iterable[int], task: str, classes: int, pooling: str = 'avg', activation: str = None, dropout: float = None):
        if self._layer_ensemble_active:
            raise ValueError("Output heads should be set only once.")
        self._layer_ensemble_active = True
        if task == Task.CLASSIFICATION:
            self.output_heads = nn.ModuleList([
                ClassificationHead(
                    in_channels=in_channel,
                    classes=classes,
                    pooling=pooling,
                    dropout=dropout,
                    activation=activation,
                ) for in_channel in in_channels 
            ])
        elif task == Task.SEGMENTATION:
            raise NotImplementedError
        elif task == Task.REGRESSION:
            raise NotImplementedError
        else:
            raise ValueError(f"Task {task} is not supported.")

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        final_layer_output = self.model(x)
        if not self._layer_ensemble_active:
            outputs = {layer: self._features[layer] for layer in self.layers}
        else: 
            outputs = {layer: head(self._features[layer]) for head, layer in zip(self.output_heads, self.layers)}
        outputs['final'] = final_layer_output
        return outputs
