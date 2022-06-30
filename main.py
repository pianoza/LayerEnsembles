# import torch
# from data.optimam_loader import optimam_healthy_nonhealthy_extraction
# from segmentation_models_pytorch.unet.model import Unet
# from torchsummary import summary
# import pandas as pd
# from torchvision.models import resnet18, resnet34, densenet121

# from torch.utils.tensorboard import SummaryWriter
# from methods.base_method_plain import BaseMethodPlain
# from methods.base_method_mnm_plain import BaseMethodMnMPlain
# from methods.base_method_mnm import BaseMethodMnM
# from methods.deep_ensembles_mnm import DeepEnsemblesMnM
# from methods.commons import get_model_for_task
# from methods.layer_ensembles import LayerEnsembles
# from utils import Task, Organ
# from data.mmg_detection_datasets import OPTIMAMDataset

import configs 
from methods.base_method import BaseMethod

if __name__ == '__main__':
    method = BaseMethod(configs, layer_ensembles=configs.IS_LAYER_ENSEMBLES)
    method.run_routine()

    # architecture = resnet34(weights=None, num_classes=2)
    # architecture.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # # Useful to get the names of the layers to use in the layer ensembles
    # all_layers = dict([*architecture.named_modules()])
    # intermediate_layers = []
    # for name, layer in all_layers.items():
    #     # Change .relu to any other component e.g., .bn or .conv the '.' is to include only sub-modules (exclude stem)
    #     if '.relu' in name:
    #         intermediate_layers.append(name)
    # # Init LayerEnsembles with the names of the intermediate layers to use as outputs
    # model = LayerEnsembles(architecture, intermediate_layers)
    # # Dummy input to get the output shapes of the layers
    # x = torch.randn(1, 1, 128, 128)
    # output = model(x)
    # out_channels = []
    # for key, val in output.items():
    #     out_channels.append(val.shape[1])
    # # Set the output heads with the number of channels of the output layers
    # model.set_output_heads(in_channels=out_channels, task=Task.CLASSIFICATION, classes=2)
    # # Model is ready to be used
    # outputs = model(x)
    # print(len(outputs))
    # for layer, out in outputs.items():
    #     print(layer, out.shape)


    # optimam_healthy_nonhealthy_extraction()

    # for status in ['Normal', 'Benign', 'Malignant']:
    #     if status == 'Normal':
    #         print('Selected client example:', selected_clients[0].total_images(status=status))
    #         print('File path:', selected_clients[0].get_image_path(status=status))
    #     else: 
    #         selected_clients = optimam_clients.get_clients_by_pathology_and_status(pathologies, status)
    #         print('Selected client example:', selected_clients[0].total_images(pathologies=pathologies, status=status))
    #     print(f'Total clients selected by status ({status}): {len(selected_clients)}')

        # # If you want to select images by center:
        # clients_selected = optimam_clients.get_images_by_site('stge')
        # print(f'Total clients selected: {len(clients_selected)}')
