import torch.nn as nn
from .modules import Activation
from torchvision.models.resnet import BasicBlock

# class SegmentationHead(nn.Sequential):

#     def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
#         conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
#         upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
#         activation = Activation(activation)
#         super().__init__(conv2d, upsampling, activation)

class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=None, activation=None, upsampling=1):
        dropout = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(dropout, conv2d, upsampling, activation)

class SegmentationHead3D(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=None, activation=None, upsampling=1):
        dropout = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear3d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()  # TODO not implemented
        activation = Activation(activation)
        super().__init__(dropout, conv3d, upsampling, activation)

class ClassificationHead(nn.Sequential):

    def __init__(self, in_channels, classes, pooling="avg", dropout=0.2, activation=None):
        if pooling not in ("max", "avg"):
            raise ValueError("Pooling should be one of ('max', 'avg'), got {}.".format(pooling))
        # conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        # relu1 = nn.ReLU(inplace=True)
        # bn1 = nn.BatchNorm2d(128)
        # conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        # relu2 = nn.ReLU(inplace=True)
        # bn2 = nn.BatchNorm2d(64)
        pool = nn.AdaptiveAvgPool2d(1) if pooling == 'avg' else nn.AdaptiveMaxPool2d(1)
        flatten = nn.Flatten()
        dropout = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        linear = nn.Linear(in_channels, classes, bias=True)  # 32 was `in_channels`
        activation = Activation(activation)
        # super().__init__(conv1, relu1, bn1, conv2, relu2, bn2, pool, flatten, dropout, linear, activation)
        super().__init__(pool, flatten, dropout, linear, activation)
