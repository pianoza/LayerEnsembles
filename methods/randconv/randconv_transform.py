import numpy as np
from methods.randconv.randconv import RandConvModule
import matplotlib.pyplot as plt

class RandConvTransform(object):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=(1, 3, 5), mixing=True, identity_prob=0.5, img_std=None, img_mean=None, rand_bias=None, clamp_out=False):
        mixing_alpha = np.random.random()  # sample mixing weights from uniform distributin (0, 1)
        self.rand_module = RandConvModule(
                          in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,  # tuple (1, 1), (3, 3) (1, 3, 5)
                          mixing=mixing,  # Mix with the original or not (True or False)
                          identity_prob=identity_prob,  # p: 0 - always transformed, 1 - keeps original
                          alpha=mixing_alpha,  
                          rand_bias=rand_bias,
                          distribution='kaiming_normal',
                          data_mean=img_mean,
                          data_std=img_std,
                          clamp_output=clamp_out,
                          )

    def __call__(self, sample):
        img = sample['scan']['data'].squeeze(-1).unsqueeze(0)
        randconv_img = self.rand_module(img)
        sample['scan']['data'] = randconv_img.squeeze(0).unsqueeze(-1).detach()
        return sample