from randconv import RandConvModule
import numpy as np
from PIL import Image
import torch
import os
import torchvision

def tensor2np(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        # if image_numpy.shape[0] == 1:  # grayscale to RGB
        #     image_numpy = np.tile(image_numpy, (3, 1, 1))
        # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
        if image_numpy.shape[0] == 1:
            image_numpy = image_numpy.reshape(image_numpy.shape[1], image_numpy.shape[2])
            image_numpy = (image_numpy + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
        else:
            image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
        
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

class RandConvAug:

    def __init__(self, kernel_size, mixing, identity_prob, mixing_alpha, img_std, img_mean, img_scale, 
                    in_channels=3, to_rgb=False, rand_bias=False, clamp_out=False):
        self.img_scale = img_scale
        self.to_rgb = to_rgb
        self.out_width = img_scale[1]
        self.out_height = img_scale[0]
        self.apply = True
        self.kernel_size = kernel_size
        self.mixing = mixing
        self.identity_prob = identity_prob
        self.img_std = img_std
        self.img_mean = img_mean
        self.in_channels = in_channels
        if to_rgb:
            self.out_channels = 3
        else:
            self.out_channels = 1
        self.rand_module = RandConvModule(
                          in_channels=in_channels,
                          out_channels=self.out_channels,
                          kernel_size=kernel_size,
                          mixing=mixing,
                          identity_prob=identity_prob,
                          alpha=mixing_alpha,
                          rand_bias=rand_bias,
                          distribution='kaiming_normal',
                          data_mean=img_mean,
                          data_std=img_std,
                          clamp_output=clamp_out,
                          )
        self.rand_module.to(device='cuda')

    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            img_u8 = results[key]
            # print(results['scale_factor'])
            img = np.uint8(img_u8) if img_u8.dtype != np.uint8 else img_u8.copy()
            # Normalize image 
            #img = (img - 123) / 57
            # ori_img_pil = Image.fromarray(img)
            # ori_img_pil.save(os.path.join('/home/lidia-garrucho/source/mmdetection/data_augmentation/original_image.png'))
            # print(f'----> img shape {img.shape}') #(1321, 800, 3)
            #img_pil = Image.fromarray(img)
            #print(f'----> img_pil size {img_pil.size}')
            randconv_img = self.rand_module(img)
            #print(randconv_img.shape)
            if len(randconv_img.shape) == 3 and self.out_channels == 3:
                image_numpy = randconv_img
                #print('Enter in 1')
            elif self.mixing and len(randconv_img.shape) == 2:
                image_numpy = torch.from_numpy(np.asarray(randconv_img))
                image_numpy = image_numpy.unsqueeze(2)
                #print('Enter in 2')
            else:
                #print('Enter in 3')
                if len(randconv_img.shape) == 3:
                    image_tensor = randconv_img.data
                    image_numpy = image_tensor.cpu().float().numpy()
                else:
                    image_tensor = randconv_img.data
                    image_numpy = image_tensor[0].cpu().float().numpy()
                image_numpy = np.transpose(image_numpy, (1, 2, 0))
                #image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
                image_numpy = image_numpy.astype(np.uint8)
                if image_numpy.shape[2] == 1: #input grayscale
                    image_numpy = np.tile(image_numpy, (1, 1, 3))
            #print(image_numpy.shape)
            #out_img_pil = Image.fromarray(image_numpy.squeeze(2)) #grayscale
            #out_img_pil = Image.fromarray(image_numpy)
            #out_img_pil.save(os.path.join('/home/lidia-garrucho/source/mmdetection/data_augmentation/randconv_test.png'))
            #print(image_numpy.shape)
            #exit(0)
            results[key] = image_numpy
            # print(f'----> IN')
            # self.apply = False
        # else:
            # print(f'----> OUT')
        #     self.apply = True
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(kernel_size={self.kernel_size}, '
        repr_str += f'(mixing={self.mixing}, '
        repr_str += f'(identity_prob={self.identity_prob}, '
        repr_str += f'(img_std={self.img_std}, '
        repr_str += f'(img_mean={self.img_mean}, '
        repr_str += f'(in_channels={self.in_channels}, '
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'to_rgb={self.to_rgb}, '
        return repr_str
