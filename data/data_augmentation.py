import numpy as np
import cv2
import torch
import torchio as tio
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from torchio.transforms.augmentation.random_transform import RandomTransform
from torchio.transforms.spatial_transform import SpatialTransform
from bezier_gan.pix2pix_setup import get_synthetic_sample


class SyntheticSampleGenerator(tio.transforms.augmentation.RandomTransform, tio.transforms.spatial_transform.SpatialTransform):
    def __init__(self, configs, model, shapes, patch_size, ssim_threshold, **kwargs):
        '''
        bezier_gan_model = get_augmentation_model(self.configs)
        shapes = ['oval', 'lobulated']
        # SyntheticSampleGenerator(
        #     configs=self.configs,
        #     model=bezier_gan_model,
        #     shapes=shapes,
        #     patch_size=(32, 32),
        #     ssim_threshold=0.35
        # ): 1,
        '''
        super().__init__(**kwargs)
        self.configs = configs
        self.model = model
        self.shapes = shapes
        self.patch_size = patch_size
        self.ssim_threshold = ssim_threshold

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        return get_synthetic_sample(self.configs, self.model, subject, self.shapes, self.patch_size, self.ssim_threshold)

# class DeformableTransform(tio.RandomTransform, tio.SpatialTransform):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)

#     def apply_transform(self, subject: tio.Subject) -> tio.Subject:
#         return self.get_defomed_sample(subject)

#     def get_deformed_sample(self, subject: tio.Subject) -> tio.Subject:
#         img = np.squeeze(subject['scan'])
#         mask = np.squeeze(subject['mask'])
#         # Merge images into separete channels (shape will be (cols, rols, 2))
#         im_merge = np.concatenate((img[..., None], mask[..., None]), axis=2)
#         # Apply transformation on image
#         im_merge_t = self.elastic_transform(image=im_merge,
#                                        alpha=im_merge.shape[1] * 8,
#                                        sigma=im_merge.shape[1] * 0.1,
#                                        alpha_affine=im_merge.shape[1] * 0.01)
#         # Split image and mask
#         im_t = np.float32(im_merge_t[..., 0])
#         im_mask_t = im_merge_t[..., 1]
#         mask_t = np.zeros_like(im_mask_t).astype(np.uint8)
#         mask_t[im_mask_t > 0.5] = 1
#         deformed_subject = tio.Subject(
#                 scan=tio.ScalarImage(tensor=torch.from_numpy(im_t[None, ..., None])),
#                 mask=tio.LabelMap(tensor=torch.from_numpy(mask_t[None, ..., None])),
#             )
#         return deformed_subject

#     def elastic_transform(self, image:np.ndarray , alpha: float, sigma:float, alpha_affine:float, random_state:int=None):
#         # Taken from: https://www.kaggle.com/bguberfain/elastic-transform-for-data-augmentation
#         # Function to distort image
#         if random_state is None:
#             random_state = np.random.RandomState(None)

#         shape = image.shape
#         shape_size = shape[:2]
        
#         # Random affine
#         center_square = np.float32(shape_size) // 2
#         square_size = min(shape_size) // 3
#         pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
#         pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
        
#         M = cv2.getAffineTransform(pts1, pts2)
#         image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

#         dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
#         dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
#         dz = np.zeros_like(dx)

#         x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
#         indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

#         return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
        
