import torchio as tio
from utils import ResampleToMask, Task, Organ
from methods.randconv.randconv_transform import RandConvTransform

def get_transforms(task, num_classes, target_image_size=256,
                   min_intensity=0, max_intensity=1, min_percentile=0.5, max_percentile=99.5, perturb_test=False):
    seg_preprocess = [
        ResampleToMask(im_size=target_image_size),
        tio.CropOrPad((target_image_size, target_image_size, 1), mask_name='mask'),
        tio.RescaleIntensity(out_min_max=(min_intensity, max_intensity), percentiles=(min_percentile, max_percentile), masking_method='mask'),
        tio.ZNormalization(masking_method='mask'),
        # tio.OneHot(num_classes=num_classes),
        ]
    clf_preprocess = [
        tio.Resize((target_image_size, target_image_size, 1)),
        tio.Resample((4, 4, 1)),  # Downsample by 4, e.g., 1024 -> 256, 2560 -> 640
        tio.RescaleIntensity(out_min_max=(min_intensity, max_intensity), percentiles=(min_percentile, max_percentile), masking_method='mask'),
        tio.ZNormalization(masking_method='mask'),
        # tio.OneHot(num_classes=num_classes),
        ]
    aug_transforms = [
        tio.RandomFlip(axes=(0, 1), p=0.2),
        RandConvTransform(kernel_size=(1, 3, 5, 7), mixing=True, identity_prob=0.8),
        tio.RandomSwap((10, 10, 1), 10, p=0.2),
    ]
    perturb_transforms = []
    if perturb_test:
        perturb_transforms = [
            tio.Compose([
                RandConvTransform(kernel_size=(37, 37), mixing=False, identity_prob=0.0),
                tio.RandomBlur((15, 15, 15, 15, 0, 0), p=1.),
                tio.RandomNoise((5., 5.001), (.7, .701), p=1.),
                tio.RandomSwap((20, 20, 1), 10, p=1.),
                tio.RescaleIntensity((0, 1), (0, 100)),
                tio.RandomGamma((5.0, 5.1), p=1.)
            ], p=1.)
        ]
    if task == Task.SEGMENTATION:
        transforms = seg_preprocess
    elif task == Task.CLASSIFICATION:
        transforms = clf_preprocess
    elif task == Task.REGRESSION:
        raise NotImplementedError
    else:
        raise NotImplementedError
    train_transforms = tio.Compose(transforms + aug_transforms)
    test_transforms = tio.Compose(transforms + perturb_transforms)

    return train_transforms, test_transforms