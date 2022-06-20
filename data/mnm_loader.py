import torch
import numpy as np
import torchio as tio
import nibabel as nib
import multiprocessing
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def get_mnm_subjects(configs, images_dir, info_file, vendor_list='all'):
    '''Load all subjects from images dir
    Arguments
    :Path images_dir: path to dataset
    :DataFrame info: pandas data frame with dataset details
    :list vendor_list: list of vendors to fetch
    
    @returns list(torchio.Subject)
    '''
    # vendor_list = ['Siemens']
    info = pd.read_csv(info_file)
    subject_names = sorted([child.name for child in Path.iterdir(images_dir) if Path.is_dir(child)])
    subjects = []
    print('Loading MnM subjects...')
    transforms = tio.Compose([tio.Resample('mask'), tio.CropOrPad((128, 128, 10), mask_name='mask')])
    for subject in tqdm(subject_names):
        case = info.loc[info[configs.MnM_CODE] == subject]
        if isinstance(vendor_list, list):
            # skip if the current subject is not in the vendor list
            if not case[configs.MnM_VENDOR_NAME].item() in vendor_list:
                continue

        # load this image
        case_code = case[configs.MnM_CODE].item()
        # add timepoints separately
        for timepoint, suffix, gt_suffix in zip(
                    [configs.MnM_ED, configs.MnM_ES],
                    [configs.MnM_ED_SUFFIX, configs.MnM_ES_SUFFIX],
                    [configs.MnM_ED_GT_SUFFIX, configs.MnM_ES_GT_SUFFIX]
                ):
            img_path = images_dir / case_code / str(case_code + suffix)
            label_path = images_dir / case_code / str(case_code + gt_suffix)
            subject = tio.Subject(
                scan=tio.ScalarImage(img_path, type=tio.INTENSITY),
                mask=tio.LabelMap(label_path, type=tio.LABEL),
                timepoint=timepoint,
                code=case_code,
                vendor_name=case[configs.MnM_VENDOR_NAME].item(),
                vendor=case[configs.MnM_VENDOR].item(),
                centre=case[configs.MnM_CENTRE].item(),
                ed=case[configs.MnM_ED].item(),
                es=case[configs.MnM_ES].item(),
            )
            subject = transforms(subject)
            subjects.append(subject)
    return subjects
    
def MnMDataLoader(
                subjects_dataset,
                batch_size,
                max_length,
                samples_per_volume,
                sampler,
                shuffle,
                num_workers
        ):

    patches_set = tio.Queue(
            subjects_dataset=subjects_dataset,
            max_length=max_length,
            samples_per_volume=samples_per_volume,
            sampler=sampler,
            num_workers=num_workers,
            shuffle_subjects=shuffle,
            shuffle_patches=shuffle,
    )

    loader = torch.utils.data.DataLoader(
        patches_set, batch_size=batch_size, num_workers=0)

    return loader

