from enum import Enum
import os
import torch
import numpy as np
import pandas as pd
import nibabel as nib
import torchio as tio
from torch.utils.data import Dataset
import skimage.measure
from PIL import Image
from utils import Task, bbox2

def get_inbreast_subjects(info_file_path:str, task:Enum):
    # read info
    raw_df = pd.read_csv(info_file_path)
    df = { 'scan': [], 'mask': [], 'status': [], 'patient_id': [],
           'laterality': [], 'view': [], 'study_id': [], 'lesion_id': [], 'unique_id': [],
    }
    if task == Task.SEGMENTATION:
        # only mass cases
        raw_df = raw_df.dropna(subset=['Mass'])
    for idx, row in raw_df.iterrows():
        # Convert Bi-Rads to 0-1
        status = row['Bi-Rads']
        if row['Bi-Rads'] in ['1', '2', '3']:
            status = 0 # benign
        else:
            status = 1 # malignant
        if task == Task.SEGMENTATION:
            for i in range(0, 3):
                if pd.notna(row['Mass_{}'.format(i)]):
                    df['scan'].append(row['Scan'])
                    df['mask'].append(row['Mass_{}'.format(i)])
                    df['status'].append(status)
                    df['patient_id'].append(row['Patient ID'])
                    df['laterality'].append(row['Laterality'])
                    df['view'].append(row['View'])
                    df['study_id'].append(row['Acquisition date'])
                    df['lesion_id'].append(row['File Name'])
                    df['unique_id'].append(f"pid_{row['Patient ID']}-{row['Laterality']}_{row['View']}-sid_{row['Acquisition date']}-lid_{row['File Name']}")
        elif task == Task.CLASSIFICATION:
            df['scan'].append(row['Scan'])
            df['mask'].append(None)
            df['status'].append(status)
            df['patient_id'].append(row['Patient ID'])
            df['laterality'].append(row['Laterality'])
            df['view'].append(row['View'])
            df['study_id'].append(row['Acquisition date'])
            df['lesion_id'].append(row['File Name'])
            df['unique_id'].append(f"pid_{row['Patient ID']}-{row['Laterality']}_{row['View']}-sid_{row['Acquisition date']}-lid_{row['File Name']}")

    df = pd.DataFrame(df)
    return df

class INBreastDataset(Dataset):

    def __init__(self, df, images_dir, transform=None):
        self.df = df
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def breast_mask(self, img):
        mask = np.zeros_like(img).astype(np.uint8)
        mask[img > 0] = 1
        labels = skimage.measure.label(mask, return_num=False)
        # maxCC_withbcg = labels == np.argmax(np.bincount(labels.flat))
        maxCC_nobcg = labels == np.argmax(np.bincount(labels.flat, weights=mask.flat))
        return maxCC_nobcg

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.df.iloc[idx]['scan']
        mask_path = self.df.iloc[idx]['mask']
        image = np.array(nib.load(os.path.join(self.images_dir, img_path)).get_fdata().squeeze()).astype(np.float32)
        # Some images have multiple masks and they are loaded as different rows in the get_inbreast_subjects() function
        if not mask_path is None:
            # segmentation will have a mask_path
            mask = np.array(nib.load(os.path.join(self.images_dir, mask_path)).get_fdata().squeeze()).astype(np.uint8)
        else:
            # breast area will be used as a mask in classification
            mask = self.breast_mask(image)
            # rmin, rmax, cmin, cmax = bbox2(mask)
            # image = image[rmin:rmax, cmin:cmax]
            # mask = mask[rmin:rmax, cmin:cmax]
        image = image.T
        mask = mask.T
        sample = tio.Subject(
            scan = tio.ScalarImage(tensor=torch.from_numpy(image[None, ..., None])),
            mask = tio.LabelMap(tensor=torch.from_numpy(mask[None, ..., None])),
            status = self.df.iloc[idx]['status'],
            patient_id = self.df.iloc[idx]['patient_id'],
            laterality = self.df.iloc[idx]['laterality'],
            view = self.df.iloc[idx]['view'],
            study_id = self.df.iloc[idx]['study_id'],
            lesion_id = self.df.iloc[idx]['lesion_id']
        )
        if self.transform:
            sample = self.transform(sample)

        return sample