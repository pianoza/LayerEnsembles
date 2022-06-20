import os
import torch
import numpy as np
import pandas as pd
import nibabel as nib
import torchio as tio
from torch.utils.data import Dataset
import skimage.measure
from PIL import Image
from data.mmg_detection_datasets import OPTIMAMDataset
from pathlib import Path
from tqdm import tqdm

def get_optimam_subjects(info_file_path:str):
    # read info
    df = pd.read_csv(info_file_path)
    return df

def breast_mask(img):
        mask = np.zeros_like(img).astype(np.uint8)
        mask[img > 0] = 1
        labels = skimage.measure.label(mask, return_num=False)
        # maxCC_withbcg = labels == np.argmax(np.bincount(labels.flat))
        maxCC_nobcg = labels == np.argmax(np.bincount(labels.flat, weights=mask.flat))
        return maxCC_nobcg

class OPTIMAMDatasetLoader(Dataset):

    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.df.iloc[idx]['scan']
        mask_path = self.df.iloc[idx]['mask']
        image = np.array(Image.open(img_path)).astype(np.float32)
        mask = np.array(Image.open(mask_path)).astype(np.uint8)
        image = image.T
        mask = mask.T
        status = 0 if self.df.iloc[idx]['status'] == 'Normal' else 1

        sample = tio.Subject(
            scan = tio.ScalarImage(tensor=torch.from_numpy(image[None, ..., None])),
            mask = tio.LabelMap(tensor=torch.from_numpy(mask[None, ..., None])),
            status = status,
            patient_id = self.df.iloc[idx]['patient_id'],
            laterality = self.df.iloc[idx]['laterality'],
            view = self.df.iloc[idx]['view'],
            study_id = self.df.iloc[idx]['study_id'],
            lesion_id = self.df.iloc[idx]['lesion_id']
        )
        if self.transform:
            sample = self.transform(sample)

        return sample

def optimam_healthy_nonhealthy_extraction():
    # OPTIMAM healthy vs abnormal
    info_csv='/datasets/OPTIMAM/png_screening_cropped_fixed/client_images_screening.csv'
    dataset_path='/datasets/OPTIMAM/png_screening_cropped_fixed/images'
    out_path = Path('/datasets/OPTIMAM/png_screening_cropped_fixed/healthy_nonhealthyMass')
    cropped_to_breast = True
    detection = False
    load_max = -1 #10 Only loads 10 images
    pathologies = ['mass']  #, 'calcifications', 'suspicious_calcifications', 'architectural_distortion'] # None to select all
    statuses = ['Normal', 'Malignant'] #['Normal', 'Benign'] 
    # Resize images keeping aspect ratio
    plot_images = False
    optimam_clients = OPTIMAMDataset(info_csv, dataset_path, detection=detection, load_max=load_max, 
                                     cropped_to_breast=cropped_to_breast)
    rescale_factor = 4
    # Get healthy clients
    df = { 'scan': [], 'mask': [], 'status': [], 'patient_id': [],
           'laterality': [], 'view': [], 'study_id': [], 'lesion_id': [], 'unique_id': [],
    }
    for client in tqdm(optimam_clients):
        for study in client.studies:
            for image in study:
                pathology_list = [p for annot in image.annotations for p in annot.pathologies]
                if image.status == 'Normal' or (image.status == 'Malignant' and 'mass' in pathology_list):
                    print(image.status, pathology_list)
                    img_pil = Image.open(image.path)  # .convert('RGB')
                    img_pil = img_pil.resize((img_pil.width//rescale_factor, img_pil.height//rescale_factor), Image.ANTIALIAS)
                    img_np = np.array(img_pil)
                    mask = breast_mask(img_np)
                    mask_pil = Image.fromarray(mask)
                    # /datasets/OPTIMAM/png_screening_cropped_fixed/images/demd134970/1.2.826.0.1.3680043.9.3218.1.1.10040326.1340.1547192291975.194.0/1.2.826.0.1.3680043.9.3218.1.1.10040326.1340.1547192291975.196.0.png
                    img_save_path = image.path.replace(str(dataset_path), str(out_path))
                    mask_save_path = image.path.replace(str(dataset_path), str(out_path)).replace('.png', '_mask.png')
                    # create directiories
                    Path(img_save_path).parent.mkdir(parents=True, exist_ok=True)
                    img_pil.save(img_save_path)
                    mask_pil.save(mask_save_path)
                    # print(f'Client id: {client.id} | Study id: {study.id} | Image id: {image.id}\nStatus: {image.status} | Laterality: {image.laterality} | View: {image.view}')
                    df['scan'].append(img_save_path)
                    df['mask'].append(mask_save_path)
                    df['status'].append(image.status)
                    df['patient_id'].append(client.id)
                    df['laterality'].append(image.laterality)
                    df['view'].append(image.view)
                    df['study_id'].append(study.id)
                    df['lesion_id'].append(None)
                    df['unique_id'].append(f"pid_{client.id}-{image.laterality}_{image.view}-sid_{study.id}")
    df = pd.DataFrame(df)
    df.to_csv('/datasets/OPTIMAM/png_screening_cropped_fixed/healthy_nonhealthyMass.csv', index=False)