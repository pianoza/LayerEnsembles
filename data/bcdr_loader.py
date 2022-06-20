import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
import pickle
from torch.utils.data import Dataset
from PIL import Image
import torchio as tio
import skimage.measure
# from utils import dist_map_transform
from torch import Tensor
from utils import Task
from enum import Enum

VIEW_DICT = {1: ['RIGHT', 'CC'],
             2: ['LEFT', 'CC'],
             3: ['RIGHT', 'MLO'],
             4: ['LEFT', 'MLO']}

IMAGE_TYPE_DICT = {0: ['RIGHT', 'CC'],
                   1: ['LEFT', 'CC'],
                   2: ['RIGHT', 'MLO'],
                   3: ['LEFT', 'MLO'],
                   4: ['PR', 'MLO'],
                   5: ['PL', 'MLO'],
                   6: ['RF', 'CC'],
                   7: ['LF', 'CC'],
                   8: ['RF', 'MLO'],
                   9: ['LF', 'MLO']}

class BCDRDataset():

    def __init__(self, info_file:str, dataset_path:str, outlines_csv=None, load_max=0, classes=None):

        self.path = dataset_path
        self.info_file_csv = info_file
        self.outlines_csv = outlines_csv
        df_outlines =  None
        if outlines_csv:
            df_outlines = BCDRSegmentationDataset(outlines_csv=outlines_csv, dataset_path=dataset_path)

        info_file_csv = pd.read_csv(info_file, delimiter=',')
        info_file_csv = info_file_csv.astype(object).replace(np.nan, '')
        # info_file_csv.head()
        self._case =  {'filename':[], 'scan':[], 'mask':[], 'pathology':[], 'classification':[],
         'patient_id':[], 'laterality':[], 'view':[], 'age':[], 'ACR':[], 'study_id':[], 'lesion_id':[]}
        counter = 0
        for index, case in info_file_csv.iterrows():
            if load_max != 0 and counter >= load_max:
                break
            image_filename = case['image_filename']
            image_filename = image_filename.replace(' ', '')
            self._case['filename'].append(image_filename)
            self._case['scan'].append(os.path.join(self.path, image_filename))
            self._case['patient_id'].append(case['patient_id'])
            self._case['study_id'].append(case['study_id'])
            self._case['age'].append(case['age'])
            image_view = int(case['image_type_id'])
            self._case['laterality'].append(IMAGE_TYPE_DICT[image_view][0])
            self._case['view'].append(IMAGE_TYPE_DICT[image_view][1])
            if isinstance(case['density'], str):
                density = case['density'].replace(' ', '')
                if density == 'NaN':
                    self._case['ACR'].append(0)
                else:
                    self._case['ACR'].append(int(density))
            else:
                self._case['ACR'].append(int(case['density']))
            if df_outlines:
                index = df_outlines.get_outlines_index_by_filename(image_filename)
                if index:
                    patient_info = df_outlines[index]
                    self._case['classification'].append(patient_info['classification'])
                    self._case['lesion_id'].append(patient_info['lesion_id'])
                    self._case['pathology'].append(patient_info['pathology'])
                    self._case['mask'].append(patient_info['mask'])
                else:
                    self._case['classification'].append('Normal')
                    self._case['lesion_id'].append(None)
                    self._case['pathology'].append(None)
                    self._case['mask'].append(None)
            else:
                self._case['classification'].append('Normal')
                self._case['lesion_id'].append(None)
                self._case['pathology'].append(None)
                self._case['mask'].append(None)
            counter += 1

    def __len__(self):
        return len(self._case['scan'])

    def __getitem__(self, idx):
        d = {}
        for k, vs in self._case.items():
            value = vs[idx]
        [d.update({k: vs[idx]}) for k, vs in self._case.items()]
        return d

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self):
            d = self.__getitem__(self.n)
            self.n += 1
            return d
        else:
            raise StopIteration
    
    def get_scan_number_by_classification(self, classification):
        if classification not in self._case['classification']:
            return None
        else:
            count = self._case['classification'].count(classification)
            return count
    
    def get_scan_number_by_density(self, ACR):
        if ACR not in self._case['ACR']:
            return None
        else:
            count = self._case['ACR'].count(ACR)
            return count
    
    def print_distribution(self):
        for ACR in [1, 2, 3, 4]:
            count = self._case['ACR'].count(ACR)
            index_list = [i for i, value in enumerate(self._case['ACR']) if value == ACR]
            normal = 0
            benign = 0
            malign = 0
            for idx in index_list:
                case = self.__getitem__(idx)
                if case['classification'] == 'Benign':
                    benign +=1
                elif case['classification'] == 'Malign':
                    malign +=1
                else:
                    normal +=1
            print(f'ACR: {ACR} - Total scans: {count}')
            print(f'Normal scans: {normal} - Benign scans: {benign} - Malign scans: {malign}')



class BCDRSegmentationDataset():

    def __init__(self, outlines_csv:str, dataset_path:str, load_max=0, classes=None):

        self.path = dataset_path
        self.outlines_csv = outlines_csv
        outlines = pd.read_csv(outlines_csv, delimiter=',')
        outlines = outlines.astype(object).replace(np.nan, '')
        # outlines.head()
        self._case =  {'filename':[], 'scan':[], 'mask':[], 'pathology':[], 'classification':[],
         'patient_id':[], 'laterality':[], 'view':[], 'age':[], 'ACR':[], 'study_id':[], 'lesion_id':[]}
        counter = 0
        for index, case in outlines.iterrows():
            if load_max != 0 and counter >= load_max:
                break
            image_filename = case['image_filename']
            image_filename = image_filename.replace(' ', '')
            self._case['filename'].append(image_filename)
            self._case['scan'].append(os.path.join(self.path, image_filename))
            self._case['classification'].append(case['classification'].replace(' ', ''))
            self._case['age'].append(case['age'])
            self._case['patient_id'].append(case['patient_id'])
            self._case['study_id'].append(case['study_id'])
            self._case['lesion_id'].append(case['lesion_id'])
            if isinstance(case['density'], str):
                density = case['density'].replace(' ', '')
                if density == 'NaN':
                    self._case['ACR'].append(0)
                else:
                    self._case['ACR'].append(int(density))
            else:
                self._case['ACR'].append(int(case['density']))
            image_view = int(case['image_view'])
            if image_view > 4:
                self._case['laterality'].append('UNKNOWN')
                self._case['view'].append('UNKNOWN')
            else:
                self._case['laterality'].append(VIEW_DICT[image_view][0])
                self._case['view'].append(VIEW_DICT[image_view][1])
            self._case['mask'].append(os.path.join(self.path, image_filename[:-4] + '_mask_id_' + str(case['lesion_id']) + '.tif'))
            pathologies = []
            if int(case['mammography_nodule']):
                pathologies.append('nodule')
            if int(case['mammography_calcification']):
                pathologies.append('calcification')
            if int(case['mammography_microcalcification']):
                pathologies.append('microcalcification')
            if int(case['mammography_axillary_adenopathy']):
                pathologies.append('axillary_adenopathy')
            if int(case['mammography_architectural_distortion']):
                pathologies.append('architectural_distortion')
            if int(case['mammography_stroma_distortion']):
                pathologies.append('stroma_distortion')
            self._case['pathology'].append(pathologies)
            counter += 1

    def __len__(self):
        return len(self._case['scan'])

    def __getitem__(self, idx):
        d = {}
        # for k, vs in self._case.items():
        #     value = vs[idx]
        [d.update({k: vs[idx]}) for k, vs in self._case.items()]
        return d

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self):
            d = self.__getitem__(self.n)
            self.n += 1
            return d
        else:
            raise StopIteration
    
    def get_outlines_index_by_filename(self, filename):
        if filename not in self._case['filename']:
            return None
        else:
            index = self._case['filename'].index(filename)
            return index
    
    def get_scan_number_by_classification(self, classification):
        if classification not in self._case['classification']:
            return None
        else:
            count = self._case['classification'].count(classification)
            return count
    
    def get_scan_number_by_density(self, ACR):
        if ACR not in self._case['ACR']:
            return None
        else:
            count = self._case['ACR'].count(ACR)
            return count
    
    def print_distribution(self):
        for ACR in [1, 2, 3, 4]:
            count = self._case['ACR'].count(ACR)
            index_list = [i for i, value in enumerate(self._case['ACR']) if value == ACR]
            normal = 0
            benign = 0
            malign = 0
            for idx in index_list:
                case = self.__getitem__(idx)
                if case['classification'] == 'Benign':
                    benign +=1
                elif case['classification'] == 'Malign':
                    malign +=1
                else:
                    normal +=1
            print(f'ACR: {ACR} - Total scans: {count}')
            print(f'Normal scans: {normal} - Benign scans: {benign} - Malign scans: {malign}')

def load_dataset(path, info_csv, outlines_csv=None):
    scans = []
    masks = []
    pathologies = []
    classifications = []
    patient_ids = []
    lateralities = []
    views = []
    ages = []
    acrs = []
    study_ids = []
    lesion_ids = []
    for idx, (p, info, outlines) in enumerate(zip(path, info_csv, outlines_csv)):
        # Load all dataset data
        if info == 'None':
            df = BCDRSegmentationDataset(outlines_csv=outlines, dataset_path=p)
        else:
            if outlines == 'None':
                outlines = None
            df = BCDRDataset(info_file=info, outlines_csv=outlines, dataset_path=p)
        # df.print_distribution()
        with tqdm(total=len(df)) as pbar:
            # Iterate trought the dictionary
            counter = 0
            for case in df:
                '''
                {'filename': 'patient_182/study_188/img_182_188_LCC.tif',
                 'scan': '/home/kaisar/Datasets/BCDR/Processed/BCDR/BCDR-F03_dataset/BCDR-F03/patient_182/study_188/img_182_188_LCC.tif',
                 'mask': '/home/kaisar/Datasets/BCDR/Processed/BCDR/BCDR-F03_dataset/BCDR-F03/patient_182/study_188/img_182_188_LCC_mask_id_188.tif',
                 'pathology': ['nodule'],
                 'classification': 'Malign',
                 'patient_id': 182,
                 'laterality': 'LEFT',
                 'view': 'CC',
                 'age': 68.7781,
                 'ACR': 4,
                 'study_id': 188, 
                 'lesion_id': 188}
                '''
                if case['mask'] is not None:
                    scans.append(case['scan'])
                    masks.append(case['mask'])
                    pathologies.append(case['pathology'])
                    classifications.append(case['classification'])
                    patient_ids.append(case['patient_id'])
                    lateralities.append(case['laterality'])
                    views.append(case['view'])
                    ages.append(case['age'])
                    acrs.append(case['ACR'])
                    study_ids.append(case['study_id'])
                    lesion_ids.append(case['lesion_id'])
    df = {
            'scan': scans,
            'mask': masks,
            'pathology': pathologies,
            'classification': classifications,
            'patient_id': patient_ids,
            'laterality': lateralities,
            'view': views,
            'age': ages,
            'ACR': acrs,
            'study_id': study_ids, 
            'lesion_id': lesion_ids
        }
    df = pd.DataFrame(df) 
    return df

def get_bcdr_subjects(images_dir:str, info_csv:str, outlines_csv:str, task:Enum):
    raw_df = load_dataset(images_dir, info_csv, outlines_csv)
    df = {
        'scan': [],
        'mask': [],
        'status': [],
        'patient_id': [],
        'laterality': [],
        'view': [],
        'study_id': [],
        'lesion_id': [],
        'pathology': [],
        'unique_id': []
    }
    # Keep only the masses
    for idx, val in raw_df.iterrows():
        if task == Task.SEGMENTATION and 'nodule' not in val['pathology']:
            continue
        df['scan'].append(val['scan'])
        df['mask'].append(val['mask'])
        df['status'].append(1 if val['classification'].lower()=='malign' else 0)
        df['patient_id'].append(val['patient_id'])
        df['laterality'].append(val['laterality'])
        df['view'].append(val['view'])
        df['study_id'].append(val['study_id'])
        df['lesion_id'].append(val['lesion_id'])
        df['pathology'].append(val['pathology'])
        df['unique_id'].append(f"pid_{val['patient_id']}-{val['laterality']}_{val['view']}-sid_{val['study_id']}-lid_{val['lesion_id']}")
    df = pd.DataFrame(df)
    df = df.drop_duplicates(subset=['unique_id'], keep='first')
    return df


class BCDRLoader(Dataset):

    def __init__(self, df, transform=None, self_training_transforms=None):  #, configs=None, da_model=None, shapes=None, patch_size=(32, 32), ssim_threshold=None):
        self.df = df
        self.transform = transform
        self.self_training_transforms = self_training_transforms
        # self.disttransform = dist_map_transform([1, 1], 2)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.df.iloc[idx]['scan']
        mask_path = self.df.iloc[idx]['mask']
        unique_id = self.df.iloc[idx]['unique_id']
        # Check if the sample is from unannotated source
        try:
            columns = list(self.df.columns.values)
            if 'dsc_norm' not in columns:
                self_training_sample = False
            elif pd.isnull(self.df.iloc[idx]['dsc_norm']):
                self_training_sample = False
            else:
                self_training_sample = True
        except:
            print('EXCEPTION: Sample size:', self.df.shape[0])
            print('EXCEPTION: Column headers:', list(self.df.columns.values))
            raise ValueError('Key error. dsc_norm does not exist')

        if unique_id == 'TAG_SYNTH':
            # load already generated synthetic sample from pickle
            with open(img_path, 'rb') as f:
                sample = pickle.load(f)
        else: 
            image = np.array(Image.open(img_path)).astype(np.float32)
            mask = np.array(Image.open(mask_path)).astype(np.float32)
            # mask is loaded as 255
            mask[mask > 0] = 1
            sample = tio.Subject(
                scan = tio.ScalarImage(tensor=torch.from_numpy(image[None, ..., None]).float()),
                mask = tio.LabelMap(tensor=torch.from_numpy(mask[None, ..., None]).float()),
                status = self.df.iloc[idx]['status'],
                patient_id = str(self.df.iloc[idx]['patient_id']),
                laterality = str(self.df.iloc[idx]['laterality']),
                view = str(self.df.iloc[idx]['view']),
                study_id = str(self.df.iloc[idx]['study_id']),
                lesion_id = str(self.df.iloc[idx]['lesion_id']),
                pathology = str(self.df.iloc[idx]['pathology'])
            )
            if self_training_sample and self.self_training_transforms:
                sample = self.self_training_transforms(sample)
            elif self.transform:
                sample = self.transform(sample)
        return sample
        
