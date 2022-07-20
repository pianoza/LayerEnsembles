from functools import partial
import configs
import torch
import torchio as tio
from torch.utils.data import DataLoader
from data.bcdr_loader import BCDRLoader, get_bcdr_subjects
from data.inbreast_loader import INBreastDataset, get_inbreast_subjects
from data.optimam_loader import OPTIMAMDatasetLoader, get_optimam_subjects
from data.mnm_loader import get_mnm_subjects, MnMDataLoader
from sklearn.model_selection import GroupShuffleSplit
from utils import Task, Organ

def unique_group_train_val_test_split(df, train_val_test_split):
    train_inds, test_inds = next(GroupShuffleSplit(test_size=train_val_test_split[2], n_splits=2, random_state=configs.RANDOM_SEED).split(df, groups=df['patient_id']))
    df_tmp = df.iloc[train_inds]
    df_test = df.iloc[test_inds]
    train_inds, val_inds = next(GroupShuffleSplit(test_size=train_val_test_split[1], n_splits=2, random_state=configs.RANDOM_SEED).split(df_tmp, groups=df_tmp['patient_id']))
    df_train = df_tmp.iloc[train_inds]
    df_val = df_tmp.iloc[val_inds]
    return df_train, df_val, df_test

def get_loaders(dataset, train_transforms, test_transforms, train_val_test_split=(0.60, 0.20, 0.20), batch_size=None, sampler=None):
    if dataset == 'bcdr':
        df = get_bcdr_subjects(configs.BCDR_PATH, configs.BCDR_INFO_CSV, configs.BCDR_OUTLINES_CSV, configs.TASK)
        # Print number of unique patient_ids in the data frame
        print(f'BCDR ---- TOTAL NUMBER OF SAMPLES: {df.shape[0]}')
        df_train, df_val, df_test = unique_group_train_val_test_split(df, train_val_test_split)
        # randomly drop half of the samples from df_train
        df_train = df_train.sample(frac=0.0625, random_state=configs.RANDOM_SEED)
        print(f'SPLITS ---- {df_train.shape[0]} TRAINING, {df_val.shape[0]} VALIDATION, {df_test.shape[0]} TESTING')
        train_dataset = BCDRLoader(df_train, transform=train_transforms)
        val_dataset = BCDRLoader(df_val, transform=test_transforms)
        test_dataset = BCDRLoader(df_test, transform=test_transforms)
    elif dataset == 'inbreast':
        # df = get_inbreast_subjects(configs.INBREAST_INFO_FILE, configs.TASK)
        # TODO remove the hard coded TASK
        df = get_inbreast_subjects(configs.INBREAST_INFO_FILE, Task.SEGMENTATION)  # for now, to keep the mass mask
        print(f'INBREAST ---- TOTAL NUMBER OF SAMPLES: {df.shape[0]}')
        if train_val_test_split[0] == 0. and train_val_test_split[1] == 0:
            print('ALL SAMPLES FOR TESTING!')
            test_dataset = INBreastDataset(df, configs.INBREAST_IMAGES_DIR, transform=test_transforms)
            test_loader = DataLoader(test_dataset, batch_size=configs.TEST_BATCH_SIZE, shuffle=False, num_workers=configs.NUM_WORKERS)
            return None, None, test_loader
        df_train, df_val, df_test = unique_group_train_val_test_split(df, train_val_test_split)
        
        print(f'SPLITS ---- {df_train.shape[0]} TRAINING, {df_val.shape[0]} VALIDATION, {df_test.shape[0]} TESTING')
        train_dataset = INBreastDataset(df_train, configs.INBREAST_IMAGES_DIR, transform=train_transforms)
        val_dataset = INBreastDataset(df_val, configs.INBREAST_IMAGES_DIR, transform=test_transforms)
        test_dataset = INBreastDataset(df_test, configs.INBREAST_IMAGES_DIR, transform=test_transforms)
    elif dataset == 'optimam':
        df = get_optimam_subjects(configs.OPTIMAM_INFO_FILE)
        # Print number of unique patient_ids in the data frame
        print(f'OPTIMAM ---- TOTAL NUMBER OF SAMPLES: {df.shape[0]}')
        df_train, df_val, df_test = unique_group_train_val_test_split(df, train_val_test_split)
        print(f'SPLITS ---- {df_train.shape[0]} TRAINING, {df_val.shape[0]} VALIDATION, {df_test.shape[0]} TESTING')
        print(f"TRAINING (NORMAL, MALIGNANT): {df_train[df_train.status == 'Normal'].shape[0]}, {df_train[df_train.status == 'Malignant'].shape[0]}")
        print(f"VALIDATION (NORMAL, MALIGNANT): {df_val[df_val.status == 'Normal'].shape[0]}, {df_val[df_val.status == 'Malignant'].shape[0]}")
        print(f"TESTING (NORMAL, MALIGNANT): {df_test[df_test.status == 'Normal'].shape[0]}, {df_test[df_test.status == 'Malignant'].shape[0]}")
        # Select the same amount of normal as malignant samples for training
        g = df_train.groupby('status')
        df_train = g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))
        g = df_val.groupby('status')
        df_val = g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))
        print(f"TRAINING AFTER BALANCING (NORMAL, MALIGNANT): {df_train[df_train.status == 'Normal'].shape[0]}, {df_train[df_train.status == 'Malignant'].shape[0]}")
        print(f"VALIDATION AFTER BALANCING (NORMAL, MALIGNANT): {df_val[df_val.status == 'Normal'].shape[0]}, {df_val[df_val.status == 'Malignant'].shape[0]}")
        train_dataset = OPTIMAMDatasetLoader(df_train, transform=train_transforms)
        val_dataset = OPTIMAMDatasetLoader(df_val, transform=test_transforms)
        test_dataset = OPTIMAMDatasetLoader(df_test, transform=test_transforms)
    elif dataset == 'mnm':
        # train_subjects = get_mnm_subjects(self.configs, self.configs.MnM_TRAIN_FOLDER, self.configs.MnM_INFO_FILE)
        # val_subjects = get_mnm_subjects(self.configs, self.configs.MnM_VALIDATION_FOLDER, self.configs.MnM_INFO_FILE)
        train_loader, val_loader = None, None
        test_subjects = get_mnm_subjects(configs, configs.MnM_TEST_FOLDER, configs.MnM_INFO_FILE)
        # print(f'MnM ---- TOTAL NUMBER OF SAMPLES: {len(train_subjects) + len(val_subjects) + len(test_subjects)}')
        # print(f'SPLITS ---- {len(train_subjects)} TRAINING, {len(val_subjects)} VALIDATION, {len(test_subjects)} TESTING')
        # train_dataset = tio.SubjectsDataset(train_subjects, transform=train_transforms)
        # val_dataset = tio.SubjectsDataset(val_subjects, transform=test_transforms)
        test_dataset = tio.SubjectsDataset(test_subjects, transform=test_transforms)
        sampler = tio.sampler.UniformSampler((128, 128, 1))
        # train_loader = MnMDataLoader(train_dataset, batch_size=configs.BATCH_SIZE, max_length=2000, samples_per_volume=5, sampler=sampler, shuffle=True, num_workers=configs.NUM_WORKERS)
        # val_loader = MnMDataLoader(val_dataset, batch_size=configs.VAL_BATCH_SIZE, max_length=2000, samples_per_volume=5, sampler=sampler, shuffle=False, num_workers=configs.NUM_WORKERS)
        test_loader = MnMDataLoader(test_dataset, batch_size=configs.TEST_BATCH_SIZE, max_length=2000, samples_per_volume=10, sampler=sampler, shuffle=False, num_workers=configs.NUM_WORKERS)
        return train_loader, val_loader, test_dataset
    else:
        raise ValueError(f'Unknown dataset: {dataset}')
    train_batch_size = batch_size if batch_size is not None else configs.BATCH_SIZE

    collator = VariableBatchCollator(configs.TASK)

    if sampler is not None:
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False, num_workers=configs.NUM_WORKERS, sampler=sampler, collate_fn=collator)
    else: 
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=configs.NUM_WORKERS, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=configs.VAL_BATCH_SIZE, shuffle=False, num_workers=configs.NUM_WORKERS, collate_fn=collator)
    test_loader = DataLoader(test_dataset, batch_size=configs.TEST_BATCH_SIZE, shuffle=False, num_workers=configs.NUM_WORKERS, collate_fn=collator)

    return train_loader, val_loader, test_loader

class VariableBatchCollator(object):
    def __init__(self, task):
        self.task = task
    def __call__(self, batch):
        '''Collate function for batches with images of different sizes.
        Arguments
            :list batch: list of tuples (data, target)
        Returns
            :tuple (data, target)
        '''
        if self.task == Task.SEGMENTATION:
            data = [item['scan']['data'] for item in batch]
            target = [item['mask']['data'] for item in batch]
        elif self.task == Task.CLASSIFICATION:
            data = [item['scan']['data'] for item in batch]
            target = [item['status'] for item in batch]
            target = torch.LongTensor(target)
        elif self.task == Task.REGRESSION:
            raise NotImplementedError
        else:
            raise ValueError(f'Unknown task: {self.task}')
        return [data, target]