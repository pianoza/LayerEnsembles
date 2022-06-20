from pathlib import Path
import torch
import multiprocessing
from utils import Task, Organ
# CLASSIFICATION LABELS: 1->Malignant; 0->Benign
RANDOM_SEED = 13

TEST_T = 10
TASK = Task.CLASSIFICATION
ORGAN = Organ.BREAST

# EXPERIMENT_NAME = 'Inbreast_TB_GRAPH'  # run it once to build and show the network graph in tensorboard
# EXPERIMENT_NAME = 'BCDR_classification_test'
# EXPERIMENT_NAME = 'Inbreast_classification_test_full'
# EXPERIMENT_NAME = 'BCDR_classification_test_full'
# EXPERIMENT_NAME = 'Inbreast_classification_test_full_cropped_to_breast_TEST'
# EXPERIMENT_NAME = 'Inbreast_TEST_1'  # inside breast area (intensity percentile (0.5, 0.95), min-max normalization), cropped to breast, downsampled 4 times, best model by f1-score
# EXPERIMENT_NAME = 'Inbreast_TEST_2'  # inside breast area (intensity percentile (0.5, 0.95), min-max normalization), resized to 256x256, best model by f1-score
# EXPERIMENT_NAME = 'Inbreast_TEST_3'  # inside breast area (intensity percentile (0.5, 0.95), min-max normalization), resized to 256x256, best model by f1-score, classification_head(1x(conv+relu+bn)+as usual)
# EXPERIMENT_NAME = 'Inbreast_TEST_4'  # inside breast area (intensity percentile (0.5, 0.95), min-max normalization), resized to 256x256, best model by f1-score, classification_head(1x(conv+relu+bn)+as usual+Dropout0.5)
# EXPERIMENT_NAME = 'Inbreast_TEST_5'  # inside breast area (intensity percentile (0.5, 0.95), min-max normalization), resized to 256x256, best model by f1-score, classification_head(2x(BasicBlock)+as usual)
EXPERIMENT_NAME = 'Inbreast_TEST_6'  # inside breast area (intensity percentile (0.5, 0.95), min-max normalization), cropped to breast, downsampled 4 times, best model by f1-score, more heads (all .relu)
# EXPERIMENT_NAME = 'OPTIMAM_TEST_1'  # inside breast area (intensity percentile (0.5, 0.95), min-max normalization), cropped to breast, downsampled 4 * 2 times, best model by f1-score, classification_head(2x(BasicBlock)+as usual), 30 epochs, from scratch, lr0.00001
# EXPERIMENT_NAME = 'OPTIMAM_TEST_2'  # inside breast area (intensity percentile (0.5, 0.95), min-max normalization), cropped to breast, downsampled 4 * 2 times, best model by f1-score, classification_head(2x(BasicBlock)+as usual), 100 epochs, from scratch
# EXPERIMENT_NAME = 'OPTIMAM_TEST_3'  # inside breast area (intensity percentile (0.5, 0.95), min-max normalization), cropped to breast, downsampled 4 times, best model by f1-score, classification_head(2x(BasicBlock)+as usual), 100 epochs, from scratch, no lr decay
# EXPERIMENT_NAME = 'OPTIMAM_TEST_4'  # inside breast area (intensity percentile (0.5, 0.95), min-max normalization), cropped to breast, downsampled 4 times, best model by f1-score, classification_head(2x(BasicBlock)+as usual), 100 epochs, from scratch
print(f'RUNNING EXPERIMENT {EXPERIMENT_NAME}')

# TODO
'''
1) 
Plot likelihood vs layer 
x-axis -> layer number
y-axis -> likelihood (predicted probability of the class)
2) make a more continuous change: add more intermediate layers 2x(basicblock(2(conv+relu+bn))+maxpool)
3) add more layer to the network and skip early layers
4) AULA like metric by comparing the probabilities of each classification head
'''

if 'BCDR' in EXPERIMENT_NAME:
    DATASET = 'bcdr'
elif 'Inbreast' in EXPERIMENT_NAME:
    DATASET = 'inbreast'
elif 'OPTIMAM' in EXPERIMENT_NAME:
    DATASET = 'optimam'
elif 'MnM' in EXPERIMENT_NAME:
    DATASET = 'mnm'
else:
    raise ValueError(f'Unknown dataset in Experiment {EXPERIMENT_NAME}')

ROOT_PATH = Path('/home/kaisar')
SOURCE_CODE_PATH = ROOT_PATH / 'EuCanImage/Coding/LayerEnsembles'

DEVICE = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
print('WILL RUN ON DEVICE:', DEVICE)

NUM_WORKERS = multiprocessing.cpu_count()
print('NUMBER OF CPU WORKERS:', NUM_WORKERS)
NUM_ENSEMBLES = 5
LR = 0.0001
LR_MIN = 0.000001
LR_DECAY_FACTOR = 0.5
TENSORBOARD = True
OVERWRITE = False
TENSORBOARD_ROOT = SOURCE_CODE_PATH / 'tensorboard'
PLOT_VALIDATION_FREQUENCY = 1
RESULTS_PATH = SOURCE_CODE_PATH / 'results'

DROPOUT_RATE = 0.0
NUM_EPOCHS = 100  # 200
EARLY_STOPPING_PATIENCE = 200
SCHEDULER_PATIENCE = 100  # NOTE same as num epochs, never reduces lr 
BATCH_SIZE = 5
VAL_BATCH_SIZE = 2 * BATCH_SIZE
TEST_BATCH_SIZE = 1  #BATCH_SIZE  # set to 1 because of GradCAM TODO change this to BATCH_SIZE

# DATASETS

INBREAST_DATASET_DIR = ROOT_PATH / 'Datasets/InBreast'
INBREAST_IMAGES_DIR = INBREAST_DATASET_DIR / 'AllPatients'
INBREAST_INFO_FILE = INBREAST_DATASET_DIR / 'INbreast_updated.csv'

BCDR_DATASET_DIR = ROOT_PATH / 'Datasets/BCDR/Processed/BCDR'

BCDR_PATH =  [BCDR_DATASET_DIR / 'BCDR-D01_dataset',
              BCDR_DATASET_DIR / 'BCDR-D02_dataset',
              BCDR_DATASET_DIR / 'BCDR-F01_dataset',
              BCDR_DATASET_DIR / 'BCDR-F02_dataset',
              BCDR_DATASET_DIR / 'BCDR-F03_dataset/BCDR-F03']
BCDR_INFO_CSV = ['None',
                 'None',
                 'None',
                 'None',
                 'None']
BCDR_OUTLINES_CSV =  [BCDR_DATASET_DIR / 'BCDR-D01_dataset/bcdr_d01_outlines.csv',
                      BCDR_DATASET_DIR / 'BCDR-D02_dataset/bcdr_d02_outlines.csv',
                      BCDR_DATASET_DIR / 'BCDR-F01_dataset/bcdr_f01_outlines.csv',
                      BCDR_DATASET_DIR / 'BCDR-F02_dataset/bcdr_f02_outlines.csv',
                      BCDR_DATASET_DIR / 'BCDR-F03_dataset/BCDR-F03/bcdr_f03_outlines.csv']

# PROCESSED OPTIMAM MASKS
# OPTIMAM_INFO_FILE = '/home/kaisar/Datasets/OPTIMAM/optimam_info_fixed.csv'
OPTIMAM_INFO_FILE = '/datasets/OPTIMAM/png_screening_cropped_fixed/healthy_nonhealthyMass.csv'
# OPTIMAM_DATASET_PATH = '/datasets/OPTIMAM/png_screening_cropped_fixed/'

# MnM Challenge Dataset
MnM_DATA_PATH = Path('/home/kaisar/Datasets/MnM/Processed')
MnM_TRAIN_FOLDER = MnM_DATA_PATH / 'Training'
MnM_VALIDATION_FOLDER = MnM_DATA_PATH / 'Validation'
MnM_TEST_FOLDER = MnM_DATA_PATH / 'Testing'
MnM_INFO_FILE = MnM_DATA_PATH.parent / '201014_M&Ms_Dataset_Information_-_opendataset.csv'

# csv columns:
MnM_CODE = 'External code'
MnM_VENDOR_NAME = 'VendorName'
MnM_VENDOR = 'Vendor'
MnM_CENTRE = 'Centre'
MnM_ED = 'ED'
MnM_ES = 'ES'

# working with the pre-processed data
MnM_ED_SUFFIX = '_ed.nii.gz'
MnM_ED_GT_SUFFIX = '_ed_gt.nii.gz'
MnM_ES_SUFFIX = '_es.nii.gz'
MnM_ES_GT_SUFFIX = '_es_gt.nii.gz'