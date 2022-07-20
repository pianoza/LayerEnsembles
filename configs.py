from pathlib import Path
import torch
import multiprocessing
from utils import Task, Organ
# CLASSIFICATION LABELS: 1->Malignant; 0->Benign
RANDOM_SEED = 13

SKIP_FIRST_T = 3  # 0 -> no skipping
TASK = Task.SEGMENTATION  # Task.CLASSIFICATION | Task.REGRESSION | Task.SEGMENTATION
# ORGAN = Organ.BREAST  # Organ.BREAST | Organ.HEART
IS_LAYER_ENSEMBLES = False
SAVE_SEGMENTATION_OUTPUTS = False  # if False, only evaluation metrics are saved

# EXPERIMENT_NAME = 'BCDR_segmentation_test'
# EXPERIMENT_NAME = 'BCDR_LE_segmentation_test'
# EXPERIMENT_NAME = 'OPTIMAM_classifcation_test'
# EXPERIMENT_NAME = 'Inbreast_classification_test'
# EXPERIMENT_NAME = 'BCDR_seg_full_set'  # 576 samples
# EXPERIMENT_NAME = 'BCDR_seg_half_set'  # 288 samples
# EXPERIMENT_NAME = 'BCDR_seg_quarter_set'  # 144 samples
# EXPERIMENT_NAME = 'BCDR_seg_quarter_of_quarter_set'  # 36 samples
EXPERIMENT_NAME = 'Plain_MnM_seg_test'

print(f'RUNNING EXPERIMENT {EXPERIMENT_NAME}')

if 'BCDR' in EXPERIMENT_NAME:
    DATASET = 'bcdr'
    ORGAN = Organ.BREAST  # Organ.BREAST | Organ.HEART
elif 'Inbreast' in EXPERIMENT_NAME:
    DATASET = 'inbreast'
    ORGAN = Organ.BREAST  # Organ.BREAST | Organ.HEART
elif 'OPTIMAM' in EXPERIMENT_NAME:
    DATASET = 'optimam'
    ORGAN = Organ.BREAST  # Organ.BREAST | Organ.HEART
elif 'MnM' in EXPERIMENT_NAME:
    DATASET = 'mnm'
    ORGAN = Organ.HEART  # Organ.BREAST | Organ.HEART
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
PLOT_VALIDATION_FREQUENCY = 10
RESULTS_PATH = SOURCE_CODE_PATH / 'results'

DROPOUT_RATE = 0.0
NUM_EPOCHS = 30  # 200
EARLY_STOPPING_PATIENCE = 200
SCHEDULER_PATIENCE = 100  # NOTE same as num epochs, never reduces lr 
BATCH_SIZE = 5
VAL_BATCH_SIZE = 2 * BATCH_SIZE
TEST_BATCH_SIZE = 2 * BATCH_SIZE  # set to 1 because of GradCAM TODO change this to BATCH_SIZE

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