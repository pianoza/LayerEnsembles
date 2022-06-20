import os
import numpy as np
from PIL import Image
from mmg_detection_datasets import *
import torch

if __name__ == '__main__':

    # Cropped scans GPU Server
    info_csv='/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/client_images_screening.csv'
    dataset_path='/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/images'

    cropped_to_breast = True
    
    detection = True
    load_max = 10 #10 Only loads 10 images
    pathologies = ['mass'] #['mass', 'calcifications', 'suspicious_calcifications', 'architectural_distortion'] # None to select all
    status = ['Normal', 'Benign', 'Malignant'] #['Normal'] 
    plot_images = False

    # Call to the OPTIMAM Dataloader
    optimam_clients = OPTIMAMDataset(info_csv, dataset_path,detection=detection, load_max=load_max, 
                            cropped_to_breast=cropped_to_breast)
    
    # If we don't select clients by pathology, status or site, the loop will be:
    # for client in optimam_clients:
    #     for study in client.studies:
    #         for image in study.images:

    clients_selected = optimam_clients.get_clients_by_pathology_and_status(pathologies)
    image_ctr = 0
    for client in clients_selected:
        for study in client.studies:
            for image in study:
                status = image.status # ['Benign', 'Malignant', 'Normal']
                site = image.site # ['adde', 'jarv', 'stge']
                manufacturer = image.manufacturer # ['HOLOGIC, Inc.', 'Philips Digital Mammography Sweden AB', 'GE MEDICAL SYSTEMS', 'Philips Medical Systems', 'SIEMENS']
                # view = image.view # MLO_VIEW = ['MLO','LMLO','RMLO', 'LMO', 'ML'] CC_VIEW = ['CC','LCC','RCC', 'XCCL', 'XCCM']
                # laterality = image.laterality # L R
                               # Get GT Bboxes
                gt_bboxes = []
                for annotation in image.annotations:
                    bbox_anno = annotation.get_bbox(fit_to_breast=True)
                    xmin, xmax, ymin, ymax = bbox_anno.xmin, bbox_anno.xmax, bbox_anno.ymin, bbox_anno.ymax
                    image_ctr += 1
    print(f'Total number of masses {image_ctr}')
