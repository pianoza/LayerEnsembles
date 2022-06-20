import os
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib.pyplot as plt

from data.resize_image import *
from data.common_classes_mmg import *

# OPTIMAM
WRONG_BBOXES_OPTIMAM = ['1.2.826.0.1.3680043.9.3218.1.1.21149576.1882.1552598184295.155.0',
                        '1.2.826.0.1.3680043.9.3218.1.1.423368237.8725.1541832007115.79.0',
                        '1.2.826.0.1.3680043.9.3218.1.1.23278785.1708.1544221170622.213.0',
                        '1.2.826.0.1.3680043.9.3218.1.1.3010695.7588.1512129590473.9978.0',
                        '1.2.826.0.1.3680043.9.3218.1.1.242205684.1858.1540020381577.17.0',
                        '1.2.826.0.1.3680043.9.3218.1.1.3370786.8916.1512165599585.7373.0',
                        '1.2.826.0.1.3680043.9.3218.1.1.3370786.8916.1512165599585.7403.0',
                        '1.2.826.0.1.3680043.9.3218.1.1.3860076.2252.1511393629935.6737.0',
                        '1.2.826.0.1.3680043.9.3218.1.1.296436949.3318.1540562694227.83.0'] # X and Y bbox flipped?

MLO_VIEWS_OPTIMAM = ['MLO','LMLO','RMLO']
CC_VIEWS_OPTIMAM = ['CC','LCC','RCC']
EXTRA_VIEWS_OPTIMAM = ['LMO', 'ML', 'XCCL', 'XCCM']
# XCCL, XCCM, LMO and ML are extra views
# CCID is an extra view for calibration when implant
# View 'FLATFIELD' is raw (Discarding)
# lesion_id: unique number of a lesion between views
# 1,2: 2 annotations present in the scan view
# 1,3 and 3,2: 3 annotations present in the scan view
# UNLINKED: No confirmed correspondency between annotations
MANUFACTURERS_OPTIMAM = ['HOLOGIC, Inc.', 'Philips Digital Mammography Sweden AB', 'GE MEDICAL SYSTEMS', 'Philips Medical Systems', 'SIEMENS']
CONSPICUITY_VALUES_OPTIMAM = ['Obvious','Occult','Subtle','Very_subtle']
PATHOLOGIES_OPTIMAM = ['mass', 'calcifications', 'suspicious_calcifications',
                'architectural_distortion', 'focal_asymmetry']
SHAPE_VALUES_OPTIMAM = ['spiculated', 'ill_defined', 'well_defined'] #included sometimes in pathologies field
STATUS_VALUES_OPTIMAM = ['Benign', 'Malignant', 'Interval Cancer', 'Normal']
SITES_OPTIMAM = ['adde', 'jarv', 'stge']

# INBREAST
MLO_VIEWS_INBREAST = ['MLO']
CC_VIEWS_INBREAST = ['CC']
PATHOLOGIES_INBREAST = ['mass']
MANUFACTURERS_INBREAST = ['MammoNovation Siemens FFDM']
SITES_INBREAST = ['Breast Centre (CHSJ Porto)']
PIXEL_SIZE_INBREAST = [0.07, 0.07] #mm
RESOLUTION_INBREAST = 2e14 #14 bit contrast
IMAGE_MATRICES_INBREAST = [(4084, 3328), (3328, 2560)] # height, width
BIRAD_DESCRIPTIONS_INBREAST = {0: 'Needs additional imaging evaluation',
                                1: 'Negative', 
                                2: 'Benign findings',
                                3: 'Probably benign findings. Short interval follow-up.',
                                4: 'Suspicious anomaly. Biopsy should be considered.',
                                5: 'Highly suggestive of malignancy', 
                                6: 'Biopsy proven malignancy'}
ACR_DESCRIPTIONS_INBREAST = {1: 'Fatty', 
                                2: 'Scattered',
                                3: 'Heterogeneously dense',
                                4: 'Extremely dense'}
# BCDR 
# 723 female and 1 male
MLO_VIEWS_BCDR = ['MLO']
CC_VIEWS_BCDR = ['CC']
PATHOLOGIES_BCDR = ['nodule', 'calcification', 'microcalcification', 'axillary_adenopathy',
                    'architectural_distortion', 'stroma_distortion']
MANUFACTURERS_BCDR = [''] #TODO Info not found
SITES_BCDR = ['Centro Hospitalar São João at University of Porto (FMUP-HSJ)']
PIXEL_SIZE_BCDR = [0.07, 0.07] #mm # INFO not verified
RESOLUTION_BCDR = 2e14 #14 bit resultion TIFFs (TIF files: 8bit)
IMAGE_MATRICES_BCDR = [(4084, 3328), (3328, 2560), (3072, 2816)] # height, width
STATUS_BCDR = ['Normal', 'Benign', 'Malign'] #biopsy proven
ACR_DESCRIPTIONS_BCDR = {1: 'Fatty', 
                        2: 'Scattered',
                        3: 'Heterogeneously dense',
                        4: 'Extremely dense'} # BI-RADS standard

class OPTIMAMAnnotation(AnnotationMMG):
    def __init__(self, lesion_id, mark_id):
        self.lesion_id = lesion_id
        self.mark_id = int(mark_id)
        self.conspicuity = None
        
    @property
    def conspicuity(self):
        return self._conspicuity

    @conspicuity.setter
    def conspicuity(self, conspicuity):
        if conspicuity in CONSPICUITY_VALUES_OPTIMAM:
            self._conspicuity = conspicuity

class OPTIMAMImage(ImageMMG):
    def __init__(self, scan_path):
        self.serie_id = None
        super().__init__(scan_path)

    def generate_COCO_dict(self, image_id, obj_count, pathologies=None, fit_to_breast=False,
                            category_id_dict={'Benign_mass': 0, 'Malignant_mass': 1},
                            use_status=False):
        if not self.height:
            self.height, self.width = np.array(Image.open(self.path)).shape[:2]
        annotations_elem = []
        img_elem = dict(
                file_name = self.path,
                height=self.height,
                width=self.width,
                id=image_id)
        for annotation in self.annotations:
            if pathologies is None:
                add_annotation = True
            elif any(item in annotation.pathologies for item in pathologies):
                add_annotation = True
            else:
                add_annotation = False
            if add_annotation:
                if use_status:
                    if pathologies is not None:
                        if annotation.status + '_' + pathologies[0] in category_id_dict.keys():
                            category_id = category_id_dict[annotation.status + '_' + pathologies[0]]
                        else:
                            break
                    else:
                        if annotation.status + '_pathology' in category_id_dict.keys():
                            category_id = category_id_dict[annotation.status + '_pathology']
                        else:
                            break
                else:
                    if 'mass' in annotation.pathologies and 'mass' in category_id_dict.keys():
                        pathology_selected = 'mass'
                    elif ('calcifications' in annotation.pathologies or 'suspicious_calcifications' in annotation.pathologies) and 'calcification' in category_id_dict.keys():
                        pathology_selected = 'calcification'
                    elif 'architectural_distortion' in annotation.pathologies and 'distortion' in category_id_dict.keys():
                        pathology_selected = 'distortion'
                    else:
                        continue
                    category_id = category_id_dict[pathology_selected]
                bbox_anno = annotation.get_bbox(fit_to_breast)
                xmin, xmax, ymin, ymax = bbox_anno.xmin, bbox_anno.xmax, bbox_anno.ymin, bbox_anno.ymax
                #poly = [xmin, ymin],[xmax, ymin],[xmax, ymax],[xmin, ymax]
                annot_elem = dict(
                    image_id=image_id,
                    id=obj_count,
                    bbox=[xmin, ymin, xmax - xmin, ymax - ymin],
                    area=(xmax - xmin)*(ymax - ymin),
                    segmentation=[[xmin, ymin, int((xmax-xmin)/2) + xmin, ymin, \
                       xmax, ymin, xmax, int((ymax-ymin)/2) + ymin, \
                       xmax, ymax, int((xmax-xmin)/2) + xmin, ymax, \
                       xmin, ymax, xmin, int((ymax-ymin)/2) + ymin]],
                    category_id=category_id,
                    iscrowd=0
                    )
                annotations_elem.append(annot_elem)
                obj_count += 1 # each bbox increases the counter (here only one per image)
                if False:
                    #poly = [xmin, ymin],[xmax, ymin],[xmax, ymax],[xmin, ymax]
                    poly = [xmin, ymin], [int((xmax-xmin)/2) + xmin, ymin], \
                       [xmax, ymin], [xmax, int((ymax-ymin)/2) + ymin], \
                       [xmax, ymax], [int((xmax-xmin)/2) + xmin, ymax], \
                       [xmin, ymax], [xmin, int((ymax-ymin)/2) + ymin]
                    img = cv2.cvtColor(np.array(Image.open(self.path).convert('RGB')), cv2.COLOR_BGR2RGB)
                    cv2.polylines(img, [np.array(poly)], True, (0, 255, 0), 2)
                    plt.figure()
                    plt.imshow(img)
                    #plt.imsave('./test.png', img)
                    plt.show()
                    # img_pil = self.open_pil(print_annotations=True)
                    # plot_image_opencv_fit_window(img_pil, title='OPTIMAM Scan', 
                    #                          screen_resolution=(1920, 1080), wait_key=True)
        return img_elem, annotations_elem, obj_count

    def generate_COCO_dict_high_density(self, data_path, data_info, image_id, obj_count, pathologies=None, fit_to_breast=False,
                            category_id_dict={'Benign_mass': 0, 'Malignant_mass': 1},
                            use_status=False):
        # Read data augmentation info
        info = pd.read_csv(data_info, index_col=False)
        image_info = info.loc[info['image_id'] == self.id]
        h_scale = image_info.h_scale.values[0]
        w_scale = image_info.w_scale.values[0]
        # Read image
        img_path = os.path.join(data_path, self.id + '.png')
        img_np = np.array(Image.open(img_path))
        height, width = img_np.shape[:2]
        annotations_elem = []
        img_elem = dict(
                file_name = img_path,
                height=height,
                width=width,
                id=image_id)
        for annotation in self.annotations:
            if pathologies is None:
                add_annotation = True
            elif any(item in annotation.pathologies for item in pathologies):
                add_annotation = True
            else:
                add_annotation = False
            if add_annotation:
                if use_status:
                    if pathologies is not None:
                        if annotation.status + '_' + pathologies[0] in category_id_dict.keys():
                            category_id = category_id_dict[annotation.status + '_' + pathologies[0]]
                        else:
                            break
                    else:
                        if annotation.status + '_pathology' in category_id_dict.keys():
                            category_id = category_id_dict[annotation.status + '_pathology']
                        else:
                            break
                else:
                    if 'mass' in annotation.pathologies and 'mass' in category_id_dict.keys():
                        pathology_selected = 'mass'
                    elif ('calcifications' in annotation.pathologies or 'suspicious_calcifications' in annotation.pathologies) and 'calcification' in category_id_dict.keys():
                        pathology_selected = 'calcification'
                    elif 'architectural_distortion' in annotation.pathologies and 'distortion' in category_id_dict.keys():
                        pathology_selected = 'distortion'
                    else:
                        continue
                    category_id = category_id_dict[pathology_selected]
                bbox_anno = annotation.get_bbox(fit_to_breast)
                xmin, xmax, ymin, ymax = bbox_anno.xmin, bbox_anno.xmax, bbox_anno.ymin, bbox_anno.ymax
                # Rescale the bboxes
                xmin, xmax = int(xmin * w_scale), int(xmax * w_scale)
                ymin, ymax = int(ymin * h_scale), int(ymax * h_scale)

                #poly = [xmin, ymin],[xmax, ymin],[xmax, ymax],[xmin, ymax]
                annot_elem = dict(
                    image_id=image_id,
                    id=obj_count,
                    bbox=[xmin, ymin, xmax - xmin, ymax - ymin],
                    area=(xmax - xmin)*(ymax - ymin),
                    segmentation=[[xmin, ymin, int((xmax-xmin)/2) + xmin, ymin, \
                       xmax, ymin, xmax, int((ymax-ymin)/2) + ymin, \
                       xmax, ymax, int((xmax-xmin)/2) + xmin, ymax, \
                       xmin, ymax, xmin, int((ymax-ymin)/2) + ymin]],
                    category_id=category_id,
                    iscrowd=0
                    )
                annotations_elem.append(annot_elem)
                obj_count += 1 # each bbox increases the counter (here only one per image)
                if False:
                    #poly = [xmin, ymin],[xmax, ymin],[xmax, ymax],[xmin, ymax]
                    poly = [xmin, ymin], [int((xmax-xmin)/2) + xmin, ymin], \
                       [xmax, ymin], [xmax, int((ymax-ymin)/2) + ymin], \
                       [xmax, ymax], [int((xmax-xmin)/2) + xmin, ymax], \
                       [xmin, ymax], [xmin, int((ymax-ymin)/2) + ymin]
                    img = cv2.cvtColor(np.array(Image.open(self.path).convert('RGB')), cv2.COLOR_BGR2RGB)
                    cv2.polylines(img, [np.array(poly)], True, (0, 255, 0), 2)
                    plt.figure()
                    plt.imshow(img)
                    plt.show()
                    # img_pil = self.open_pil(print_annotations=True)
                    # plot_image_opencv_fit_window(img_pil, title='OPTIMAM Scan', 
                    #                          screen_resolution=(1920, 1080), wait_key=True)
        return img_elem, annotations_elem, obj_count

class OPTIMAMDataset(DatasetMMG):

    def __init__(self, info_csv:str, dataset_path:str, cropped_to_breast=False, 
                client_ids=None, image_ids=None, load_max=-1, detection=False):
        
        super().__init__(info_csv, dataset_path, cropped_to_breast, 
                         client_ids, image_ids, load_max)

        self.clients_dict = {}

        def get_client(client_id, site):
            if client_id in self.clients_dict.keys():
                index = self.clients_dict[client_id]
                return self.clients[index]
            new_client = ClientMMG(client_id)
            new_client.site = site
            return new_client

        def update_client(client:ClientMMG):
            if len(self.clients):
                if client.id in self.clients_dict.keys():
                    index = self.clients_dict[client.id]
                    self.clients[index] = client
                    return
            self.clients.append(client)
            self.clients_dict[client.id] = len(self.clients) - 1

        info = pd.read_csv(info_csv)
        info = info.astype(object).replace(np.nan, '')

        if detection:
            df_bbox = info.loc[info['x1'] != '']
            unique_study_id_df = df_bbox.groupby(['study_id'], as_index=False)
        else:
            unique_study_id_df = info.groupby(['study_id'], as_index=False)
        for study_name, study_group in unique_study_id_df:
            if client_ids:
                if study_group.client_id.values[0] not in client_ids:
                    continue
            unique_image_id_df = study_group.groupby(['image_id'], as_index=False)
            row_client = get_client(study_group.client_id.values[0], study_group.site.values[0])
            bool_update_client = False
            for image_name, image_group in unique_image_id_df:
                if image_ids:
                    if image_name not in image_ids:
                        continue
                    if image_name in WRONG_BBOXES_OPTIMAM:
                        print(f'Skipping {image_name}')
                        continue
                # Create new image
                scan_path = os.path.join(dataset_path, image_group.client_id.values[0])
                scan_path = os.path.join(scan_path, image_group.study_id.values[0])
                scan_path = os.path.join(scan_path, image_group.image_id.values[0] + '.png')
                # Create new OPTIMAMImage
                new_image = OPTIMAMImage(scan_path)
                view = image_group.view.values[0]
                valid_view =  True
                if view in CC_VIEWS_OPTIMAM:
                    new_image.view = 'CC'
                elif view in MLO_VIEWS_OPTIMAM:
                    new_image.view = 'MLO'
                elif view in EXTRA_VIEWS_OPTIMAM:
                    new_image.view = view
                else:
                    #print(f'Error: view {view} not found in list -> Discard image')
                    valid_view =  False
                if not valid_view:
                    continue
                init_image = True
                for idx_mark, image in enumerate(image_group.itertuples()):
                    if detection and image.x1 == '': # Check if Annotation is available
                        continue
                    status = image.status
                    if status == 'Interval Cancer':
                        status = 'Malignant'
                    if status == 'Normal':
                        stop = True
                    if init_image:
                        new_image.id = image.image_id
                        new_image.laterality = image.laterality
                        new_image.status = status
                        new_image.site = image.site
                        new_image.manufacturer = image.manufacturer
                        new_image.serie_id = image.serie_id
                        new_image.pixel_spacing = image.pixel_spacing.split()[0]
                        new_image.implant = image.implant
                        new_image.breast_bbox = BBox(image.xmin_cropped, image.xmax_cropped, image.ymin_cropped, image.ymax_cropped)
                        new_image.cropped_to_breast = cropped_to_breast
                        new_image.age = image.age
                        init_image = False
                    if image.x1:
                        new_annotation = OPTIMAMAnnotation(lesion_id=image.lesion_id,
                                                            mark_id=image.mark_id)
                        new_annotation.status = status
                        new_annotation.conspicuity = image.conspicuity
                        new_annotation.pathologies = image.pathologies.split(' ')
                        new_annotation.bbox = BBox(image.x1,image.x2,image.y1, image.y2)
                        new_annotation.breast_bbox = BBox(image.xmin_cropped, image.xmax_cropped, image.ymin_cropped, image.ymax_cropped)
                        new_image.add_annotation(new_annotation)
                        self.annotation_ctr += 1
                if not init_image:
                    # Update study
                    row_study = row_client.get_study(image.study_id)
                    if row_study is None:
                        row_study = StudyMMG(image.study_id)
                    row_study.add_image(new_image)
                    # Update client
                    row_client.update_study(row_study)
                    self.images_ctr += 1
                    bool_update_client = True
                    if self.images_ctr == self.load_max:
                        break
            if bool_update_client:
                update_client(row_client)
                if self.images_ctr == self.load_max:
                    break

class INBreastAnnotation(AnnotationMMG):
    def __init__(self, mark_id):
        self.mark_id = int(mark_id)
    
class INBreastImage(ImageMMG):
    def __init__(self, scan_path):
        self.breast_width = None
        self.breast_height = None
        super().__init__(scan_path)

    def generate_COCO_dict(self, image_id, obj_count, pathologies=None, fit_to_breast=False,
                            category_id_dict={'Benign_mass': 0, 'Malignant_mass': 1},
                            use_status=False):
        if not self.height:
            self.height, self.width = np.array(Image.open(self.path)).shape[:2]
        annotations_elem = []
        if fit_to_breast and self.breast_width:
            img_elem = dict(
                    file_name = self.path,
                    height=self.breast_height,
                    width=self.breast_width,
                    id=image_id)
        else:
            img_elem = dict(
                    file_name = self.path,
                    height=self.height,
                    width=self.width,
                    id=image_id)
        for annotation in self.annotations:
            if pathologies is None:
                add_annotation = True
            elif any(item in annotation.pathologies for item in pathologies):
                add_annotation = True
            else:
                add_annotation = False
            if add_annotation:
                if use_status:
                    if pathologies is not None:
                        if annotation.status + '_' + pathologies[0] in category_id_dict.keys():
                            category_id = category_id_dict[annotation.status + '_' + pathologies[0]]
                        else:
                            break
                    else:
                        if annotation.status + '_pathology' in category_id_dict.keys():
                            category_id = category_id_dict[annotation.status + '_pathology']
                        else:
                            break
                else:
                    if 'mass' in annotation.pathologies and 'mass' in category_id_dict.keys():
                        pathology_selected = 'mass'
                    elif ('calcifications' in annotation.pathologies or 'suspicious_calcifications' in annotation.pathologies) and 'calcification' in category_id_dict.keys():
                        pathology_selected = 'calcification'
                    elif 'architectural_distortion' in annotation.pathologies and 'distortion' in category_id_dict.keys():
                        pathology_selected = 'distortion'
                    else:
                        continue
                    category_id = category_id_dict[pathology_selected]
                bbox_anno = annotation.get_bbox()
                xmin, xmax, ymin, ymax = bbox_anno.xmin, bbox_anno.xmax, bbox_anno.ymin, bbox_anno.ymax
                #poly = [xmin, ymin],[xmax, ymin],[xmax, ymax],[xmin, ymax]
                annot_elem = dict(
                    image_id=image_id,
                    id=obj_count,
                    bbox=[xmin, ymin, xmax - xmin, ymax - ymin],
                    area=(xmax - xmin)*(ymax - ymin),
                    segmentation=[[xmin, ymin, int((xmax-xmin)/2) + xmin, ymin, \
                       xmax, ymin, xmax, int((ymax-ymin)/2) + ymin, \
                       xmax, ymax, int((xmax-xmin)/2) + xmin, ymax, \
                       xmin, ymax, xmin, int((ymax-ymin)/2) + ymin]],
                    category_id=category_id,
                    iscrowd=0
                    )
                annotations_elem.append(annot_elem)
                obj_count += 1 # each bbox increases the counter (here only one per image)
                if False:
                    #poly = [xmin, ymin],[xmax, ymin],[xmax, ymax],[xmin, ymax]
                    poly = [xmin, ymin], [int((xmax-xmin)/2) + xmin, ymin], \
                       [xmax, ymin], [xmax, int((ymax-ymin)/2) + ymin], \
                       [xmax, ymax], [int((xmax-xmin)/2) + xmin, ymax], \
                       [xmin, ymax], [xmin, int((ymax-ymin)/2) + ymin]
                    img = cv2.cvtColor(np.array(Image.open(self.path).convert('RGB')), cv2.COLOR_BGR2RGB)
                    cv2.polylines(img, [np.array(poly)], True, (0, 255, 0), 2)
                    plt.figure()
                    plt.imshow(img)
                    #plt.imsave('./test.png', img)
                    plt.show()
                    # img_pil = self.open_pil(print_annotations=True)
                    # plot_image_opencv_fit_window(img_pil, title='INBreast Scan', 
                    #                          screen_resolution=(1920, 1080), wait_key=True)
        return img_elem, annotations_elem, obj_count

class INBreastDataset(DatasetMMG):
    def __init__(self, info_csv:str, dataset_path:str, cropped_to_breast=False, 
                client_ids=None, image_ids=None, load_max=-1, detection=False):
        
        super().__init__(info_csv, dataset_path, cropped_to_breast, 
                         client_ids, image_ids, load_max)

        self.clients_dict = {}

        def get_client(client_id, site, breast_density):
            if client_id in self.clients_dict.keys():
                index = self.clients_dict[client_id]
                return self.clients[index]
            new_client = ClientMMG(client_id)
            new_client.site = site
            new_client.breast_density = breast_density
            return new_client

        def update_client(client:ClientMMG):
            if len(self.clients):
                if client.id in self.clients_dict.keys():
                    index = self.clients_dict[client.id]
                    self.clients[index] = client
                    return
            self.clients.append(client)
            self.clients_dict[client.id] = len(self.clients) - 1

        info = pd.read_csv(info_csv)
        info = info.astype(object).replace(np.nan, '')

        if detection:
            df_bbox = info.loc[info['mass_0_x2'] != '']
            unique_patient_id_df = df_bbox.groupby(['patient_id'], as_index=False)
        else:
            unique_patient_id_df = info.groupby(['patient_id'], as_index=False)
        for patient_id, patient_group in unique_patient_id_df:
            if client_ids:
                if patient_group.patient_id.values[0] not in client_ids:
                    continue
            unique_image_id_df = patient_group.groupby(['image_id'], as_index=False)
            row_client = get_client(patient_group.patient_id.values[0], SITES_INBREAST[0], patient_group.ACR.values[0])
            bool_update_client = False
            for image_name, image_group in unique_image_id_df:
                if image_ids:
                    if image_name not in image_ids:
                        continue
                # Create new image
                scan_png_path = image_group.scan_path.values[0].replace('nii.gz', 'png')
                scan_path = os.path.join(dataset_path, scan_png_path)
                # Create new INBreastImage
                new_image = INBreastImage(scan_path)
                view = image_group.view.values[0]
                valid_view =  True
                if view in CC_VIEWS_INBREAST:
                    new_image.view = 'CC'
                elif view in MLO_VIEWS_INBREAST:
                    new_image.view = 'MLO'
                else:
                    #print(f'Error: view {view} not found in list -> Discard image')
                    valid_view =  False
                if not valid_view:
                    continue
                init_image = True
                for idx_mark, image in enumerate(image_group.itertuples()):
                    if detection and image.mass_0_x2 == '': # Check if Annotation is available
                        continue
                    status = int(list(image.BIRADS)[0])
                    if status == 1:
                        status = 'Normal'
                    elif status in [2, 3]:
                        status = 'Benign'
                    elif status in [4, 5, 6]:
                        status = 'Malignant'
                    else:
                        print(f'Wrong BIRAD: {image.BIRADS} in image: {image.image_id}')
                        continue
                    if init_image:
                        new_image.id = image.image_id
                        new_image.laterality = image.laterality 
                        new_image.birad = image.BIRADS
                        new_image.status = status
                        new_image.site = SITES_INBREAST[0]
                        new_image.manufacturer = MANUFACTURERS_INBREAST[0]
                        new_image.pixel_spacing = PIXEL_SIZE_INBREAST[0]
                        new_image.implant = 'NO'
                        new_image.breast_bbox = BBox(image.breast_x1, image.breast_x2, image.breast_y1, image.breast_y2)
                        new_image.cropped_to_breast = cropped_to_breast
                        new_image.width = image.scan_width
                        new_image.height = image.scan_height
                        new_image.breast_width = image.breast_width
                        new_image.breast_height = image.breast_height
                        new_image.breast_density = image.ACR
                        init_image = False
                    if image.mass_0_x2:
                        new_annotation = INBreastAnnotation(mark_id=0)
                        new_annotation.status = status
                        new_annotation.pathologies = ['mass']
                        new_annotation.bbox = BBox(image.mass_0_x1, image.mass_0_x2,
                                                             image.mass_0_y1, image.mass_0_y2)
                        new_annotation.breast_bbox = BBox(image.breast_x1, image.breast_x2, image.breast_y1, image.breast_y2)
                        new_image.add_annotation(new_annotation)
                        self.annotation_ctr += 1
                    if image.mass_1_x2:
                        new_annotation = INBreastAnnotation(mark_id=1)
                        new_annotation.status = status
                        new_annotation.pathologies = ['mass']
                        new_annotation.bbox = BBox(image.mass_1_x1, image.mass_1_x2,
                                                             image.mass_1_y1, image.mass_1_y2)
                        new_annotation.breast_bbox = BBox(image.breast_x1, image.breast_x2, image.breast_y1, image.breast_y2)
                        new_image.add_annotation(new_annotation)
                        self.annotation_ctr += 1
                    if image.mass_2_x2:
                        new_annotation = INBreastAnnotation(mark_id=2)
                        new_annotation.status = status
                        new_annotation.pathologies = ['mass']
                        new_annotation.bbox = BBox(image.mass_2_x1, image.mass_2_x2,
                                                             image.mass_2_y1, image.mass_2_y2)
                        new_annotation.breast_bbox = BBox(image.breast_x1, image.breast_x2, image.breast_y1, image.breast_y2)
                        new_image.add_annotation(new_annotation)
                        self.annotation_ctr += 1
                if not init_image:
                    # Update study
                    row_study = row_client.get_study(1)
                    if row_study is None:
                        row_study = StudyMMG(1)
                    row_study.add_image(new_image)
                    # Update client
                    row_client.update_study(row_study)
                    self.images_ctr += 1
                    bool_update_client = True
                    if self.images_ctr == self.load_max:
                        break
            if bool_update_client:
                update_client(row_client)
                if self.images_ctr == self.load_max:
                    break

class BCDRAnnotation(AnnotationMMG):
    def __init__(self, lesion_id, segmentation_id):
        if lesion_id == '':
            lesion_id = 0
        self.lesion_id = int(lesion_id)
        self.segmentation_id = int(segmentation_id)
        self.segmentation = None

    def set_segmentation(self, lw_x_points, lw_y_points):
        # Read segmentation
        lw_x_points = list(map(int, lw_x_points.split(" ")[1:]))
        lw_y_points = list(map(int, lw_y_points.split(" ")[1:]))
        outline_lesion = []
        for i in range(len(lw_x_points)):
            outline_lesion.append([lw_x_points[i], lw_y_points[i]])
        # Close polygon
        outline_lesion.append([lw_x_points[0], lw_y_points[0]])
        self.segmentation = np.array(outline_lesion)

class BCDRImage(ImageMMG):
    def __init__(self, scan_path):
        self.breast_width = None
        self.breast_height = None
        super().__init__(scan_path)

    def generate_COCO_dict(self, image_id, obj_count, pathologies=None, fit_to_breast=False,
                            category_id_dict={'Benign_mass': 0, 'Malignant_mass': 1},
                            use_status=False):
        if not self.height:
            self.height, self.width = np.array(Image.open(self.path)).shape[:2]
        annotations_elem = []
        if fit_to_breast and self.breast_width:
            img_elem = dict(
                    file_name = self.path,
                    height=self.breast_height,
                    width=self.breast_width,
                    id=image_id)
        else:
            img_elem = dict(
                    file_name = self.path,
                    height=self.height,
                    width=self.width,
                    id=image_id)
        for annotation in self.annotations:
            if pathologies is None:
                add_annotation = True
            elif any(item in annotation.pathologies for item in pathologies):
                add_annotation = True
            else:
                add_annotation = False
            if add_annotation:
                if use_status:
                    if pathologies is not None:
                        if annotation.status + '_' + pathologies[0] in category_id_dict.keys():
                            category_id = category_id_dict[annotation.status + '_' + pathologies[0]]
                        else:
                            break
                    else:
                        if annotation.status + '_pathology' in category_id_dict.keys():
                            category_id = category_id_dict[annotation.status + '_pathology']
                        else:
                            break
                else:
                    if 'mass' in annotation.pathologies and 'mass' in category_id_dict.keys():
                        pathology_selected = 'mass'
                    elif ('calcifications' in annotation.pathologies or 'suspicious_calcifications' in annotation.pathologies) and 'calcification' in category_id_dict.keys():
                        pathology_selected = 'calcification'
                    elif 'architectural_distortion' in annotation.pathologies and 'distortion' in category_id_dict.keys():
                        pathology_selected = 'distortion'
                    else:
                        continue
                    category_id = category_id_dict[pathology_selected]
                bbox_anno = annotation.get_bbox()
                xmin, xmax, ymin, ymax = bbox_anno.xmin, bbox_anno.xmax, bbox_anno.ymin, bbox_anno.ymax
                #poly = [xmin, ymin],[xmax, ymin],[xmax, ymax],[xmin, ymax]
                annot_elem = dict(
                    image_id=image_id,
                    id=obj_count,
                    bbox=[xmin, ymin, xmax - xmin, ymax - ymin],
                    area=(xmax - xmin)*(ymax - ymin),
                    segmentation=[[xmin, ymin, int((xmax-xmin)/2) + xmin, ymin, \
                       xmax, ymin, xmax, int((ymax-ymin)/2) + ymin, \
                       xmax, ymax, int((xmax-xmin)/2) + xmin, ymax, \
                       xmin, ymax, xmin, int((ymax-ymin)/2) + ymin]],
                    category_id=category_id,
                    iscrowd=0
                    )
                annotations_elem.append(annot_elem)
                obj_count += 1 # each bbox increases the counter (here only one per image)
                if False:
                    #poly = [xmin, ymin],[xmax, ymin],[xmax, ymax],[xmin, ymax]
                    poly = [xmin, ymin], [int((xmax-xmin)/2) + xmin, ymin], \
                       [xmax, ymin], [xmax, int((ymax-ymin)/2) + ymin], \
                       [xmax, ymax], [int((xmax-xmin)/2) + xmin, ymax], \
                       [xmin, ymax], [xmin, int((ymax-ymin)/2) + ymin]
                    img = cv2.cvtColor(np.array(Image.open(self.path).convert('RGB')), cv2.COLOR_BGR2RGB)
                    cv2.polylines(img, [np.array(poly)], True, (0, 255, 0), 2)
                    plt.figure()
                    plt.imshow(img)
                    #plt.imsave('./test.png', img)
                    plt.show()
                    # img_pil = self.open_pil(print_annotations=True)
                    # plot_image_opencv_fit_window(img_pil, title='BCDR Scan', 
                    #                          screen_resolution=(1920, 1080), wait_key=True)
        return img_elem, annotations_elem, obj_count

class BCDRDataset(DatasetMMG):
    def __init__(self, info_csv:str, dataset_path:str, cropped_to_breast=False, 
                client_ids=None, image_ids=None, load_max=-1, detection=False):
        if isinstance(info_csv, str):
            info_csv = [info_csv]
        if isinstance(dataset_path, str):
            dataset_path = [dataset_path]
        
        super().__init__(info_csv, dataset_path, cropped_to_breast, 
                         client_ids, image_ids, load_max)

        self.clients_dict = {}

        def get_client(client_id, site, breast_density):
            if client_id in self.clients_dict.keys():
                index = self.clients_dict[client_id]
                return self.clients[index]
            new_client = ClientMMG(client_id)
            new_client.site = site
            new_client.breast_density = breast_density
            return new_client

        def update_client(client:ClientMMG):
            if len(self.clients):
                if client.id in self.clients_dict.keys():
                    index = self.clients_dict[client.id]
                    self.clients[index] = client
                    return
            self.clients.append(client)
            self.clients_dict[client.id] = len(self.clients) - 1

        for dataset_path_i, info_csv_i in zip(dataset_path, info_csv):
            dataset_name = os.path.basename(dataset_path_i)
            info = pd.read_csv(info_csv_i)
            info = info.astype(object).replace(np.nan, '')
            if detection:
                df_bbox = info.loc[info['lesion_x2'] != '']
                unique_patient_id_df = df_bbox.groupby(['patient_id'], as_index=False)
            else:
                unique_patient_id_df = info.groupby(['patient_id'], as_index=False)
            for patient_id, patient_group in unique_patient_id_df:
                # if str(patient_id) == '143':
                #     stop = True
                if client_ids:
                    if patient_group.patient_id.values[0] not in client_ids:
                        continue
                unique_study_id_df = patient_group.groupby(['study_id'], as_index=False)
                # if len(unique_study_id_df) > 1:
                #     stop = True
                for study_id, patient_study_group in unique_study_id_df:
                    unique_series_df = patient_study_group.groupby(['series'], as_index=False)
                    for series_id, series_group in unique_series_df:
                        unique_image_id_df = series_group.groupby(['scan_path'], as_index=False)
                        client_name = dataset_name + '_' + str(patient_id)
                        row_client = get_client(client_name, SITES_BCDR[0], series_group.density.values[0])
                        bool_update_client = False
                        for image_name, image_group in unique_image_id_df:
                            if image_ids:
                                image_name = '/'.join([dataset_path_i.split('/')[-1], image_name])
                                if image_name not in image_ids:
                                    continue
                            # Create new image
                            scan_png_path = image_group.scan_path.values[0]
                            scan_path = os.path.join(dataset_path_i, scan_png_path)
                            # Create new BCDRImage
                            new_image = BCDRImage(scan_path)
                            view = image_group.view.values[0]
                            valid_view =  True
                            if view in CC_VIEWS_BCDR:
                                new_image.view = 'CC'
                            elif view in MLO_VIEWS_BCDR:
                                new_image.view = 'MLO'
                            else:
                                #print(f'Error: view {view} not found in list -> Discard image')
                                valid_view =  False
                            if not valid_view:
                                continue
                            init_image = True
                            for idx_mark, image in enumerate(image_group.itertuples()):
                                if detection and image.lesion_x2 == '': # Check if Annotation is available
                                    break
                                if image.classification == 'Malign':
                                    status = 'Malignant'
                                else:
                                    status = image.classification
                                if init_image:
                                    new_image.id = '/'.join([dataset_path_i.split('/')[-1], image.scan_path])
                                    if image.laterality in ['RIGHT', 'R']:
                                        new_image.laterality = 'R'
                                    else:
                                        new_image.laterality = 'L'
                                    new_image.status = status
                                    new_image.site = SITES_BCDR[0]
                                    new_image.manufacturer = MANUFACTURERS_BCDR[0]
                                    new_image.pixel_spacing = PIXEL_SIZE_BCDR[0]
                                    if client_name in ['BCDR-D01_dataset_511', 'BCDR-D01_dataset_129']:
                                        new_image.implant = 'YES'
                                    else:
                                        new_image.implant = 'NO'
                                    new_image.breast_bbox = BBox(image.breast_x1, image.breast_x2, image.breast_y1, image.breast_y2)
                                    new_image.cropped_to_breast = cropped_to_breast
                                    new_image.width = image.scan_width
                                    new_image.height = image.scan_height
                                    new_image.breast_width = image.breast_width
                                    new_image.breast_height = image.breast_height
                                    new_image.breast_density = image.density
                                    new_image.age = image.age
                                    init_image = False
                                if image.lesion_x2 != '':
                                    new_annotation = BCDRAnnotation(image.lesion_id, image.segmentation_id)
                                    new_annotation.status = status
                                    pathologies = image.lesion_pathologies.strip('][').replace("'", '').split(', ')
                                    new_annotation.pathologies = ['mass' if (x == 'nodule') else x for x in pathologies]
                                    new_annotation.bbox = BBox(image.lesion_x1, image.lesion_x2,
                                                                        image.lesion_y1, image.lesion_y2)
                                    new_annotation.breast_bbox = BBox(image.breast_x1, image.breast_x2, image.breast_y1, image.breast_y2)
                                    new_image.add_annotation(new_annotation)
                                    self.annotation_ctr += 1
  
                            if not init_image:
                                study_name = str(study_id) + '_' + str(series_id)
                                # Update study
                                row_study = row_client.get_study(study_name)
                                if row_study is None:
                                    row_study = StudyMMG(study_name)
                                row_study.add_image(new_image)
                                # Update client
                                row_client.update_study(row_study)
                                self.images_ctr += 1
                                bool_update_client = True
                                if self.images_ctr == self.load_max:
                                    break
                        if bool_update_client:
                            update_client(row_client)
                            if self.images_ctr == self.load_max:
                                break