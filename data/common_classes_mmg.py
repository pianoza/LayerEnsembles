
from abc import ABC, abstractmethod
from PIL import Image
import cv2
import numpy as np
# from data.plot_image import plot_image_opencv_fit_window

class BBox():
    def __init__(self, xmin:int, xmax:int, ymin:int, ymax:int):
        self.xmin = int(xmin)
        self.xmax = int(xmax)
        self.ymin = int(ymin)
        self.ymax = int(ymax)
        self.top = int(xmin)
        self.left = int(ymin)
        self.bottom = int(xmax)
        self.right = int(ymax)

class AnnotationMMG(ABC):
    def __init__(self):
        self.breast_bbox = None

    @property
    def pathologies(self):
        return self._pathologies
    
    @pathologies.setter
    def pathologies(self, pathologies:list):
        if not isinstance(pathologies, list):
            self._pathologies = [pathologies]
        else:
            self._pathologies = pathologies

    @property
    def status(self):
        return self._status
    
    @status.setter
    def status(self, status:str):
        self._status = status

    @property
    def breast_bbox(self):
        return self._breast_bbox
    
    @breast_bbox.setter
    def breast_bbox(self, bbox:BBox):
        self._breast_bbox = bbox

    @property
    def bbox(self):
        return self._bbox
    
    @bbox.setter
    def bbox(self, bbox:BBox):
        self._bbox = bbox
    
    def get_bbox(self, fit_to_breast:bool=False):
        if fit_to_breast and self.breast_bbox is not None:
            return BBox(max(0, self.bbox.xmin - self.breast_bbox.xmin), min(self.bbox.xmax - self.breast_bbox.xmin, \
                        self.breast_bbox.xmax), max(0, self.bbox.ymin - self.breast_bbox.ymin), min(self.bbox.ymax - self.breast_bbox.ymin, self.breast_bbox.ymax))
        else:
            return self.bbox

class ImageMMG(ABC):
    def __init__(self, scan_path):
        self.path = scan_path
        self.id = None
        self.site = None
        self.manufacturer = None
        self.laterality = None
        self.width = None
        self.height = None
        self.cropped_to_breast = False
        self.breast_bbox = None
        self.age = None
        self.breast_density = None
        self.annotations = []
        self.status = None
        self.implant = False
        self.birad = None

    # @abstractmethod
    # def generate_COCO_dict(self, *args, **kwargs):
    #     pass
    
    @property
    def implant(self):
        return self._implant
    
    @implant.setter
    def implant(self, implant):
        if isinstance(implant, str):
            if implant.lower() in ['yes', 'y']:
                self._implant = True
            else:
                self._implant = False
        elif isinstance(implant, bool):
            self._implant = implant
        else:
            raise ValueError('Implant value should be a string or a bool value')

    @property
    def pixel_spacing(self):
        return self._pixel_spacing
    
    @pixel_spacing.setter
    def pixel_spacing(self, pixel_spacing):
        self._pixel_spacing = float(pixel_spacing)

    def add_annotation(self, annotation:AnnotationMMG):
        self.annotations.append(annotation)
    
    def total_annotations(self, pathologies=None):
        if pathologies:
            counter = 0
            for annotation in self.annotations:
                if any(item in annotation.pathologies for item in pathologies):
                    counter += 1
            return counter
        else:
            return len(self.annotations)
    
    def get_pathologies(self):
        pathologies = []
        for annotation in self.annotations:
            for pathology in annotation.pathologies:
                pathologies.append(pathology)
        return list(set(pathologies))
    
    def fit_to_breast(self, channels=1):
        low_int_threshold = 0.05
        max_value = 255
        img_pil = Image.open(self.path)
        img_pil = np.array(img_pil)
        height, width = img_pil.shape[0], img_pil.shape[1]
        img_8u = (img_pil.astype('float32')/img_pil.max()*max_value).astype('uint8')
        
        low_th = int(max_value*low_int_threshold)
        _, img_bin = cv2.threshold(img_8u, low_th, maxval=max_value, type=cv2.THRESH_BINARY)
        contours,_ = cv2.findContours(img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cont_areas = [ cv2.contourArea(cont) for cont in contours ]
        idx = np.argmax(cont_areas)
        # # fill the contour.
        # breast_mask = cv2.drawContours(np.zeros_like(img_bin), contours, idx, 255, -1) 
        # # segment the breast.
        # img_breast_only = cv2.bitwise_and(img_pil, img_pil, mask=breast_mask)
        x,y,w,h = cv2.boundingRect(contours[idx])
        if channels == 3:
            img_pil = Image.open(self.path).convert('RGB')
            img_pil = np.array(img_pil)
        img_breast = img_pil[y:y+h, x:x+w]
        (xmin, xmax, ymin, ymax) = (x, x + w, y, y + h)
        self.breast_bbox = BBox(xmin, xmin, ymin, ymax)
        for annotation in self.annotations:
            annotation.breast_bbox = BBox(xmin, xmin, ymin, ymax)
        return img_breast, self.breast_bbox

    def open_pil(self, print_annotations=False, color_bbox=(0, 255, 0),
                bbox_thickness=4, fit_to_breast=False):
        img_pil = Image.open(self.path).convert('RGB')
        img_pil = np.array(img_pil)
        if self.height is None and self.width is None:
            self.height, self.width = img_pil.shape[0], img_pil.shape[1]
        if fit_to_breast and not self.cropped_to_breast:
            img_pil, _ = self.fit_to_breast(channels=3)
        if print_annotations and self.annotations: 
            for annotation in self.annotations:
                bbox = annotation.get_bbox(fit_to_breast=fit_to_breast)
                cv2.rectangle(img_pil, (bbox.top, bbox.left), (bbox.bottom, bbox.right), color_bbox, bbox_thickness)
        return img_pil
    
class StudyMMG(ABC):
    def __init__(self, study_id):
        self.id = study_id
        self.images = []
    
    def add_image(self, image:ImageMMG):
        self.images.append(image)

    def total_images(self, pathologies=None, status=None):
        if pathologies:
            counter = 0
            if status:
                if not isinstance(status, list):
                    status = [status]
            for image in self.images:
                image_pathologies = image.get_pathologies()
                if any(item in image_pathologies for item in pathologies):
                    if status:
                        if image.status in status:
                            counter += 1
                    else:
                        counter += 1
            return counter
        else:
            if status:
                if not isinstance(status, list):
                    status = [status]
                counter = 0
                for image in self.images:
                    if image.status in status:
                        counter += 1
                return counter
            else:
                return len(self.images)
            
    def total_annotations(self, pathologies=None):
        counter = 0
        for image in self.images:
            counter += image.total_annotations(pathologies)
        return counter

    def get_image(self, image_id):
        for image in self.images:
            if image.id == image_id:
                return image
        return None
    
    def get_images_by_pathology(self, pathologies):
        images = []
        if pathologies:
            for image in self.images:
                image_pathologies = image.get_pathologies()
                if any(item in image_pathologies for item in pathologies):
                    images.append(image)
            return images
        else:
            return self.images
    
    def get_images_by_status(self, status):
        images = []
        for image in self.images:
            if image.status in list(status):
                images.append(image)
        return images
    
    def get_images_by_site(self, site):
        images = []
        if not isinstance(site,list):
            site = [site]
        for image in self.images:
            if image.site in list(site):
                images.append(image)
        return images
    
    def plot_study(self, print_annotations=True, fit_to_breast=False):
        title = ''
        multi_view = []
        img_height, img_width = None, None
        
        for view, laterality in zip(['CC', 'CC', 'MLO', 'MLO'], ['L', 'R', 'L', 'R']):
            view_img = self.get_images_by_view_laterality(view, laterality)
            if len(view_img):
                img_pil = view_img[0].open_pil(print_annotations=print_annotations,
                                fit_to_breast=fit_to_breast)
                if not img_width:
                    img_height, img_width = img_pil.shape[0], img_pil.shape[1]
                else:
                    img_pil = cv2.resize(img_pil, dsize=(img_width, img_height), interpolation=cv2.INTER_CUBIC)
                multi_view.append(img_pil)
                title += f' {laterality}{view} |'

        if len(multi_view) == 4:
            img_pil = np.concatenate((multi_view[0], multi_view[2], multi_view[1], multi_view[3]), axis=1)
        else:
            for i, image in enumerate(multi_view):
                if i+1 == len(multi_view): break
                img_pil = np.concatenate((multi_view[i], multi_view[i+1]), axis=1)
        print(title)
            
        if img_pil is not None:
            plot_image_opencv_fit_window(img_pil, title='Studies', 
                                        screen_resolution=(1920, 1080), wait_key=True)

    def get_images_by_view_laterality(self, view, laterality):
        images = []
        if not isinstance(view, list):
            view = [view]
        if not isinstance(laterality, list):
            laterality = [laterality]
        for image in self.images:
            if image.laterality in laterality:
                if image.view in view:
                    images.append(image)
        return images

    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        return self.images[idx]
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

class ClientMMG(ABC):
    def __init__(self, client_id):
        self.id=client_id
        self.breast_density = None
        self.site = None
        # self.status = None
        self.studies = []

    def update_study(self, study:StudyMMG):
        for idx, s in enumerate(self.studies):
            if s.id == study.id:
                self.studies[idx] = study
                return
        self.studies.append(study)

    def total_studies(self, pathologies=None):
        if not pathologies:
            return len(self.studies)
        else:
            counter = 0
            for study in self.studies:
                image_counter = study.total_images(pathologies)
                if image_counter:
                    counter += 1
            return counter
            
    def total_images(self, pathologies=None, status=None):
        counter = 0
        for study in self.studies:
            counter += study.total_images(pathologies=pathologies, status=status)
        return counter
    
    def total_annotations(self, pathologies=None):
        counter = 0
        for study in self.studies:
            counter += study.total_annotations(pathologies)
        return counter

    def get_study(self, study_id):
        for study in self.studies:
            if study.id == study_id:
                return study
        return None

    def get_images_by_pathology(self, pathologies):
        client_images = []
        for study in self.studies:
            images = study.get_images_by_pathology(pathologies)
            if len(images):
                for image in images:
                    client_images.append(image)
        return client_images
    
    def get_images_by_site(self, site):
        client_images = []
        for study in self.studies:
            images = study.get_images_by_site(site)
            if len(images):
                for image in images:
                    client_images.append(image)
        return client_images
    
    def get_images_by_status(self, status):
        client_images = []
        for study in self.studies:
            images = study.get_images_by_status(status)
            if len(images):
                for image in images:
                    client_images.append(image)
        return client_images

    def get_image(self, image_id):
        for study in self.studies:
            image = study.get_image(image_id)
            if image:
                return image
        return image

    def __len__(self):
        return len(self.studies)
    def __getitem__(self, idx):
        return self.studies[idx]
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

class DatasetMMG(ABC):
    def __init__(self, info_csv:str, dataset_path:str, cropped_to_breast=False,
                 client_ids=None, image_ids=None, load_max=-1):   
        self.info_csv = info_csv
        self.dataset_path = dataset_path
        self.load_max = load_max
        self.cropped_to_breast = cropped_to_breast
        self.client_ids = client_ids
        self.clients = []
        self.images_ctr = 0
        self.annotation_ctr = 0

    def total_clients(self, pathologies=None, status=None):
        if not pathologies and not status:
            return len(self.clients)
        else:
            counter = 0
            for idx, client in enumerate(self.clients):
                if client.total_images(pathologies, status) > 0:
                    counter += 1
            return counter

    def total_images(self, pathologies=None, status=None):
        counter = 0
        for client in self.clients:
            counter += client.total_images(pathologies, status)
        return counter

    def total_annotations(self, pathologies=None):
        counter = 0
        for client in self.clients:
            counter += client.total_annotations(pathologies)
        return counter
    
    def get_client(self, client_id):
        for client in self.clients:
            if client.id == client_id:
                return client
        return None

    def get_clients_by_pathology_and_status(self, pathologies, status=None):
        clients = []
        for client in self.clients:
            images = client.total_images(pathologies, status)
            if images > 0:
                clients.append(client)
        return clients
    
    def get_clients_by_status(self, status):
        clients = []
        for client in self.clients:
            images = client.total_images(status=status)
            if images > 0:
                clients.append(client)
        return clients

    def get_images_by_site(self, site):
        clients = []
        for client in self.clients:
            images = client.get_images_by_site(site)
            if len(images) > 0:
                clients.append(client)
        return clients

    def get_image(self, image_id):
        for client in self.clients:
            image = client.get_image(image_id)
            if image:
                return image
        return image

    def __len__(self):
        return len(self.clients)
    def __getitem__(self, idx):
        return self.clients[idx]
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
    
    def plot_dataset(self, print_annotations=True, fit_to_breast=False, max=-1):
        image_ctr = 0
        for client in self.clients:
            for study in client:
                for image in study:
                    img_pil = image.open_pil(print_annotations=print_annotations,
                                fit_to_breast=fit_to_breast)
                    plot_image_opencv_fit_window(img_pil, title='MMG Scan', 
                                             screen_resolution=(1920, 1080), wait_key=True)
                    image_ctr += 1
                    if image_ctr >= max:
                        return
