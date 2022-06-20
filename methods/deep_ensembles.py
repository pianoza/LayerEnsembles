from lib2to3.pytree import Base
import scipy
import warnings 
from scipy.ndimage.morphology import binary_erosion
from torchio.transforms.preprocessing.label.one_hot import OneHot
import pickle
import cv2
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from sklearn.model_selection import train_test_split
import torchio as tio
from data.bcdr_loader import BCDRLoader, get_bcdr_subjects
from data.inbreast_loader import INBreastDataset, get_inbreast_subjects
from data.optimam_loader import OPTIMAMDataset, get_optimam_subjects
# import segmentation_models_pytorch as smp
from segmentation_models_pytorch.unet.model import Unet
from segmentation_models_pytorch.losses.dice import DiceLoss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import ResampleToMask, normalise, make_folders, EarlyStopping, \
                  show_cam_on_image
from metrics import get_evaluations
from methods.base_method import BaseMethod

class DeepEnsembles(BaseMethod):
    def __init__(self, configs, num_ensembles):
        super(DeepEnsembles, self).__init__(configs)
        self.num_ensembles = num_ensembles

    def folders(self):
        models_path, figures_path, seg_out_path = make_folders(self.results_path, self.experiment_name)
        best_model_path = [models_path / f'DE_{i+1}_best_model.pt' for i in range(self.num_ensembles)]
        final_model_path = [models_path / f'DE_{i+1}_final_model.pt' for i in range(self.num_ensembles)]
        results_csv_file = f'results_{self.experiment_name}.csv'
        return best_model_path, final_model_path, figures_path, seg_out_path, results_csv_file
    
    def prepare_model(self, model_path=None):
        models_list = []
        for i in range(self.num_ensembles):
            model = Unet(
                encoder_name="resnet18",
                encoder_weights=None,
                decoder_channels=(1024, 512, 256, 128, 64),
                decoder_attention_type='scse',
                in_channels=1,
                classes=2,
            )
            if model_path is not None:
                model.load_state_dict(torch.load(model_path[i]))
            model.to(self.configs.DEVICE)
            models_list.append(model)
        return models_list

    def run_routine(self, run_test=True):
        # Make experiment folders
        best_model_path, final_model_path, figures_path, seg_out_path, results_csv_file = self.folders()
        # Load data
        train_transforms, test_transforms = self.prepare_transforms()
        self_training_transforms, _ = self.prepare_transforms(is_patches=True) if self.configs.SELF_TRAINING else (None, None)
        train_loader, val_loader, test_loader = self.loaders(self.dataset, train_transforms, test_transforms, self_training=self.configs.SELF_TRAINING,
                                                             self_training_transforms=self_training_transforms)
        # Check final model paths list if all of them exist
        if all([path.exists() for path in final_model_path]):
            print(f'EXPERIMENT {self.experiment_name} EXISTS!')
            print(f'TESTING ONLY')
            # Load pretrained model if already exists and not overwriting
            self.model = self.prepare_model(model_path=best_model_path)
        else:
            # If not exists or overwriting, create new model and start training
            # If tensorboard is enabled, create a new writer
            if self.configs.TENSORBOARD:
                self.tensorboard()
            # Initialise model, loss, optimiser, scheduler, and early_stopping
            self.model = self.prepare_model()
            self.criterion = self.prepare_criterion()
            self.optimizer = self.prepare_optimizer(self.model)
            self.scheduler = self.prepare_scheduler(self.optimizer)
            self.early_stopping = self.prepare_early_stopping(best_model_path)
            # Run training
            self.train(train_loader, val_loader, final_model_path)

        if run_test:
            # Run testing
            self.test(test_loader, seg_out_path, results_csv_file, T=self.num_ensembles)
            # self_training_df = pd.read_csv(self.configs.OPTIMAM_INFO_FILE)
            # self.test(test_loader, seg_out_path, results_csv_file, T=self.num_ensembles, self_training_mode=True, self_training_df=self_training_df, self_training_info_save_path=self.configs.OPTIMAM_INFO_FILE)

        # Close tensorboard writer
        if self.tb_writer is not None:
            self.tb_writer.close()
    
    def prepare_optimizer(self, model):
        # return [torch.optim.Adam(model[i].parameters(), lr=self.configs.LR) for i in range(self.num_ensembles)]
        return [torch.optim.AdamW(model[i].parameters(), lr=self.configs.LR) for i in range(self.num_ensembles)]

    def prepare_scheduler(self, optimizer):
        return [torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer[i], factor=self.configs.LR_DECAY_FACTOR, patience=self.configs.SCHEDULER_PATIENCE,
                                                          min_lr=self.configs.LR_MIN, verbose=True) for i in range(self.num_ensembles)]
    
    def run_test_passes(self, model, loader, iters, T, active_learning=False):
        for i in range(T):
            model[i].eval()
        images = []
        labels = []
        statuses = []
        outnames = []
        pathologies = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(loader)):
                img = batch['scan'][tio.DATA]
                lbl = batch['mask'][tio.DATA].argmax(axis=1)
                status = batch['status'][0]
                pid = batch['patient_id'][0] if isinstance(batch['patient_id'][0], str) else batch['patient_id'][0].item()
                sid = batch['study_id'][0] if isinstance(batch['study_id'][0], str) else batch['study_id'][0].item()
                lid = batch['lesion_id'][0] if isinstance(batch['lesion_id'][0], str) else batch['lesion_id'][0].item()

                outname = f"case_{i}-pid_{pid}-{batch['laterality'][0]}_{batch['view'][0]}-sid_{sid}-lid_{lid}.png"
                outnames.append(outname)
                images.append(np.squeeze(img.numpy()))
                labels.append(np.squeeze(lbl.numpy()))
                statuses.append('Benign' if status==0 else 'Malignant')
                pathologies.append(batch['pathology'])
                inputs, _ = self.prepare_batch(batch, self.device)
                for t in range(T):
                    output = self.forward(model[t], inputs)
                    iters[i, t] = output.detach().cpu().numpy()
        return iters, images, labels, statuses, pathologies, outnames
    
    def train(self, train_loader, val_loader, final_model_path, transforms=None):
        for alpha, epoch in zip(np.linspace(0.01, 0.99, num=self.num_epochs), range(self.num_epochs)):
            val_losses = []
            val_dscs = []
            train_losses = []
            for ensemble_idx in range(self.num_ensembles):
                # Train and validate
                train_loss = self.train_epoch(self.model[ensemble_idx], train_loader, self.criterion, self.optimizer[ensemble_idx], alpha)
                val_loss, val_dice = self.validate_epoch(self.model[ensemble_idx], val_loader, self.criterion, alpha)
                # Update scheduler
                self.scheduler[ensemble_idx].step(val_loss)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                val_dscs.append(val_dice)
            # Combine ensemble losses
            val_loss = sum(val_losses) / self.num_ensembles
            val_dice = sum(val_dscs) / self.num_ensembles
            train_loss = sum(train_losses) / self.num_ensembles
            # Early stopping also accepts list of models and save them in the same order in case of overall val loss improvement
            self.early_stopping(val_loss, self.model)
            # Push to tensorboard if enabled
            if self.tb_writer is not None:
                log_losses = {'train': train_loss, 'val': val_loss, 'val_dsc': val_dice}
                self.tb_writer.add_scalars("Losses", log_losses, epoch)
                # plot validation
                if epoch == 1 or epoch % self.plot_validation_frequency == 0:
                    self.plot_validation(epoch, val_loader)
                self.tb_writer.flush()

            if self.early_stopping.early_stop:
                print(f'EARLY STOPPING at EPOCH {epoch+1}')
                break
            
        for ensemble_idx in range(self.num_ensembles):
            torch.save(self.model[ensemble_idx].state_dict(), final_model_path[ensemble_idx])

    def plot_validation(self, epoch, val_loader):
        one_batch = next(iter(val_loader))
        batch_img, batch_label = self.prepare_batch(one_batch, self.device)
        with torch.set_grad_enabled(False):
            # Get predictions for each model 
            ensemble_logits = torch.stack([self.forward(self.model[i], batch_img) for i in range(self.num_ensembles)], dim=0)
            # Average the ensemble predictions
            logits = ensemble_logits.mean(axis=0)
            batch_seg = logits.argmax(axis=1).unsqueeze(1)
            batch_label = batch_label.argmax(axis=1).unsqueeze(1)
            slices = torch.cat((batch_img[:8], batch_seg[:8], batch_label[:8]))
            batch_plot = make_grid(slices, nrow=8, normalize=True, scale_each=True)
            self.tb_writer.add_image(f'ImgSegLbl/val_epoch_{epoch+1}', batch_plot, global_step=epoch)
