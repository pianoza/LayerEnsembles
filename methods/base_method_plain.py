from tkinter import W
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
import torch.nn.functional as F
from torchvision.utils import make_grid
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import torchio as tio
from data.bcdr_loader import BCDRLoader, get_bcdr_subjects
from data.inbreast_loader import INBreastDataset, get_inbreast_subjects
from data.optimam_loader import OPTIMAMDatasetLoader, get_optimam_subjects
from data.mnm_loader import get_mnm_subjects, MnMDataLoader
# import segmentation_models_pytorch as smp
from segmentation_models_pytorch.unet.model import Unet
from segmentation_models_pytorch.losses.dice import DiceLoss
# from segmentation_models_pytorch.losses.boundary_loss import BoundaryLoss, HausdorffLoss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import ResampleToMask, normalise, make_folders, EarlyStopping, \
                  show_cam_on_image
from metrics import get_evaluations
from methods.randconv.randconv_transform import RandConvTransform

from pathlib import Path
from PIL import Image
import SimpleITK as sitk 

# torch.autograd.set_detect_anomaly(True)

class BaseMethodPlain:
    def __init__(self, configs):
        self.configs = configs
        self.results_path = configs.RESULTS_PATH
        self.experiment_name = configs.EXPERIMENT_NAME
        self.dataset = configs.DATASET
        self.tb_writer = None
        self.tboard_exp = configs.TENSORBOARD_ROOT / configs.EXPERIMENT_NAME
        self.model = None
        self.overwrite = configs.OVERWRITE
        self.device = configs.DEVICE
        self.num_epochs = configs.NUM_EPOCHS
        self.plot_validation_frequency = configs.PLOT_VALIDATION_FREQUENCY

    def run_routine(self, run_test=True):
        # Make experiment folders
        best_model_path, final_model_path, figures_path, seg_out_path, results_csv_file = self.folders()
        # Load data
        train_transforms, test_transforms = self.prepare_transforms()
        train_loader, val_loader, test_loader = self.loaders(self.dataset, train_transforms, test_transforms)
        # Load pretrained model if already exists and not overwriting
        if final_model_path.exists() and not self.overwrite:
            print(f'EXPERIMENT {self.experiment_name} EXISTS!')
            print(f'TESTING ONLY')
            self.model = self.prepare_model(best_model_path)
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
            self.test(test_loader, seg_out_path, results_csv_file, T=1)

        # Close tensorboard writer
        if self.tb_writer is not None:
            self.tb_writer.close()
    
    def folders(self):
        models_path, figures_path, seg_out_path = make_folders(self.results_path, self.experiment_name)
        best_model_path = models_path / 'best_model.pt'
        final_model_path = models_path / 'final_model.pt'
        results_csv_file = f'results_{self.experiment_name}.csv'
        return best_model_path, final_model_path, figures_path, seg_out_path, results_csv_file
    
    def tensorboard(self):
        if not self.tboard_exp.is_dir():
            self.tboard_exp.mkdir(parents=True, exist_ok=True)
        dirs = [d for d in self.tboard_exp.iterdir()]
        next_run = len(dirs)+1
        self.tb_writer = SummaryWriter(log_dir=self.tboard_exp / str(next_run))
    
    def prepare_model(self, model_path=None):
        model = Unet(
            encoder_name="resnet18",
            encoder_weights=None,
            decoder_channels=(1024, 512, 256, 128, 64),
            decoder_attention_type='scse',
            in_channels=1,
            classes=2,
            layer_ensembles=False,
        )
        if model_path is not None:
            model.load_state_dict(torch.load(model_path))
        model.to(self.configs.DEVICE)
        return model
    
    def prepare_criterion(self):
        return DiceLoss(
            mode='multilabel',
            classes=[1,],
            log_loss=False,
            from_logits=True,
            smooth=0.0000001,
            ignore_index=None,
        )
    
    def prepare_optimizer(self, model):
        return torch.optim.Adam(model.parameters(), lr=self.configs.LR)
    
    def prepare_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.configs.LR_DECAY_FACTOR, patience=self.configs.SCHEDULER_PATIENCE,
                                                          min_lr=self.configs.LR_MIN, verbose=True)

    def prepare_early_stopping(self, best_model_path):
        return EarlyStopping(patience=self.configs.EARLY_STOPPING_PATIENCE, verbose=True, path=best_model_path)
    
    def prepare_transforms(self, target_image_size=256,
                           min_intensity=0, max_intensity=1,
                           min_percentile=0, max_percentile=100,
                           masking_method=None, is_patches=False):
        train_transforms = tio.Compose([
            ResampleToMask(im_size=target_image_size),
            tio.CropOrPad((target_image_size, target_image_size, 1), mask_name='mask'), 
            tio.RandomFlip(axes=(0, 1), p=0.2),
            RandConvTransform(kernel_size=(1, 3, 5, 7), mixing=True, identity_prob=0.8),
            tio.RandomSwap((10, 10, 1), 10, p=0.2),
            tio.ZNormalization(),
            tio.OneHot(num_classes=2),
        ])
        test_transforms = tio.Compose([
            ResampleToMask(im_size=target_image_size),
            tio.CropOrPad((target_image_size, target_image_size, 1), mask_name='mask'), 
            tio.ZNormalization(),
            tio.OneHot(num_classes=2),
            # RandConvTransform(kernel_size=(37, 37), mixing=False, identity_prob=0.0),
            # tio.RescaleIntensity((0, 1), (1, 90)),
            # tio.RandomNoise((140, 145), (100.0, 100.5), p=1.0),
            # tio.RandomBlur((5, 5, 5, 5, 0, 0), p=2.0),
            # tio.RandomSwap((20, 20, 1), 10, p=1.0),
        ])
        return train_transforms, test_transforms
    
    def loaders(self, dataset, train_transforms, test_transforms, train_val_test_split=(0.60, 0.20, 0.20), batch_size=None, sampler=None):
        if dataset == 'bcdr':
            df = get_bcdr_subjects(self.configs.BCDR_PATH, self.configs.BCDR_INFO_CSV, self.configs.BCDR_OUTLINES_CSV)
            print(f'BCDR ---- TOTAL NUMBER OF SAMPLES: {df.shape[0]}')
            df_train, df_val, df_test = self.unique_group_train_val_test_split(df, train_val_test_split)
            print(f'SPLITS ---- {df_train.shape[0]} TRAINING, {df_val.shape[0]} VALIDATION, {df_test.shape[0]} TESTING')
            train_dataset = BCDRLoader(df_train, transform=train_transforms)
            val_dataset = BCDRLoader(df_val, transform=test_transforms)
            test_dataset = BCDRLoader(df_test, transform=test_transforms)
        elif dataset == 'inbreast':
            df = get_inbreast_subjects(self.configs.INBREAST_INFO_FILE)
            print(f'INBREAST ---- TOTAL NUMBER OF SAMPLES: {df.shape[0]}')
            if train_val_test_split[0] == 0. and train_val_test_split[1] == 0:
                print('ALL SAMPLES FOR TESTING!')
                test_dataset = INBreastDataset(df, self.configs.INBREAST_IMAGES_DIR, transform=test_transforms)
                test_loader = DataLoader(test_dataset, batch_size=self.configs.TEST_BATCH_SIZE, shuffle=False, num_workers=self.configs.NUM_WORKERS)
                return None, None, test_loader
            df_train, df_val, df_test = self.unique_group_train_val_test_split(df, train_val_test_split)
            print(f'SPLITS ---- {df_train.shape[0]} TRAINING, {df_val.shape[0]} VALIDATION, {df_test.shape[0]} TESTING')
            train_dataset = INBreastDataset(df_train, self.configs.INBREAST_IMAGES_DIR, transform=train_transforms)
            val_dataset = INBreastDataset(df_val, self.configs.INBREAST_IMAGES_DIR, transform=test_transforms)
            test_dataset = INBreastDataset(df_test, self.configs.INBREAST_IMAGES_DIR, transform=test_transforms)
        elif dataset == 'optimam':
            df = get_optimam_subjects(self.configs.OPTIMAM_INFO_FILE)
            print(f'OPTIMAM ---- TOTAL NUMBER OF SAMPLES: {df.shape[0]}')
            test_dataset = OPTIMAMDatasetLoader(df, transform=test_transforms)
            test_loader = DataLoader(test_dataset, batch_size=self.configs.TEST_BATCH_SIZE, shuffle=False, num_workers=self.configs.NUM_WORKERS)
            return None, None, test_loader
        elif dataset == 'mnm':
            train_subjects = get_mnm_subjects(self.configs, self.configs.MnM_TRAIN_FOLDER, self.configs.MnM_INFO_FILE)
            val_subjects = get_mnm_subjects(self.configs, self.configs.MnM_VALIDATION_FOLDER, self.configs.MnM_INFO_FILE)
            # train_loader, val_loader = None, None
            test_subjects = get_mnm_subjects(self.configs, self.configs.MnM_TEST_FOLDER, self.configs.MnM_INFO_FILE)
            print(f'MnM ---- TOTAL NUMBER OF SAMPLES: {len(train_subjects) + len(val_subjects) + len(test_subjects)}')
            print(f'SPLITS ---- {len(train_subjects)} TRAINING, {len(val_subjects)} VALIDATION, {len(test_subjects)} TESTING')
            train_dataset = tio.SubjectsDataset(train_subjects, transform=train_transforms)
            val_dataset = tio.SubjectsDataset(val_subjects, transform=test_transforms)
            test_dataset = tio.SubjectsDataset(test_subjects, transform=test_transforms)
            sampler = tio.sampler.UniformSampler((128, 128, 1))
            train_loader = MnMDataLoader(train_dataset, batch_size=self.configs.BATCH_SIZE, max_length=2000, samples_per_volume=5, sampler=sampler, shuffle=True, num_workers=self.configs.NUM_WORKERS)
            val_loader = MnMDataLoader(val_dataset, batch_size=self.configs.VAL_BATCH_SIZE, max_length=2000, samples_per_volume=5, sampler=sampler, shuffle=False, num_workers=self.configs.NUM_WORKERS)
            test_loader = MnMDataLoader(test_dataset, batch_size=self.configs.TEST_BATCH_SIZE, max_length=2000, samples_per_volume=10, sampler=sampler, shuffle=False, num_workers=self.configs.NUM_WORKERS)
            return train_loader, val_loader, test_dataset
        else:
            raise ValueError(f'Unknown dataset: {dataset}')
        train_batch_size = batch_size if batch_size is not None else self.configs.BATCH_SIZE

        if sampler is not None:
            train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False, num_workers=self.configs.NUM_WORKERS, sampler=sampler)
        else: 
            train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=self.configs.NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=self.configs.VAL_BATCH_SIZE, shuffle=False, num_workers=self.configs.NUM_WORKERS)
        test_loader = DataLoader(test_dataset, batch_size=self.configs.TEST_BATCH_SIZE, shuffle=False, num_workers=self.configs.NUM_WORKERS)
        return train_loader, val_loader, test_loader

    def unique_group_train_val_test_split(self, df, train_val_test_split):
        train_inds, test_inds = next(GroupShuffleSplit(test_size=train_val_test_split[2], n_splits=2, random_state=self.configs.RANDOM_SEED).split(df, groups=df['patient_id']))
        df_tmp = df.iloc[train_inds]
        df_test = df.iloc[test_inds]
        train_inds, val_inds = next(GroupShuffleSplit(test_size=train_val_test_split[1], n_splits=2, random_state=self.configs.RANDOM_SEED).split(df_tmp, groups=df_tmp['patient_id']))
        df_train = df_tmp.iloc[train_inds]
        df_val = df_tmp.iloc[val_inds]
        return df_train, df_val, df_test

    # def stratified_train_val_test_split(self, df, train_val_test_split):
    #     df_train, df_test = train_test_split(df, test_size=train_val_test_split[2], random_state=self.configs.RANDOM_SEED, stratify=df['status'])
    #     df_train, df_val = train_test_split(df_train, test_size=train_val_test_split[1], random_state=self.configs.RANDOM_SEED, stratify=df_train['status'])
        # return df_train, df_val, df_test
    
    def train(self, train_loader, val_loader, final_model_path):
        for alpha, epoch in zip(np.linspace(0.01, 0.99, num=self.num_epochs), range(self.num_epochs)):
            # Train and validate
            train_loss, all_head_losses = self.train_epoch(self.model, train_loader, self.criterion, self.optimizer, alpha)
            print('All head losses:', all_head_losses)
            val_loss, val_dice = self.validate_epoch(self.model, val_loader, self.criterion, alpha)
            # Update scheduler and early stopping
            self.scheduler.step(val_loss)
            self.early_stopping(val_loss, self.model)
            # Push to tensorboard if enabled
            if self.tb_writer is not None:
                log_losses = {'train': train_loss, 'val': val_loss, 'val_dice': val_dice}
                self.tb_writer.add_scalars("Losses", log_losses, epoch)
                # plot validation
                if epoch == 1 or epoch % self.plot_validation_frequency == 0:
                    self.plot_validation(epoch, val_loader)
                self.tb_writer.flush()

            if self.early_stopping.early_stop:
                print(f'EARLY STOPPING at EPOCH {epoch+1}')
                break

        torch.save(self.model.state_dict(), final_model_path)
        return train_loss, val_loss
    
    def plot_validation(self, epoch, val_loader):
        one_batch = next(iter(val_loader))
        batch_img, batch_label = self.prepare_batch(one_batch, self.device)
        with torch.set_grad_enabled(False):
            logits = self.forward(self.model, batch_img)
            batch_seg = logits.argmax(axis=1).unsqueeze(1)
            batch_label = batch_label.argmax(axis=1).unsqueeze(1)
            slices = torch.cat((batch_img[:8], batch_seg[:8], batch_label[:8]))
            batch_plot = make_grid(slices, nrow=8, normalize=True, scale_each=True)
            self.tb_writer.add_image(f'ImgSegLbl/val_epoch_{epoch+1}', batch_plot, global_step=epoch)
    
    def forward(self, model, inputs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            logits = model(inputs)
        return logits

    def prepare_batch(self, batch, device):
        inputs = batch['scan'][tio.DATA].squeeze(-1).to(device)
        targets = batch['mask'][tio.DATA].squeeze(-1).to(device)
        return inputs, targets
    
    def train_epoch(self, model, train_loader, criterion, optimizer, alpha):
        model.train()
        train_loss = 0
        all_train_losses = []
        for batch_idx, batch in enumerate(train_loader):
            data, target = self.prepare_batch(batch, self.device)
            optimizer.zero_grad()
            outputs = model(data)
            losses = criterion(outputs, target)  # Dice loss <- accepts logits
            all_train_losses.append(losses.item())
            losses.backward()
            train_loss += losses
            optimizer.step()
        all_train_losses = np.asarray(all_train_losses).mean(axis=0)
        return train_loss / len(train_loader), all_train_losses
    
    def validate_epoch(self, model, val_loader, criterion, alpha):
        model.eval()
        val_loss = 0
        val_dice = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                data, target = self.prepare_batch(batch, self.device)
                outputs = model(data)
                # Get the output from the last segmentation head
                seg = torch.argmax(outputs.squeeze(), dim=1).detach().cpu().numpy()
                lbl = torch.argmax(target.squeeze(), dim=1).detach().cpu().numpy()
                for s, l in zip(seg, lbl):
                    evals = get_evaluations(s, l, spacing=(1, 1))
                    val_dice.append(evals['dsc_seg'])
                loss = criterion(outputs, target)  # Dice loss <- accepts logits
                val_loss += loss.item()
        assert all([dsc <= 1.0001 for dsc in val_dice])
        return val_loss / len(val_loader), sum(val_dice) / len(val_dice)

    def run_test_passes(self, model, loader, iters, T):
        model.eval()
        images = []
        labels = []
        statuses = []
        outnames = []
        pathologies = []
        layer_agreement = []
        all_agreements_for_plot = []
        counter = 0
        with torch.no_grad():
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
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
                inputs, target = self.prepare_batch(batch, self.device)
                outputs = self.forward(model, inputs)[-T:]  # skipping first N - T layers
                # fig, ax = plt.subplots(1, len(outputs)+3, figsize=(len(outputs)*5, 5))  # +3 is for the image, ground truth and prediction depth curve 
                # ax[0].imshow(inputs.squeeze().cpu().numpy(), cmap='gray')
                # ax[0].set_title('Input')
                # ax[-2].imshow(target.squeeze().cpu().numpy().argmax(axis=0), cmap='gray')
                # ax[-2].set_title('Ground Truth')
                agreement = []  # agreement between layer output
                last_layer = outputs[-1].squeeze().cpu().numpy().argmax(axis=0)
                for j in range(len(outputs)):
                    evals = get_evaluations(outputs[j].squeeze().cpu().numpy().argmax(axis=0), last_layer, spacing=(1, 1))
                    agreement.append(evals['dsc_seg'])
                # prev_layer = outputs[0].squeeze().cpu().numpy().argmax(axis=0) 
                # for j in range(1, len(outputs)+1):
                #     cur_layer = outputs[j-1].squeeze().cpu().numpy().argmax(axis=0)
                #     if j > 1:
                #         evals = get_evaluations(prev_layer, cur_layer, spacing=(1, 1))
                #         agreement.append(evals['dsc_seg'])
                #         prev_layer = cur_layer
                ax.plot(range(len(agreement)), agreement, color='green', alpha=0.2)
                all_agreements_for_plot.append(agreement)
                area_under_agreement = np.trapz(agreement, dx=1)  # remove the hard coded [4:]
                layer_agreement.append(-area_under_agreement)
                    # ax[j].imshow(cur_layer, cmap='gray')
                    # ax[j].set_title(f'Layer {j}')
                # ax[-1].plot(np.arange(len(agreement)), agreement)
                # ax[-1].set_ylim(0, 1)
                # ax[-1].set_title('Layer agreement (DSC)')
                # counter += 1
                # plt.savefig('/tmp/nuem/'+str(counter)+'.png')
                # plt.close()
                # early layers learn simple functions and easier samples first, later layers learn more complex functions and memorize details.
                # The agreement between layers is a measure of how well the network is learning the details of the image.
                # It basically shows the evolution of the network as it learns more complex functions.
                # Hence, it can be used to measure the uncertainty of the network (both aleatoric and epistemic)
                # Also, AULA is a good measure of the uncertainty of segmentation, because Var, MI, and Entropy are pixel-wise, wheras AULA gives a global image uncertainty.

                for t in range(len(outputs)-T, len(outputs)):
                    iters[i, t] = outputs[t].detach().cpu().numpy()
            all_agreements_for_plot = np.asarray(all_agreements_for_plot)
            ax.plot(range(all_agreements_for_plot.shape[1]), np.mean(all_agreements_for_plot, axis=0), color='red', linewidth=3.0, label='Mean')
            ax.legend()
            ax.set_title('Layer agreement')
            ax.set_xlabel('Layer')
            ax.set_ylabel('DSC')
            ax.set_xlim(0, T)
            # ax.set_ylim(0, 1)
            plt.savefig('All_layer_agreement.png', dpi=300)
            plt.close()
        # return None
        return iters, images, labels, statuses, pathologies, outnames, layer_agreement

    def test(self, loader, seg_out_path, results_csv_file, T, active_learning_mode=False, w=256, h=256, batch_size=1, self_training_mode=False, self_training_df=None, self_training_info_save_path=None):
        if self_training_mode == True and (self_training_df is None or self_training_info_save_path is None):
            raise(ValueError(f'Self-training DataFrame and New Save Path should be provided when self-training mode is True'))
        # Test images
        seg_masks = []
        len_dataset = len(loader.dataset)

        iters = np.zeros((len_dataset, T, 2)+(w, h))

        iters, images, labels, statuses, pathologies, outnames, area_under_agreement = self.run_test_passes(self.model, loader, iters, T)

        # for each image
        calibration_pairs = []  # tuple (y_true, y_prob) only for the positive class
        # heatmaps
        entropy_maps = []
        variance_maps = []
        mi_maps = []
        # numerical metrics
        avg_entropy = []
        avg_variance = []
        avg_mi = []
        dsc_all = []
        hd_all = []
        mhd_all = []
        nll_all = []

        if self_training_mode:
            image_paths = []
            seg_paths = []

        for idx, (img, lbl, it) in enumerate(zip(images, labels, iters)):
            it = scipy.special.softmax(it, axis=1)
            calibration_pairs.append((lbl, it.mean(axis=0)[1]))
            # TODO - flag for using average or STAPLE
            # Final segmentation
            # 1) Average over all
            # tmp = it.mean(axis=0)
            # seg = tmp.argmax(axis=0)
            # 2) STAPLE
            tmp = it.argmax(axis=1)  # (T, C, H, W) -> (T, H, W)
            seg_stack = [sitk.GetImageFromArray(tmp[i].astype(np.int16)) for i in range(T)]
            # Run STAPLE algorithm
            STAPLE_seg_sitk = sitk.STAPLE(seg_stack, 1.0) # 1.0 specifies the foreground value
            # convert back to numpy array
            seg = sitk.GetArrayFromImage(STAPLE_seg_sitk)
            seg[seg < 0.000001] = 0
            # 3) Final layer only
            # seg = it[-1].argmax(axis=0)

            seg_masks.append(seg.astype(np.float32))
            # Estimate uncertainty metrics
            # entropy
            tmp = it  # TODO fix this
            entropy = -np.sum(np.mean(tmp, axis=0) * np.log(np.mean(tmp, axis=0) + 1e-5), axis=0)
            norm_entropy = normalise(entropy)
            entropy_maps.append(norm_entropy)
            avg_entropy.append(norm_entropy.sum())  # if active_learning_mode or self_training_mode else norm_entropy.mean())
            # variance DO it after argmax
            variance = it.argmax(axis=1).var(axis=0)
            norm_variance = normalise(variance)
            variance_maps.append(norm_variance)
            avg_variance.append(norm_variance.sum())  # if active_learning_mode or self_training_mode else norm_variance.mean())
            # mutual information
            expected_entropy = -np.mean(np.sum(tmp * np.log(tmp + 1e-5), axis=1), axis=0)
            mi = entropy - expected_entropy
            norm_mi = normalise(mi)
            mi_maps.append(norm_mi)
            avg_mi.append(norm_mi.sum())  # if active_learning_mode or self_training_mode else norm_mi.mean())

            # caluclate segmentation scores
            evals = get_evaluations(seg, lbl, spacing=(1, 1))
            dsc_all.append(evals['dsc_seg'])
            hd_all.append(evals['hd'])
            mhd_all.append(evals['mhd'])
            # calculate calibration scores
            tmp = it.mean(axis=0)
            y_true = np.stack((1-lbl, lbl))
            nll = -np.mean(np.sum(y_true * np.log(tmp), axis=0))
            nll_all.append(nll)

        results = dict()
        results['pid'] = outnames
        results['dsc_norm'] = dsc_all
        results['hd'] = hd_all
        results['mhd'] = mhd_all
        results['nll'] = nll_all
        results['avg_entropy'] = avg_entropy
        results['avg_variance'] = avg_variance
        results['avg_mi'] = avg_mi
        results['status'] = statuses
        results['pathologies'] = pathologies
        results['aula'] = area_under_agreement

        # Save ROIs with segmentations and uncertainties
        if not active_learning_mode:
            for idx in range(len(images)):
                # out_name = f'case_{idx}'
                out_name = outnames[idx]
                # images will be arranged the following way:
                # image with gt and seg outline + 3 images superimposed with 3 unc metrics
                gt_outline = labels[idx] - binary_erosion(binary_erosion(labels[idx]))  # two pix boundary
                seg_outline = seg_masks[idx] - binary_erosion(binary_erosion(seg_masks[idx]))  # two pix boundary
                # image with gt and seg outlines
                image_gt_seg = np.dstack((normalise(images[idx], 255, 0), normalise(images[idx], 255, 0), normalise(images[idx], 255, 0))).astype(np.uint8)
                image_gt_seg[gt_outline>0, 0] = 0
                image_gt_seg[gt_outline>0, 1] = 255
                image_gt_seg[gt_outline>0, 2] = 0
                image_gt_seg[seg_outline>0, 0] = 0
                image_gt_seg[seg_outline>0, 1] = 0
                image_gt_seg[seg_outline>0, 2] = 255

                image_rgb = cv2.cvtColor(normalise(images[idx], 1, 0), cv2.COLOR_GRAY2RGB)

                # image with superimposed variance map
                image_var = show_cam_on_image(image_rgb, variance_maps[idx])
                image_var[gt_outline>0, :] = 0

                # image with superimposed entropy map
                image_ent = show_cam_on_image(image_rgb, entropy_maps[idx])
                image_ent[gt_outline>0, :] = 0

                # image with superimposed MI map
                image_mi = show_cam_on_image(image_rgb, mi_maps[idx])
                image_mi[gt_outline>0, :] = 0

                if self_training_mode:
                    pil_image = Image.fromarray(normalise(images[idx], 255, 0).astype(np.uint8))
                    pil_seg = Image.fromarray(normalise(seg_masks[idx], 255, 0).astype(np.uint8))

                    pil_image_name = self_training_df.iloc[idx]['lesion_id']+'_'+str(Path(self_training_df.iloc[idx]['scan']).name)
                    pil_patch_path = Path(self_training_df.iloc[idx]['mask']).parent
                    pil_image_path = pil_patch_path / pil_image_name
                    pil_seg_path = Path(self_training_df.iloc[idx]['mask'])
                    tmp = str(pil_image_path).split('/')
                    tmp[5] = 'images'
                    pil_image_path = Path('/'.join(tmp))
                    pil_image_path.parent.mkdir(exist_ok=True, parents=True)

                    image_paths.append(str(pil_image_path))
                    seg_paths.append(str(pil_seg_path))
                    # Save patches
                    pil_image.save(str(pil_image_path))
                    pil_seg.save(str(pil_seg_path))
                else:
                    fig, ax = plt.subplots(1, 4, figsize=(9, 2))
                    # image with gt and seg outline
                    ax[0].imshow(image_gt_seg)
                    ax[0].set_title(f'Seg (DSC {dsc_all[idx]:.3f})')
                    # image with superimposed variance map
                    # trick to get colorbars
                    varax = ax[1].imshow(variance_maps[idx], cmap='jet')
                    ax[1].set_title(f'Var (Avg. {avg_variance[idx]:.3f})')
                    fig.colorbar(varax, ax=ax[1])
                    ax[1].imshow(image_var)
                    # image with superimposed entropy map
                    # trick to get colorbars
                    entax = ax[2].imshow(entropy_maps[idx], cmap='jet')
                    ax[2].set_title(f'Entropy (Avg. {avg_entropy[idx]:.3f})')
                    fig.colorbar(entax, ax=ax[2])
                    ax[2].imshow(image_ent)
                    # image with superimposed MI map
                    # trick to get colorbars
                    miax = ax[3].imshow(mi_maps[idx], cmap='jet')
                    ax[3].set_title(f'MI (Avg. {avg_mi[idx]:.2f})')
                    fig.colorbar(miax, ax=ax[3])
                    ax[3].imshow(image_mi)
                    for a in ax:
                        a.axis('off')
                    plt.tight_layout()
                    # plt.show()
                    plt.savefig(str(seg_out_path / out_name), bbox_inches='tight')  # , dpi=300)
                    # plt.savefig(str(seg_out_path / str(out_name+'.eps')), bbox_inches='tight' , dpi=300)
                    plt.close(fig=fig)
        
        if self_training_mode:
            results['scan'] = image_paths
            results['mask'] = seg_paths
            results['patient_id'] = self_training_df['patient_id'].tolist()
            results['laterality'] = self_training_df['laterality'].tolist()
            results['view'] = self_training_df['view'].tolist()
            results['study_id'] = self_training_df['study_id'].tolist()
            results['lesion_id'] = self_training_df['lesion_id'].tolist()
            results['unique_id'] = self_training_df['unique_id'].tolist()
            df = pd.DataFrame(results)
            df.to_csv(self_training_info_save_path, index=False)
            return df

        df = pd.DataFrame(results)
        if not active_learning_mode:
            df.to_csv(str(seg_out_path / results_csv_file))
            pickle.dump(calibration_pairs, open(str(seg_out_path / str(results_csv_file[:-4]+"-calibration_pairs.pkl")), "wb"))
            return seg_out_path / results_csv_file
        else:
            return df
