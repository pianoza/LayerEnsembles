import scipy
import warnings 
from scipy.ndimage.morphology import binary_erosion
import pickle
import cv2
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, average_precision_score, classification_report, precision_recall_curve, roc_curve
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
import torchio as tio
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import normalise, make_folders, EarlyStopping, \
                  show_cam_on_image, Task, Organ
from metrics import get_evaluations
from methods.commons import get_model_for_task, get_criterion_for_task, SingleOutputModel
from methods.transforms import get_transforms
from methods.loaders import get_loaders
from methods.evaluation_utils import get_segmentation_metrics, get_uncertainty_metrics, save_segmentation_images

from pathlib import Path
from PIL import Image
import SimpleITK as sitk 

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# torch.autograd.set_detect_anomaly(True)

class BaseMethod:
    def __init__(self, configs, layer_ensembles):
        # Task is either 'segmentation' or 'classification' or 'regression'
        # Organ is either 'breast' or 'heart'
        self.layer_ensembles = layer_ensembles
        self.task = configs.TASK
        self.organ = configs.ORGAN
        if self.organ == Organ.BREAST:
            self.num_classes = 2
        elif self.organ == Organ.HEART:
            self.num_classes = 4
        else:
            raise ValueError(f'Organ {self.organ} not supported')
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
        # Make experiment folders
        self.best_model_path, self.final_model_path, self.figures_path, self.metrics_out_path, self.results_csv_file = self.folders()
        self.target_shape = (1, 1, 256, 256)  # TODO fix this

    def run_routine(self, run_test=True):
        # Load data
        train_transforms, test_transforms = self.prepare_transforms()
        train_loader, val_loader, test_loader = self.loaders(self.dataset, train_transforms, test_transforms)
        # Load pretrained model if already exists and not overwriting
        if self.final_model_path.exists() and not self.overwrite:
            print(f'EXPERIMENT {self.experiment_name} EXISTS!')
            print(f'TESTING ONLY')
            # self.model = self.prepare_model(self.final_model_path)
            self.model, self.intermediate_layers = self.prepare_model(target_shape=self.target_shape, model_path=self.best_model_path)
        else:
            # If not exists or overwriting, create new model and start training
            # Initialise model, loss, optimiser, scheduler, and early_stopping
            self.model, self.intermediate_layers = self.prepare_model(self.target_shape)
            # If tensorboard is enabled, create a new writer
            if self.configs.TENSORBOARD:
                self.tensorboard()
            self.criterion = self.prepare_criterion()
            self.optimizer = self.prepare_optimizer(self.model)
            self.scheduler = self.prepare_scheduler(self.optimizer)
            self.early_stopping = self.prepare_early_stopping(self.best_model_path)
            # Run training
            self.train(train_loader, val_loader)

        if run_test:
            # Run testing
            self.test(test_loader)

        # Close tensorboard writer
        if self.tb_writer is not None:
            self.tb_writer.close()
    
    def folders(self):
        models_path, figures_path, metrics_out_path = make_folders(self.results_path, self.experiment_name)
        best_model_path = models_path / 'best_model.pt'
        final_model_path = models_path / 'final_model.pt'
        results_csv_file = f'results_{self.experiment_name}.csv'
        return best_model_path, final_model_path, figures_path, metrics_out_path, results_csv_file
    
    def tensorboard(self):
        if not self.tboard_exp.is_dir():
            self.tboard_exp.mkdir(parents=True, exist_ok=True)
        dirs = [d for d in self.tboard_exp.iterdir()]
        next_run = len(dirs)+1
        self.tb_writer = SummaryWriter(log_dir=self.tboard_exp / str(next_run))
        # # add graph to tensorboard
        # self.tb_writer.add_graph(self.model, torch.rand(1, 1, 256, 256).to(self.configs.DEVICE))
    
    def prepare_model(self, target_shape, model_path=None):
        model, intermediate_layers = get_model_for_task(self.task, self.organ, self.layer_ensembles, target_shape, encoder_weights=None)
        if model_path is not None:
            model.load_state_dict(torch.load(model_path))
        model.to(self.configs.DEVICE)
        return model, intermediate_layers
    
    def prepare_criterion(self):
        # This could be directly called in self.run_routine() but let's keep it here for now
        classes = list(range(self.num_classes))
        criterion = get_criterion_for_task(self.task, classes)
        return criterion
    
    def prepare_optimizer(self, model):
        # A separate function in case we want to add different optimizers
        return torch.optim.Adam(model.parameters(), lr=self.configs.LR)
    
    def prepare_scheduler(self, optimizer):
        # A separate function in case we want to add different schedulers
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.configs.LR_DECAY_FACTOR, patience=self.configs.SCHEDULER_PATIENCE,
                                                          min_lr=self.configs.LR_MIN, verbose=True)

    def prepare_early_stopping(self, best_model_path):
        # A separate function in case we want to add different early stopping strategies
        return EarlyStopping(patience=self.configs.EARLY_STOPPING_PATIENCE, verbose=True, path=best_model_path)
    
    def prepare_transforms(self, target_image_size=256,
                           min_intensity=0, max_intensity=1,
                           min_percentile=0, max_percentile=100,
                           perturb_test=False,
                           ):
        # This could be called directly in self.run_routine(), but keep it here as it may change in the future
        train_transforms, test_transforms = get_transforms(self.task, self.num_classes, target_image_size, min_intensity, max_intensity, min_percentile, max_percentile, perturb_test)
        return train_transforms, test_transforms

    def loaders(self, dataset, train_transforms, test_transforms, train_val_test_split=(0.60, 0.20, 0.20), batch_size=None, sampler=None):
        # This could be called directly in self.run_routine(), but keep it here as it may change in the future
        train_loader, val_loader, test_loader = get_loaders(dataset, train_transforms, test_transforms, train_val_test_split, batch_size, sampler)
        return train_loader, val_loader, test_loader
    
    def train(self, train_loader, val_loader):
        for alpha, epoch in zip(np.linspace(0.01, 0.99, num=self.num_epochs), range(self.num_epochs)):
            # Train
            train_loss, all_head_losses = self.train_epoch(self.model, train_loader, self.criterion, self.optimizer, alpha)
            print(f'EPOCH {epoch}: All head losses:', all_head_losses)
            # Validate
            if self.task == Task.SEGMENTATION:
                val_loss, val_dice = self.validate_epoch(self.model, val_loader, self.criterion, alpha)
            elif self.task == Task.CLASSIFICATION:
                val_loss, preds, truths = self.validate_epoch(self.model, val_loader, self.criterion, alpha)
            else:
                raise ValueError(f'Unknown task: {self.task}')
            # Update scheduler and early stopping
            self.scheduler.step(val_loss)
            if self.task == Task.CLASSIFICATION:
                clf_report = classification_report(truths, preds.argmax(axis=1), digits=4, output_dict=True)
                val_loss = -clf_report['weighted avg']['f1-score']  # Negative because we want to maximize the metric
            self.early_stopping(val_loss, self.model)
            # Push to tensorboard if enabled
            if self.tb_writer is not None:
                log_losses = {'train': train_loss, 'val': val_loss} 
                if self.task == Task.SEGMENTATION:
                    log_seg_metrics = {'val_dice': val_dice}
                    log_losses.update(log_seg_metrics)
                elif self.task == Task.CLASSIFICATION:
                    # preds are softmax probabilities, need to argmax them for the classification report
                    # clf_report = classification_report(truths, preds.argmax(axis=1), digits=4, output_dict=True)
                    log_clf_metrics = {'val_acc': clf_report['accuracy'], 'val_precision': clf_report['weighted avg']['precision'], 'val_recall': clf_report['weighted avg']['recall'], 'val_f1': clf_report['weighted avg']['f1-score']}
                    log_losses.update(log_clf_metrics)
                else:
                    raise ValueError(f'Unknown task: {self.task}')
                self.tb_writer.add_scalars("Losses", log_losses, epoch)
                # plot validation
                if epoch == 1 or epoch % self.plot_validation_frequency == 0:
                    if self.task == Task.SEGMENTATION:
                        self.plot_validation(epoch, val_loader=val_loader)
                    elif self.task == Task.CLASSIFICATION:
                        self.plot_validation(epoch, preds=preds, truths=truths)
                    else:
                        raise ValueError(f'Unknown task: {self.task}')
                self.tb_writer.flush()

            if self.early_stopping.early_stop:
                print(f'EARLY STOPPING at EPOCH {epoch+1}')
                break

        torch.save(self.model.state_dict(), self.final_model_path)
        return train_loss, val_loss
    
    def plot_validation(self, epoch, val_loader=None, preds=None, truths=None):
        if self.task == Task.SEGMENTATION:
            # Plot segmentation masks and images | Displayed in tensorboard
            one_batch = next(iter(val_loader))
            batch_img, batch_label = self.prepare_batch(one_batch, self.device)
            with torch.set_grad_enabled(False):
                outputs = self.forward(self.model, batch_img)
                if self.layer_ensembles:
                    # average over layer ensembles
                    batch_seg = torch.mean(torch.stack([output for _, output in outputs.items()], dim=0), dim=0)
                    batch_seg = batch_seg.argmax(dim=1).unsqueeze(1)
                else:
                    batch_seg = outputs.argmax(axis=1).unsqueeze(1)
                # TODO Fix this for multi-class segmentation!!!
                # batch_label = batch_label.argmax(axis=1).unsqueeze(1)
                slices = torch.cat((batch_img[:8], batch_seg[:8], batch_label[:8]))
                batch_plot = make_grid(slices, nrow=8, normalize=True, scale_each=True)
                self.tb_writer.add_image(f'ImgSegLbl/val_epoch_{epoch+1}', batch_plot, global_step=epoch)
        elif self.task == Task.CLASSIFICATION:
            # Plot ROC curve | Saved in results/EXPERIMENT_NAME/figures/epoch_X.png
            # Classification labels: 0 - benign, 1 - malignant
            fpr, tpr, _ = roc_curve(truths, preds[:, 1])
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots(figsize=(8, 7))
            ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
            ax.plot(fpr, tpr, color='b', label=r'ROC (AUC = %0.2f)' % (roc_auc), lw=2, alpha=.8)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'Receiver operating characteristic | Epoch {epoch+1}')
            ax.legend(loc="lower right")
            fig.savefig(f'{self.figures_path}/epoch_{epoch+1}.png')
            plt.close(fig)
    
    def forward(self, model, inputs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            logits = model(inputs)
        return logits

    def prepare_batch(self, batch, device):
        '''This function always tries to pad the batch samples to the same size.
        '''
        # pad to match longest height and width in a batch
        max_height = max([x.shape[1] for x in batch[0]])
        max_width = max([x.shape[2] for x in batch[0]])
        # The needed padding is the difference between the
        # max width/height and the image's actual width/height.
        inputs = torch.stack([
            F.pad(img, [0, 0, 0, max_width - img.size(2), 0, max_height - img.size(1), 0, 0])
            for img in batch[0]
        ], dim=0).squeeze(-1).to(device)

        if self.task == Task.SEGMENTATION:
            # pad the masks too and hope it will be the same as the input paddings
            targets = torch.stack([
                F.pad(img, [0, 0, 0, max_width - img.size(2), 0, max_height - img.size(1), 0, 0])
                for img in batch[1]
            ], dim=0).squeeze(-1).to(device)
        elif self.task == Task.CLASSIFICATION:
            targets = batch[1].to(device)
        else:
            raise ValueError(f'Unknown task: {self.task}')

        return inputs, targets
    
    def train_epoch(self, model, train_loader, criterion, optimizer, alpha):
        model.train()
        train_loss = 0
        all_train_losses = []
        for batch_idx, batch in enumerate(train_loader):
            data, target = self.prepare_batch(batch, self.device)
            optimizer.zero_grad()
            outputs = model(data)
            if self.layer_ensembles:
                losses = [criterion(output, target) for _, output in outputs.items()]  # DiceLoss and CrossEntropyLoss <- accepts logits
                total_loss = 0
                all_train_losses.append([loss.item() for loss in losses])
                for loss in losses:
                    total_loss = total_loss + loss
            else:
                total_loss = criterion(outputs, target)
                all_train_losses.append(total_loss.item())
            total_loss.backward()
            train_loss += total_loss
            optimizer.step()
        all_train_losses = np.asarray(all_train_losses).mean(axis=0)
        return train_loss / len(train_loader), all_train_losses

    def validate_epoch(self, model, val_loader, criterion, alpha):
        if self.task == Task.SEGMENTATION:
            model.eval()
            val_loss = 0
            val_dice = []
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    data, target = self.prepare_batch(batch, self.device)
                    outputs = model(data)
                    if self.layer_ensembles:
                        # average the outputs of all heads
                        # TODO TEST THIS!!!
                        output = torch.stack([output for _, output in outputs.items()], dim=1).mean(dim=0)
                        seg = torch.argmax(output, dim=1).detach().cpu().numpy()
                    else:
                        seg = torch.argmax(outputs.squeeze(), dim=1).detach().cpu().numpy()
                    # lbl = torch.argmax(target.squeeze(), dim=1).detach().cpu().numpy()
                    # TODO MULTI-CLASS SEGMENTATION!!! Maybe we should bring back the OneHot?
                    lbl = target.squeeze().detach().cpu().numpy()
                    for s, l in zip(seg, lbl):
                        evals = get_evaluations(s, l, spacing=(1, 1))
                        val_dice.append(evals['dsc_seg'])
                    if self.layer_ensembles:
                        losses = [criterion(output, target) for _, output in outputs.items()]  # DiceLoss and CrossEntropyLoss <- accepts logits
                        total_loss = 0
                        for loss in losses:
                            total_loss = total_loss + loss
                    else:
                        total_loss = criterion(outputs, target)
                    val_loss += total_loss.item()
            assert all([dsc <= 1.0001 for dsc in val_dice])
            return val_loss / len(val_loader), sum(val_dice) / len(val_dice)
        elif self.task == Task.CLASSIFICATION:
            model.eval()
            val_loss = 0
            all_predictions = []
            all_truths = []
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    data, target = self.prepare_batch(batch, self.device)
                    outputs = model(data)
                    # loss = criterion(outputs[-1], target)  # CrossEntropyLoss <- accepts logits
                    losses = [criterion(output, target) for _, output in outputs.items()]
                    total_loss = 0
                    for loss in losses:
                        total_loss = total_loss + loss
                    val_loss += total_loss
                    # TODO instead of taking the last head, take an average
                    # Softmax the last head
                    mean_outputs = torch.stack([out for _, out in outputs.items()]).mean(dim=0)
                    pred = mean_outputs.softmax(dim=1).detach().cpu().numpy()
                    all_predictions.append(pred)
                    all_truths.append(target.detach().cpu().numpy())
            all_predictions = np.concatenate(all_predictions, axis=0)
            all_truths = np.concatenate(all_truths, axis=0)
            return val_loss / len(val_loader), all_predictions, all_truths
        else:
            raise ValueError(f'Unknown task: {self.task}')

    def test(self, loader):
        if self.task == Task.SEGMENTATION:
            # Get predictions
            predictions, images, labels = self.run_test_passes_segmentation(loader)
            # Evaluate segmentation
            seg_metrics = get_segmentation_metrics(predictions, labels, T=self.configs.SKIP_FIRST_T)
            seg_metrics = pd.DataFrame(seg_metrics)
            if self.layer_ensembles:
                # Get uncertainty metrics
                uncertainty_metrics, entropy_maps, variance_maps, mi_maps = get_uncertainty_metrics(predictions, labels, self.configs.SKIP_FIRST_T)
                uncertainty_metrics = pd.DataFrame(uncertainty_metrics)
                seg_metrics = pd.concat([seg_metrics, uncertainty_metrics], axis=1)                
                kwargs = {'entropy_maps': entropy_maps, 'variance_maps': variance_maps, 'mi_maps': mi_maps, 'T': self.configs.SKIP_FIRST_T}
            else:
                kwargs = {}
            # Save results
            seg_metrics.to_csv(self.metrics_out_path / self.results_csv_file)
            save_segmentation_images(images, labels, predictions, self.figures_path, **kwargs)
        elif self.task == Task.CLASSIFICATION:
            iters, truths, images, cam_hmaps = self.run_test_passes_classification(self.model, loader, self.configs.SKIP_FIRST_T)
            # Save results
            self.save_classification_results(iters, truths, images, cam_hmaps)
        else:
            raise ValueError(f'Unknown task: {self.task}')

    def run_test_passes_classification(self, model, loader, T):
        model.eval()
        truths = []
        iters = []
        images = []
        cam_hmaps = []
        use_cuda = True if self.configs.DEVICE == 'cuda' else False
        # GradCAM models for each classification head
        # TODO automatic SingleOutputModel population based on the number of output_heads in the model
        all_outputs = self.intermediate_layers + ['final']
        num_intermediate_layers = len(all_outputs)
        cam_models = [SingleOutputModel(model, layer) for layer in all_outputs]
        # cam_models = [SingleOutputModel(model, ''), SingleOutputModel(model, 1),
        #               SingleOutputModel(model, 2), SingleOutputModel(model, 3), SingleOutputModel(model, 4)]
        # target_layers = [cam_models[0].model.encoder.relu,
        #                  cam_models[1].model.encoder.layer1[-1].bn2, cam_models[2].model.encoder.layer2[-1].bn2,
        #                  cam_models[3].model.encoder.layer3[-1].bn2, cam_models[4].model.encoder.layer4[-1].bn2,]
        # target_layers = [cam_models[0].model.output_heads[0][4],  # activation layer of the classification head
        #                  cam_models[1].model.output_heads[1][4], cam_models[2].model.output_heads[2][4],
        #                  cam_models[3].model.output_heads[3][4], cam_models[4].model.output_heads[4][4]]
        target_layers = [cam_models[i].model.output_heads[i][4] for i in range(num_intermediate_layers-1)] + [cam_models[-1].model.model.fc]
        neg_cam_target = [ClassifierOutputTarget(0)]
        pos_cam_target = [ClassifierOutputTarget(1)]
        for i, batch in enumerate(tqdm(loader)):
            with torch.no_grad():
                inputs, targets = self.prepare_batch(batch, self.device)
                images.append(inputs.cpu().numpy())
                truths.append(targets.cpu().numpy())
                outputs = self.forward(model, inputs)  # [-T:]  # skipping first N - T layers
                tmp = torch.stack([out for _, out in outputs.items()], dim=1) # (BS, T, C)
                iters.append(tmp)
            # GradCAM heatmaps
            # Now for the last layer only
            all_cams = []
            for k in range(num_intermediate_layers):
                cam = GradCAM(model=cam_models[k], target_layers=[target_layers[k]], use_cuda=use_cuda)
                neg_grayscale_cam = cam(input_tensor=inputs, targets=neg_cam_target)  #, aug_smooth=True, eigen_smooth=True)
                pos_grayscale_cam = cam(input_tensor=inputs, targets=pos_cam_target)  #, aug_smooth=True, eigen_smooth=True)
                grayscale_cam = np.stack([neg_grayscale_cam, pos_grayscale_cam], axis=3)  # (BS, H, W, 2)
                all_cams.append(grayscale_cam)
            cam_hmaps.append(all_cams)  # [N x [ClfHead0, ClfHead1, ...]]
        images = [im.squeeze() for bat in images for im in bat]
        # cam_hmaps = [hmap for bat in cam_hmaps for hmap in bat]
        iters = torch.cat(iters, dim=0) # (N*BS, T, C)
        truths = np.concatenate(truths, axis=0) # (N*BS,)
        return iters, truths, images, cam_hmaps

    def save_classification_results(self, iters, truths, images, cam_hmaps, active_learning_mode=False):
        """Save results of classification task.
        iters torch.Tensor (N, T, C)
        truths torch.Tensor (N,)
        metrics_out_path str
        results_csv_file str
        active_learning_mode bool
        """
        # average over all heads
        preds = iters.mean(dim=1)
        # softmax over all classes
        preds = preds.softmax(dim=1)
        # classification report
        print(classification_report(truths, preds.argmax(dim=1).cpu().numpy()))
        # split two classes
        positive = preds[:, 1]
        negative = preds[:, 0]
        # plot ROC curve
        fpr, tpr, _ = roc_curve(truths, positive.cpu().numpy())
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
        ax.plot(fpr, tpr, color='b', label=r'ROC (AUC = %0.2f)' % (roc_auc), lw=2, alpha=.8)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'Receiver operating characteristic | Test set')
        ax.legend(loc="lower right")
        fig.savefig(f'{self.figures_path}/TestSetROC.png')
        plt.close(fig)
        # plot Precision-Recall curve
        precision, recall, _ = precision_recall_curve(truths, positive.cpu().numpy())
        average_precision = average_precision_score(truths, positive.cpu().numpy())
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.plot([0, 1], [0.5, 0.5], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
        ax.plot(recall, precision, color='b', label=r'Precision-Recall curve (AP = %0.2f)' % (average_precision), lw=2, alpha=.8)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall curve | Test set')
        ax.legend(loc="lower right")
        fig.savefig(f'{self.figures_path}/TestSetPrecisionRecall.png')
        plt.close(fig)

        # uncertainty metrics
        preds = iters.softmax(dim=2)  # (N, T, C)
        calibration_pairs = []  # tuple (y_true, y_prob) only for the positive class
        entropy_all = []
        variance_all = []
        mi_all = []
        nll_all = []
        for i in range(preds.shape[0]):
            tmp = preds[i].cpu().numpy()  # (T, C)
            # entropy
            entropy = -np.sum(np.mean(tmp, axis=0) * np.log(np.mean(tmp, axis=0) + 1e-5), axis=0)
            entropy_all.append(entropy)
            # variance
            # positive = tmp[..., 1]  # (T, C) -> (T)
            variance = tmp.var(axis=0).sum()  # (T, C) -> (C) -> (1)
            variance_all.append(variance)
            # mutual information
            expected_entropy = -np.mean(np.sum(tmp * np.log(tmp + 1e-5), axis=1), axis=0)
            mi = entropy - expected_entropy
            mi_all.append(mi)
            # negative log likelihood
            y_true = np.eye(tmp.shape[-1])[truths[i]]
            y_pred = iters[i].mean(dim=0).softmax(dim=0).cpu().numpy()  # (T, C) -> (C)
            nll = -np.mean(np.sum(y_true * np.log(y_pred+1e-5), axis=0))
            nll_all.append(nll)
            # plot image with GradCAM heatmap
            image_rgb = cv2.cvtColor(normalise(np.squeeze(images[i]), 1, 0), cv2.COLOR_GRAY2RGB)
            fig, ax = plt.subplots(len(self.intermediate_layers)+1, 3)  #, figsize=(8, 15.5))
            plot_pos = []
            for x in range(len(self.intermediate_layers)+1):
                row = []
                for y in range(3):
                    row.append([x, y])
                plot_pos.append(row)
            for k in range(len(self.intermediate_layers)+1):  # +1 for the final output
                neg_cam_image = show_cam_on_image(image_rgb, np.squeeze(cam_hmaps[i][k][..., 0]))
                pos_cam_image = show_cam_on_image(image_rgb, np.squeeze(cam_hmaps[i][k][..., 1]))
                ax[plot_pos[k][0][0]][plot_pos[k][0][1]].imshow(image_rgb)
                gt = 'Positive' if truths[i] == 1 else 'Negative'
                y_pred_head = iters[i][k].softmax(dim=0).cpu().numpy()  # (T, C) -> (C)
                ax[plot_pos[k][0][0]][plot_pos[k][0][1]].set_title(f'Original image\n{gt}')
                ax[plot_pos[k][1][0]][plot_pos[k][1][1]].imshow(neg_cam_image)
                ax[plot_pos[k][1][0]][plot_pos[k][1][1]].set_title(f'Negative GradCAM\nPredicted Prob {y_pred_head[0]:.2f}')
                ax[plot_pos[k][2][0]][plot_pos[k][2][1]].imshow(pos_cam_image)
                ax[plot_pos[k][2][0]][plot_pos[k][2][1]].set_title(f'Positive GradCAM\nPredicted Prob {y_pred_head[1]:.2f}')
                for a in ax[k]:
                    a.axis('off')
            fig.suptitle(f'true: {truths[i]} | pred: {y_pred.argmax()} | ent:{entropy:.2f} | var:{variance:.2f} | mi:{mi:.2f} | nll:{nll:.2f}')
            fig.tight_layout()
            plt.savefig(f'{self.metrics_out_path}/{i+1}.png', bbox_inches='tight')
            plt.close(fig)
            # neg_cam_image = show_cam_on_image(image_rgb, np.squeeze(cam_hmaps[i][..., 0]))
            # pos_cam_image = show_cam_on_image(image_rgb, np.squeeze(cam_hmaps[i][..., 1]))
            # fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            # ax[0].imshow(image_rgb)
            # gt = 'Positive' if truths[i] == 1 else 'Negative'
            # ax[0].set_title(f'Original image | {gt}')
            # ax[1].imshow(neg_cam_image)
            # ax[1].set_title(f'Negative GradCAM | Predicted Prob {y_pred[0]:.2f}')
            # ax[2].imshow(pos_cam_image)
            # ax[2].set_title(f'Positive GradCAM | Predicted Prob {y_pred[1]:.2f}')
            # for a in ax:
            #     a.axis('off')
            # plt.suptitle(f'true: {truths[i]} | pred: {y_pred.argmax()} | ent:{entropy:.2f} | var:{variance:.2f} | mi:{mi:.2f} | nll:{nll:.2f}')
            # plt.savefig(f'{self.metrics_out_path}/{i+1}.png', bbox_inches='tight')
            # plt.close(fig)
        
        preds = iters.mean(dim=1).softmax(dim=1).cpu().numpy()  # (N, T, C) -> (N, C)
        results = dict()
        results['truth'] = truths
        results['prediction'] = preds.argmax(axis=1)
        results['negative_prob'] = preds[:, 0]
        results['positive_prob'] = preds[:, 1]
        results['nll'] = nll_all
        results['entropy'] = entropy_all
        results['variance'] = variance_all
        results['mi'] = mi_all
        df = pd.DataFrame(results)
        df.to_csv(str(self.metrics_out_path / self.results_csv_file))


    def run_test_passes_segmentation(self, loader):
        self.model.eval()
        predictions = []
        images, labels = [], []
        with torch.no_grad():
            for batch in tqdm(loader):
                # test batch
                inputs, targets = self.prepare_batch(batch, self.device)
                outputs = self.forward(self.model, inputs)
                if self.layer_ensembles:
                    # outputs is a dict of {'layer_name': (B, C, H, W)} T times
                    # concatenate all the outputs to get a (B, T, C, H, W)
                    outputs = torch.stack([output for _, output in outputs.items()], dim=1)
                    predictions.append(outputs.cpu().numpy())
                else:
                    # outputs is a tensor of (B, C, H, W)
                    # just add it to the list
                    predictions.append(outputs.cpu().numpy())
                # save images and labels for visualization and evaluation
                images.append(inputs.squeeze().cpu().numpy())  # (B, H, W)
                labels.append(targets.squeeze().cpu().numpy())  # (B, H, W)
        # concatenate all batches to get (N, T, C, H, W) or (N, C, H, W)
        predictions = np.concatenate(predictions, axis=0)
        images = np.concatenate(images, axis=0)
        labels = np.concatenate(labels, axis=0)
        return predictions, images, labels

    def save_segmentation_results(self, iters, images, labels, statuses, pathologies, outnames, area_under_agreement, prediction_depth_all, active_learning_mode=False):
        # segmentation
        seg_masks = []
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

        for idx, (img, lbl, it) in enumerate(zip(images, labels, iters)):
            it = scipy.special.softmax(it, axis=1)
            calibration_pairs.append((lbl, it.mean(axis=0)[1]))
            # TODO - flag for using average or STAPLE
            # Final segmentation
            # 1) Average over all
            tmp = it.mean(axis=0)
            seg = tmp.argmax(axis=0)
            # 2) STAPLE
            # tmp = it.argmax(axis=1)  # (T, C, H, W) -> (T, H, W)
            # seg_stack = [sitk.GetImageFromArray(tmp[i].astype(np.int16)) for i in range(self.configs.TEST_T)]
            # # Run STAPLE algorithm
            # STAPLE_seg_sitk = sitk.STAPLE(seg_stack, 1.0) # 1.0 specifies the foreground value
            # # convert back to numpy array
            # seg = sitk.GetArrayFromImage(STAPLE_seg_sitk)
            # seg[seg < 0.000001] = 0
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

                fig, ax = plt.subplots(1, 4, figsize=(9, 2))
                # image with gt and seg outline
                ax[0].imshow(image_gt_seg)
                ax[0].set_title(f'Seg (DSC {dsc_all[idx]:.3f})')
                # image with superimposed variance map
                # trick to get colorbars
                varax = ax[1].imshow(variance_maps[idx], cmap='jet')
                ax[1].set_title(f'Var (Sum {avg_variance[idx]:.3f})')
                fig.colorbar(varax, ax=ax[1])
                ax[1].imshow(image_var)
                # image with superimposed entropy map
                # trick to get colorbars
                entax = ax[2].imshow(entropy_maps[idx], cmap='jet')
                ax[2].set_title(f'Entropy (Sum {avg_entropy[idx]:.3f})')
                fig.colorbar(entax, ax=ax[2])
                ax[2].imshow(image_ent)
                # image with superimposed MI map
                # trick to get colorbars
                miax = ax[3].imshow(mi_maps[idx], cmap='jet')
                ax[3].set_title(f'MI (Sum {avg_mi[idx]:.2f})')
                fig.colorbar(miax, ax=ax[3])
                ax[3].imshow(image_mi)
                for a in ax:
                    a.axis('off')
                plt.tight_layout()
                # plt.show()
                plt.savefig(str(self.metrics_out_path / out_name), bbox_inches='tight')  # , dpi=300)
                # plt.savefig(str(metrics_out_path / str(out_name+'.eps')), bbox_inches='tight' , dpi=300)
                plt.close(fig=fig)

        df = pd.DataFrame(results)
        if not active_learning_mode:
            df.to_csv(str(self.metrics_out_path / self.results_csv_file))
            pickle.dump(calibration_pairs, open(str(self.metrics_out_path / str(self.results_csv_file[:-4]+"-calibration_pairs.pkl")), "wb"))
            pickle.dump(prediction_depth_all, open(str(self.metrics_out_path / str(self.results_csv_file[:-4]+"-prediction_depth_all.pkl")), "wb"))
            return self.metrics_out_path / self.results_csv_file
        else:
            return df
