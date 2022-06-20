from calendar import c
import scipy
import warnings 
from scipy.ndimage.morphology import binary_erosion
from torchio.transforms.preprocessing.label.one_hot import OneHot
import pickle
import cv2
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from sklearn.model_selection import train_test_split
import torchio as tio
# import segmentation_models_pytorch as smp
from segmentation_models_pytorch.unet.model import Unet
from segmentation_models_pytorch.losses.dice import DiceLoss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import normalise, make_folders, EarlyStopping, \
                  show_cam_on_image
from metrics import get_evaluations
from methods.base_method import BaseMethod
from methods.randconv.randconv_transform import RandConvTransform
import SimpleITK as sitk 

class BaseMethodMnM(BaseMethod):
    def __init__(self, configs):
        super(BaseMethodMnM, self).__init__(configs)

    def prepare_transforms(self, target_image_size=128, min_intensity=0, max_intensity=1, min_percentile=0, max_percentile=100, masking_method=None, is_patches=False):
        train_transforms = tio.Compose([
            tio.RandomFlip(axes=(0, 1), p=0.5),
            # RandConvTransform(kernel_size=(1, 3, 5, 7), mixing=True, identity_prob=0.8),
            tio.RandomSwap((10, 10, 1), 10, p=0.3),
            tio.RandomAffine(
                scales=0,  # keep the same scale
                degrees=(-15, 15),  # rotate between -15 and 15 degrees
                translation=0, # don't translate
                isotropic=True, # keep the same resolution
                center='image', # rotate around the image center
                p=0.5, # 50% chance to apply
            ),
            tio.ZNormalization(),
            tio.OneHot(num_classes=4),
        ])
        test_transforms = tio.Compose([
            tio.ZNormalization(),
            tio.Compose([
                tio.RandomNoise((0.3, 0.301), (.7, .701), p=1.0),
            ], p=1.0),
            tio.OneHot(num_classes=4),
        ])
        return train_transforms, test_transforms

    def prepare_criterion(self):
        return torch.nn.CrossEntropyLoss(
            weight=torch.Tensor([.1, .3, .3, .3]).to(self.configs.DEVICE)
        )
        # return DiceLoss(
        #     mode='multilabel',
        #     classes=[1, 2, 3],
        #     log_loss=False,
        #     from_logits=True,
        #     smooth=0.0000001,
        #     ignore_index=None,
        # )
    

    def prepare_model(self, model_path=None):
        model = Unet(
            encoder_name="resnet18",
            encoder_weights=None,
            decoder_channels=(1024, 512, 256, 128, 64),
            decoder_attention_type='scse',
            in_channels=1,
            classes=4,
        )
        if model_path is not None:
            model.load_state_dict(torch.load(model_path))
        model.to(self.configs.DEVICE)
        return model

    def get_aula(self, layer_outputs):
        pass

    def run_test_passes(self, model, loader, iters, T):
        model.eval()
        images = []
        labels = []
        outnames = []
        layer_agreement = []  # (N, 4): [[LA_all, LA_cl1, LA_cl2, LA_cl3], [LA_all, LA_cl1, LA_cl2, LA_cl3], ...]
        all_agreements_for_plot = [] # (3, T)
        prediction_depth_all = [] # (N, T-1)
        with torch.no_grad():
            fig, ax = plt.subplots(1, 4, figsize=(12, 2.6))  # all and three classes separately
            for i, batch in enumerate(tqdm(loader)):  # in this case the loader is tio.SubjectsDataset
                prediction_depth = [] # [pred_d_all, pred_d_cl1, pred_d_cl2, pred_d_cl3]
                pid = batch['code']  # if isinstance(batch['code'][0], str) else batch['code'][0].item()
                sid = batch['vendor_name']  # [0] if isinstance(batch['vendor_name'][0], str) else batch['vendor_name'][0].item()
                lid = batch['timepoint']  # [0] if isinstance(batch['timepoint'][0], str) else batch['timepoint'][0].item()
                image, label, outname = [], [], []
                for slice in range(10):  # 10 is the number of slices in images
                    img = batch['scan'][tio.DATA][..., slice].unsqueeze(0)  # add batch dimension
                    lbl = batch['mask'][tio.DATA][..., slice] # .argmax(axis=1)
                    outname.append(f"case_{i}-pid_{pid}-sid_{sid}-lid_{lid}-slice_{slice+1}.png")
                    image.append(np.squeeze(img.numpy()))
                    label.append(np.squeeze(lbl.numpy()))
                    inputs = img.to(self.device)
                    outputs = self.forward(model, inputs)
                    outputs = outputs[-T:]  # skipping first N - T layers
                    for t in range(len(outputs)-T, len(outputs)):
                        iters[i, t, ..., slice] = outputs[t].detach().cpu().numpy()
                images.append(image)
                labels.append(label)
                outnames.append(outname)
                # BEGIN AULA
                # 1) DSC for 3D and all classes (1 + 2 + 3) -> Single AULA metric for the entire image
                #    WON'T BE IMPLEMENTED
                # 2) DSC for 3D and each class separately (1, 2, 3) -> 3 AULA metrics for the entire image
                agreement = []  # (T, 4)
                prev_layer = scipy.special.softmax(iters[i, 0], axis=0).argmax(axis=0) # (w, h, d)
                for j in range(1, len(iters[i, :])):
                    cl_agreement = []
                    cur_layer = scipy.special.softmax(iters[i, j], axis=0).argmax(axis=0)
                    evals = get_evaluations(prev_layer, cur_layer, spacing=(1.32, 1.32, 9.2))
                    cl_agreement.append(evals['dsc_seg'])
                    for cl in range(1, 4):
                        pl_c = np.zeros_like(prev_layer).astype(np.bool)
                        pl_c[prev_layer == cl] = True
                        cl_c = np.zeros_like(cur_layer).astype(np.bool)
                        cl_c[cur_layer == cl] = True
                        evals = get_evaluations(pl_c, cl_c, spacing=(1.32, 1.32, 9.2))
                        cl_agreement.append(evals['dsc_seg'])
                    agreement.append(cl_agreement)
                    prev_layer = cur_layer

                # calculate the prediction depth based on the layer agreement, threshold for agreement are set here
                # dsc_threshold = [0.95, 0.90, 0.90, 0.90]  # considered to be the same if agreement[i]-agreement[i-1] <= 1 - dsc_th (combined, lv, myo, rv)
                # for cl in range(4):
                #     class_agreement = [ag[cl] for ag in agreement]
                #     pred_depth = self.get_prediction_depth(class_agreement, 0.95)
                #     prediction_depth.append(pred_depth)
                # prediction_depth_all.append(prediction_depth)
                combined_agreement = [ag[0] for ag in agreement]
                prediction_depth_all.append(combined_agreement)

                # # Layer agreement for each slice and all classes together
                # agreement = []  # agreement between layer output
                # prev_layer = outputs[0].squeeze().cpu().numpy().argmax(axis=0) 
                # for j in range(1, len(outputs)+1):
                #     cur_layer = outputs[j-1].squeeze().cpu().numpy().argmax(axis=0)
                #     if j > 1:
                #         evals = get_evaluations(prev_layer, cur_layer, spacing=(1, 1)) # spacing=(1.32, 1.32, 9.2))
                #         agreement.append(evals['dsc_seg'])
                #         prev_layer = cur_layer

                agreement = np.asarray(agreement)
                single_image_layer_agreement = []  # (4,)
                single_image_agreement_plot = []  # (4, T)
                for cl in range(0, 4):  # 0 is for all classes together
                    ax[cl].plot(range(10-T+2, 11), agreement[:, cl], color='green', alpha=0.2)
                    single_image_agreement_plot.append(agreement[:, cl])
                    area_under_agreement = np.trapz(agreement[:, cl], dx=1)
                    single_image_layer_agreement.append(-area_under_agreement)
                layer_agreement.append(single_image_layer_agreement)
                all_agreements_for_plot.append(single_image_agreement_plot)

            all_agreements_for_plot = np.asarray(all_agreements_for_plot)
            for cl, title in zip(range(0, 4), ['Combined', 'LV', 'MYO', 'RV']):  # 0 is for all classes together
                all_agreement_for_class = all_agreements_for_plot[:, cl, :]
                ax[cl].plot(range(10-T+2, 11), np.mean(all_agreement_for_class, axis=0), color='red', linewidth=1.5, label='Mean')
                ax[cl].set_title(title)
                ax[cl].set_xlabel('Layer')
                if cl == 0:
                    ax[cl].set_ylabel('DSC')
                ax[cl].set_xlim(10-T+2, 10)
                ax[cl].set_xticks(range(10-T+2, 11))
                ax[cl].set_ylim(0, 1)
            plt.savefig('All_layer_agreement.png', dpi=300)
            plt.close()
            # END OF AULA
        # early layers learn simple functions and easier samples first, later layers learn more complex functions and memorize details.
        # The agreement between layers is a measure of how well the network is learning the details of the image.
        # It basically shows the evolution of the network as it learns more complex functions.
        # Hence, it can be used to measure the uncertainty of the network (both aleatoric and epistemic)
        # Also, AULA is a good measure of the uncertainty of segmentation, because Var, MI, and Entropy are pixel-wise, wheras AULA gives a global image uncertainty.
        # For 3D and multi-class, AULA can be computed in different ways:
        # 1) DSC for 3D and all classes (1 + 2 + 3) -> Single AULA metric for the entire image
        # 2) DSC for 3D and each class separately (1, 2, 3) -> 3 AULA metrics for the entire image
        # These ones need to be taken care of because heart slices are not aligned. Find the middle slice for each, then calculate the distance.
        # 3) DSC for each slice and all classes (1 + 2 + 3) -> 10 AULA metrics for the entire image | maybe not interesting
        # 4) DSC for each slice and each class separately (1, 2, 3) -> 30 AULA metrics for the entire image | discard this
        # Separating by class is important
        # MHD distance sometimes is better for cardiac
        # 3) and 2) should be the way to go as a way of showing the application of LayerEnsembles
        # We show the application of LE and AULA to detect segmentation failures for three classes. Also, show another application to localize the segmentation failure using slice-wise AULA.
        # If we show which slice has the failure, then it doesn't matter if we point out exactly what structure is wrong, because the doctor will look at all of them anyway. Cite (Jorg Sander et al. 2020) They are using slice error detection.
        return iters, images, labels, outnames, layer_agreement, prediction_depth_all

    def test(self, loader, seg_out_path, results_csv_file, T, active_learning_mode=False, w=128, h=128, d=10, batch_size=1):
        # Test images
        seg_masks = []
        len_dataset = len(loader) # 10 is the number of slices in images, loader is tio.SubjectsDataset

        iters = np.zeros((len_dataset, T, 4)+(w, h, d))

        iters, images, labels, outnames, area_under_agreement, prediction_depth_all = self.run_test_passes(self.model, loader, iters, T)
        df_outnames = ['-'.join(outname[0].split('-')[:-1]) for outname in outnames]

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
        dsc_classes_all = []
        hd_classes_all = []
        mhd_classes_all = []
        hd_all = []
        mhd_all = []
        nll_all = []

        for idx, (img, lbl, it) in enumerate(zip(images, labels, iters)):
            # it is a ndarray of shape (T, 4, w, h, d)
            # lbl now is a list of [ndarray(4, w, h), ndarray(4, w, h), ndarray(4, w, h), ndarray(4, w, h)]
            # convert lbl to a numpy array stacking on the last axis
            lbl = np.stack(lbl, axis=-1)
            it = scipy.special.softmax(it, axis=1)  # (T, 4, w, h, d) -> (T, 4, w, h, d)
            # UNCOMMENT THE NEXT LINE
            # calibration_pairs.append((lbl, it.mean(axis=0)))  # lbl (4, w, h, d) it (T, 4, w, h, d) -> (4, w, h, d)
            # TODO - flag for using average or STAPLE
            # Final segmentation
            # 1) Average over all  Deep Ensembles
            # tmp = it.mean(axis=0)
            # seg = tmp.argmax(axis=0)
            # # 2) STAPLE
            tmp = it.argmax(axis=1)  # (T, 4, h, w, d) -> (T, h, w, d)
            seg_stack = [sitk.GetImageFromArray(tmp[i].astype(np.uint64)) for i in range(T)]
            # Run STAPLE algorithm
            # STAPLE_seg_sitk = sitk.STAPLE(seg_stack, 1.0) # 1.0 specifies the foreground value
            STAPLE_seg_sitk = sitk.MultiLabelSTAPLE(seg_stack, 0)  # 0 is the label for undecided pixels
            # convert back to numpy array
            seg = sitk.GetArrayFromImage(STAPLE_seg_sitk)
            seg[seg < 0.000001] = 0
            # 3) Final layer only
            # seg = it[-1].argmax(axis=0)

            seg_masks.append(seg.astype(np.float32))
            # Estimate uncertainty metrics
            # entropy
            tmp = it  # TODO fix this tmp (T, 4, w, h, d)
            entropy = -np.sum(np.mean(tmp, axis=0) * np.log(np.mean(tmp, axis=0) + 1e-5), axis=0)
            norm_entropy = normalise(entropy)
            entropy_maps.append(norm_entropy)
            avg_entropy.append(norm_entropy.sum())  # if active_learning_mode or self_training_mode else norm_entropy.mean())
            # variance DO it after argmax
            variance = it.argmax(axis=1).var(axis=0) # (T, 4, w, h, d) -> (w, h, d)
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
            # each class independently
            dsc, hd, mhd = [], [], []
            for cl in range(1, 4):
                cl_seg = np.zeros_like(seg)
                cl_seg[seg == cl] = 1
                cl_lbl = np.zeros_like(lbl.argmax(axis=0))
                cl_lbl[lbl.argmax(axis=0) == cl] = 1
                evals = get_evaluations(cl_seg, cl_lbl, spacing=(1.32, 1.32, 9.2))
                dsc.append(evals['dsc_seg'])
                hd.append(evals['hd'])
                mhd.append(evals['mhd'])
            dsc_classes_all.append(dsc)
            hd_classes_all.append(hd)
            mhd_classes_all.append(mhd)
            # all classes together
            evals = get_evaluations(seg, lbl.argmax(axis=0), spacing=(1.32, 1.32, 9.2))  # spacing=(1, 1))
            dsc_all.append(evals['dsc_seg'])
            hd_all.append(evals['hd'])
            mhd_all.append(evals['mhd'])
            # calculate calibration scores
            tmp = it.mean(axis=0)  # (T, 4, w, h, d) -> (4, w, h, d)
            nll = -np.mean(np.sum(lbl * np.log(tmp), axis=0))
            nll_all.append(nll)
        dsc_classes_all = np.asarray(dsc_classes_all)
        hd_classes_all = np.asarray(hd_classes_all)
        mhd_classes_all = np.asarray(mhd_classes_all)
        results = dict()
        results['pid'] = df_outnames
        results['dsc_norm'] = dsc_all
        results['hd'] = hd_all
        results['mhd'] = mhd_all
        results['dsc_cl1'] = dsc_classes_all[:, 0]
        results['dsc_cl2'] = dsc_classes_all[:, 1]
        results['dsc_cl3'] = dsc_classes_all[:, 2]
        results['hd_cl1'] = hd_classes_all[:, 0]
        results['hd_cl2'] = hd_classes_all[:, 1]
        results['hd_cl3'] = hd_classes_all[:, 2]
        results['mhd_cl1'] = mhd_classes_all[:, 0]
        results['mhd_cl2'] = mhd_classes_all[:, 1]
        results['mhd_cl3'] = mhd_classes_all[:, 2]
        results['nll'] = nll_all
        results['avg_entropy'] = avg_entropy
        results['avg_variance'] = avg_variance
        results['avg_mi'] = avg_mi
        area_under_agreement = np.asarray(area_under_agreement)
        results['aula'] = area_under_agreement[:, 0]
        results['aula_cl1'] = area_under_agreement[:, 1]
        results['aula_cl2'] = area_under_agreement[:, 2]
        results['aula_cl3'] = area_under_agreement[:, 3]

        # Save ROIs with segmentations and uncertainties
        # In this case, we save each slice individually to have an instant visual feedback (not opening the whole volume in itksnap)
        if True: 
            if not active_learning_mode:
                for idx in range(len(images)):
                    for jdx in range(len(images[0])):
                        out_name = outnames[idx][jdx]
                        # images will be arranged the following way:
                        # image with gt and seg outline + 3 images superimposed with 3 unc metrics
                        image_gt_seg = np.dstack((normalise(images[idx][jdx], 255, 0), normalise(images[idx][jdx], 255, 0), normalise(images[idx][jdx], 255, 0))).astype(np.uint8)
                        gt_outline = np.zeros_like(seg_masks[idx][..., jdx]).astype(np.bool)
                        seg_colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
                        gt_all_outline = np.zeros_like(gt_outline).astype(np.bool)
                        for cl in range(1, 4):
                            cl_lbl = np.zeros_like(labels[idx][jdx].argmax(axis=0)).astype(np.bool)
                            cl_lbl[labels[idx][jdx].argmax(axis=0) == cl] = True
                            cl_seg = np.zeros_like(seg_masks[idx][..., jdx]).astype(np.bool)
                            cl_seg[seg_masks[idx][..., jdx] == cl] = True
                            gt_outline = (cl_lbl ^ binary_erosion(cl_lbl))  # one pix boundary
                            gt_all_outline = gt_all_outline | gt_outline
                            seg_outline = cl_seg ^ binary_erosion(cl_seg)  # one pix boundary
                            # image with gt and seg outlines
                            image_gt_seg[gt_outline>0, 0] = 0
                            image_gt_seg[gt_outline>0, 1] = 0
                            image_gt_seg[gt_outline>0, 2] = 0
                            image_gt_seg[seg_outline>0, 0] = seg_colors[cl-1][0]
                            image_gt_seg[seg_outline>0, 1] = seg_colors[cl-1][1]
                            image_gt_seg[seg_outline>0, 2] = seg_colors[cl-1][2]

                        image_rgb = cv2.cvtColor(normalise(images[idx][jdx], 1, 0), cv2.COLOR_GRAY2RGB)

                        # image with superimposed variance map
                        image_var = show_cam_on_image(image_rgb, variance_maps[idx][..., jdx])
                        image_var[gt_all_outline>0, :] = 0

                        # image with superimposed entropy map
                        image_ent = show_cam_on_image(image_rgb, entropy_maps[idx][..., jdx])
                        image_ent[gt_all_outline>0, :] = 0

                        # image with superimposed MI map
                        image_mi = show_cam_on_image(image_rgb, mi_maps[idx][..., jdx])
                        image_mi[gt_all_outline>0, :] = 0

                        fig, ax = plt.subplots(1, 4, figsize=(9, 2))
                        # image with gt and seg outline
                        ax[0].imshow(image_gt_seg)
                        ax[0].set_title(f'Seg (DSC {dsc_all[idx]:.3f})')
                        # image with superimposed variance map
                        # trick to get colorbars
                        varax = ax[1].imshow(variance_maps[idx][..., jdx], cmap='jet')
                        ax[1].set_title(f'Var (Sum {avg_variance[idx]:.3f})')
                        fig.colorbar(varax, ax=ax[1])
                        ax[1].imshow(image_var)
                        # image with superimposed entropy map
                        # trick to get colorbars
                        entax = ax[2].imshow(entropy_maps[idx][..., jdx], cmap='jet')
                        ax[2].set_title(f'Entropy (Sum {avg_entropy[idx]:.3f})')
                        fig.colorbar(entax, ax=ax[2])
                        ax[2].imshow(image_ent)
                        # image with superimposed MI map
                        # trick to get colorbars
                        miax = ax[3].imshow(mi_maps[idx][..., jdx], cmap='jet')
                        ax[3].set_title(f'MI (Sum {avg_mi[idx]:.2f})')
                        fig.colorbar(miax, ax=ax[3])
                        ax[3].imshow(image_mi)
                        for a in ax:
                            a.axis('off')
                        plt.tight_layout()
                        # plt.show()
                        plt.savefig(str(seg_out_path / out_name), bbox_inches='tight')  # , dpi=300)
                        # plt.savefig(str(seg_out_path / str(out_name+'.eps')), bbox_inches='tight' , dpi=300)
                        plt.close(fig=fig)

        df = pd.DataFrame(results)
        if not active_learning_mode:
            df.to_csv(str(seg_out_path / results_csv_file), index=False)
            # UNCOMMENT THE NEXT LINE
            # pickle.dump(calibration_pairs, open(str(seg_out_path / str(results_csv_file[:-4]+"-calibration_pairs.pkl")), "wb"))
            pickle.dump(prediction_depth_all, open(str(seg_out_path / str(results_csv_file[:-4]+"-prediction_depth_all.pkl")), "wb"))
            return seg_out_path / results_csv_file
        else:
            return df
