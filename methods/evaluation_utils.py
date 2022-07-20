import math
import configs
import scipy
import numpy as np
import pandas as pd
import cv2
import matplotlib
import SimpleITK as sitk
from metrics import get_evaluations
from PIL import Image
from matplotlib import pyplot as plt
from skimage.morphology import binary_erosion
from utils import normalise, show_cam_on_image


def get_segmentation_metrics(predictions, labels, T):
    '''Calculates the segmentation metrics
    Args:
        predictions: A numpy array of shape (N, C, H, W) or (N, T, C, H, W)
        labels: A numpy array of shape (N, H, W)
    Returns:
        A dictionary of metrics
    '''
    if configs.IS_LAYER_ENSEMBLES:
        num_heads = predictions.shape[1]
        assert T < num_heads, 'SKIP_FIRST_T must be less than the number of heads'
        preds = scipy.special.softmax(np.mean(predictions[:, T:, ...], axis=1), axis=1)  # (N, [T:], C, H, W) -> (N, C, H, W)
    else:
        preds = scipy.special.softmax(predictions, axis=1)
    num_classes = preds.shape[1]
    # convert labels to one hot
    labels = np.eye(num_classes)[labels.astype(np.uint8)]  # (N, H, W) -> (N, H, W, C)
    labels = np.transpose(labels, (0, 3, 1, 2))  # (N, H, W, C) -> (N, C, H, W)
    metrics = dict()
    for i in range(1, num_classes):  # ignore background
        metrics[f'dsc_{i}'] = []
        metrics[f'hd_{i}'] = []
        metrics[f'mhd_{i}'] = []

    for pred, label, in zip(preds, labels):
        # this fixes float precision issues when converting to uint8
        pred[pred > 0.9] = 1
        for i in range(1, num_classes):  # ignore background
            seg = pred[i].astype(np.uint8)
            gt = label[i].astype(np.uint8)
            evals = get_evaluations(seg, gt, spacing=(1, 1))
            metrics[f'dsc_{i}'].append(evals['dsc_seg'])
            metrics[f'hd_{i}'].append(evals['hd'] if not math.isinf(evals['hd']) else gt.shape[0] / 2)
            metrics[f'mhd_{i}'].append(evals['mhd'] if not math.isinf(evals['mhd']) else gt.shape[0] / 2)
    return metrics

def get_uncertainty_metrics(predictions, labels, T):
    '''Calculates the uncertainty metrics
    Args:
        predictions: A numpy array of shape (N, C, H, W) or (N, T, C, H, W)
        labels: A numpy array of shape (N, H, W) used to calculate the Negative Log-Likelihood
        T: The number of initial heads to skip in the ensemble to calculate uncertainty
    Returns:
        A dictionary of metrics (Entropy, Mutual Information, Variance, Negative Log-Likelihood)
    '''
    # (N, num_heads, C, H, W)
    num_heads = predictions.shape[1]
    assert T < num_heads, 'SKIP_FIRST_T must be less than the number of heads'
    num_classes = predictions.shape[2]

    # these are uncertainty heatmaps
    entropy_maps = []
    variance_maps = []
    mi_maps = []
    # these are uncertainty metrics for each sample
    entropy_sum = []
    variance_sum = []
    mi_sum = []
    # area under layer agreement curve AULA
    aula_per_class = dict()
    for i in range(1, num_classes):  # ignore background
        aula_per_class[f'aula_{i}'] = []
    # calibration (NLL)
    nlls = []
        
    # convert labels to one hot
    labels = np.eye(num_classes)[labels.astype(np.uint8)]  # (N, H, W) -> (N, H, W, C)
    labels = np.transpose(labels, (0, 3, 1, 2))  # (N, H, W, C) -> (N, C, H, W)

    for predicted, label in zip(predictions, labels):
        # softmax along channel axis (NH, C, H, W)
        pred = scipy.special.softmax(predicted[T:, ...], axis=1)
        # average along layer ensemble heads. Keep only the last T heads
        # ([T:], C, H, W) -> (C, H, W)
        avg_pred = np.mean(pred, axis=0)

        # calculate entropy
        entropy = -np.sum(np.mean(pred, axis=0) * np.log(np.mean(pred, axis=0) + 1e-5), axis=0)
        entropy_maps.append(entropy)
        entropy_sum.append(np.sum(entropy))
        
        # calculate variance (after argmax on channel axis)
        variance = np.var(np.argmax(pred, axis=1), axis=0)
        variance_maps.append(variance)
        variance_sum.append(np.sum(variance))

        # calculate mutual information
        expected_entropy = -np.mean(np.sum(pred * np.log(pred + 1e-5), axis=1), axis=0)
        mi = entropy - expected_entropy
        mi_maps.append(mi)
        mi_sum.append(np.sum(mi))

        # calculate Area Under Layer Agreement Curve (AULA)
        for i in range(1, num_classes):  # ignore background
            agreement = []
            prev_layer = pred[0, i]
            for j in range(1, num_heads-T):
                cur_layer = pred[j, i]
                dsc = get_evaluations(cur_layer, prev_layer, spacing=(1, 1))['dsc_seg']
                agreement.append(dsc)
                prev_layer = cur_layer
            aula = np.trapz(agreement, dx=1)
            aula_per_class[f'aula_{i}'].append(-aula)


        # calculate negative log-likelihood
        # label (C, H, W); avg_pred (C, H, W)
        nll = -np.mean(np.sum(label * np.log(avg_pred + 1e-5), axis=0))
        nlls.append(nll)
    
    metrics = {
        'entropy': entropy_sum,
        'variance': variance_sum,
        'mi': mi_sum,
        'nll': nlls
    }
    metrics.update(aula_per_class)
    return metrics, entropy_maps, variance_maps, mi_maps

def save_segmentation_images(images, labels, predictions, out_path, **kwargs):
    '''Saves the segmentation images
    Args:
        images: A numpy array of shape (N, H, W)
        labels: A numpy array of shape (N, H, W)
        predictions: A numpy array of shape (N, C, H, W) or (N, T, C, H, W)
        out_path: The path to save the images
        kwargs: A dictionary of keyword arguments, including `entropy_maps`, `variance_maps`, `mi_maps`, `T`
    '''
    if configs.IS_LAYER_ENSEMBLES:
        heatmaps = True
        num_heads = predictions.shape[1]
        entropy_maps = kwargs['entropy_maps']
        variance_maps = kwargs['variance_maps']
        mi_maps = kwargs['mi_maps']
        T = kwargs['T']
        assert T < num_heads, 'SKIP_FIRST_T must be less than the number of heads'
        preds = scipy.special.softmax(predictions, axis=2)  # softmax on channel axis
        preds = np.mean(preds[:, T:, ...], axis=1)  # (N, [T:], C, H, W) -> (N, C, H, W)
    else:
        heatmaps = False
        preds = scipy.special.softmax(predictions, axis=1)  # softmax on channel axis
    # one hot to label for predictions
    preds = np.argmax(preds, axis=1)  # (N, C, H, W) -> (N, H, W)
    for i in range(images.shape[0]):
        image = images[i]
        label = labels[i]
        pred = preds[i]

        # overlay prediction and label on image
        final_image = overlay_segmentation(image, label, pred)
        if heatmaps:
            entropy_map = entropy_maps[i]
            variance_map = variance_maps[i]
            mi_map = mi_maps[i]
            # overlay entropy on image
            overlay_entropy = overlay_heatmap(image, entropy_map)
            # overlay variance on image
            overlay_variance = overlay_heatmap(image, variance_map)
            # overlay mutual information on image
            overlay_mi = overlay_heatmap(image, mi_map)
            # concat heatmaps to image
            final_image = np.concatenate([final_image, overlay_entropy, overlay_variance, overlay_mi], axis=1)
        # save image
        final_image = final_image.astype(np.uint8)
        final_image = Image.fromarray(final_image)
        final_image.save(out_path / f'{i}.png')

def overlay_segmentation(image, label, pred):
    '''Overlays the segmentation on the image
    Args:
        image: A numpy array of shape (H, W)
        label: A numpy array of shape (H, W)
        pred: A numpy array of shape (H, W)
    Returns:
        sitk.Image of shape (H, W, 3)
    '''
    overlay = np.zeros((image.shape[0], image.shape[1], 3)).astype(np.uint8)
    overlay[:, :, 0] = normalise(image, 255, 0)
    overlay[:, :, 1] = normalise(image, 255, 0)
    overlay[:, :, 2] = normalise(image, 255, 0)
    # get segmentation contours
    pred_contours = normalise(get_contours(pred), 255, 0)
    pred_contours[pred_contours == 0] = overlay[pred_contours == 0, 0]
    label_contours = normalise(get_contours(label), 255, 0)
    label_contours[label_contours == 0] = overlay[label_contours == 0, 0]
    # label in green channel
    overlay[:, :, 1] = label_contours
    # prediction in blue channel
    overlay[:, :, 2] = pred_contours
    return overlay

def get_contours(segmentation):
    '''Gets the contours of the segmentation
    Args:
        segmentation: A numpy array of shape (H, W)
    Returns:
        A numpy array of shape (H, W)
    '''
    # get contours
    contours = segmentation - binary_erosion(binary_erosion(segmentation))
    return contours

def overlay_heatmap(image, heatmap):
    '''Overlays the heatmap on the image
    Args:
        image: A numpy array of shape (H, W)
        heatmap: A numpy array of shape (H, W)
    Returns:
        A numpy array of shape (H, W, 3)
    '''
    image_rgb = cv2.cvtColor(normalise(image, 1, 0), cv2.COLOR_GRAY2RGB)
    overlay = normalise(show_cam_on_image(image_rgb, heatmap), 255, 0)
    return overlay