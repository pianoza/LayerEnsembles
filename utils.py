import warnings
import cv2
import enum
import torch
import numpy as np
import pandas as pd
import torchio as tio
import torch.nn.functional as F
from torchvision import transforms
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from scipy.ndimage.morphology import binary_erosion
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from operator import itemgetter, mul
from functools import partial, reduce

# # from segmentation_models_pytorch.losses.boundary_loss import one_hot2dist
# from typing import Callable, Tuple, Union
# from torch import Tensor

# # Boundary-loss functions -------- Begin
# D = Union[Image.Image, np.ndarray, Tensor]
# def gt_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
#         return transforms.Compose([
#                 lambda img: np.array(img)[...],
#                 lambda nd: torch.tensor(nd, dtype=torch.int64)[None, ...],  # Add one dimension to simulate batch
#                 # partial(class2one_hot, K=K),
#                 itemgetter(0)  # Then pop the element to go back to img shape
#         ])

# def dist_map_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
#         return transforms.Compose([
#                 gt_transform(resolution, K),
#                 lambda t: t.cpu().numpy(),
#                 partial(one_hot2dist, resolution=resolution),
#                 lambda nd: torch.tensor(nd, dtype=torch.float32)
#         ])
# # Boundary-loss functions --------- End

class Task(enum.Enum):
    CLASSIFICATION = 'classification'
    SEGMENTATION = 'segmentation'
    REGRESSION = 'regression'
class Organ(enum.Enum):
    BREAST = 'breast'
    HEART = 'heart'

def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    # Taken from: https://www.kaggle.com/bguberfain/elastic-transform-for-data-augmentation
    # Function to distort image
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

def get_deformed_samples(configs, pool, num_samples):
    synthetic_subjects = []
    counter = 0
    while counter < num_samples:
        subject = pool[np.random.randint(0, len(pool))]
        img = np.squeeze(subject['scan'])
        mask = np.squeeze(subject['mask'])
        # Merge images into separete channels (shape will be (cols, rols, 2))
        im_merge = np.concatenate((img[..., None], mask[..., None]), axis=2)
        # Apply transformation on image
        im_merge_t = elastic_transform(image=im_merge,
                                       alpha=im_merge.shape[1] * 8,
                                       sigma=im_merge.shape[1] * 0.1,
                                       alpha_affine=im_merge.shape[1] * 0.01)
        # Split image and mask
        im_t = np.float32(im_merge_t[..., 0])
        im_mask_t = im_merge_t[..., 1]
        mask_t = np.zeros_like(im_mask_t).astype(np.uint8)
        mask_t[im_mask_t > 0.5] = 1
        synthetic_subjects.append(
            tio.Subject(
                scan=tio.ScalarImage(tensor=torch.from_numpy(im_t[None, ..., None])),
                mask=tio.LabelMap(tensor=torch.from_numpy(mask_t[None, ..., None])),
            )
        )
        counter += 1
        if configs.SYNTH_SAVE_IMAGES:
            synth_image_stack(img, mask, im_t, mask_t, configs.SYNTH_SAVE_PATH, counter)
    return synthetic_subjects

def synth_image_stack(img, out, mask, orig, save_path, name):
    outline = mask - binary_erosion(mask)
    outl = np.dstack((out, out, out))
    outl[outline>0, 0] = 1
    outl[outline>0, 1] = 0
    outl[outline>0, 2] = 0

    img = normalise(img, 255, 0)
    out = normalise(out, 255, 0)
    mask = normalise(mask, 255, 0)
    orig = normalise(orig, 255, 0)
    outl = normalise(outl, 255, 0)

    img = np.dstack((img, img, img))
    out = np.dstack((out, out, out))
    mask = np.dstack((mask, mask, mask))
    orig = np.dstack((orig, orig, orig))

    img_mask_pair = np.concatenate((img, mask, out, orig, outl), axis=1).astype(np.uint8)  # side by side
    # img_mask_pair = np.concatenate((mask, out, orig), axis=1).astype(np.uint8)  # side by side
    im = Image.fromarray(img_mask_pair)
    im = im.convert("L")
    save_path.mkdir(exist_ok=True)
    im.save(str(save_path / str(name))+'.png')  # , quality=100, dpi=(300, 300))
    # im.save(str(save_path / str(name))+'.eps', quality=100, dpi=(300, 300))

def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = True,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.
    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    source: https://github.com/jacobgil/pytorch-grad-cam/blob/05fad53b1f1c324e1d7584c688c71d69cd4e3296/pytorch_grad_cam/utils/image.py#L31
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        print(np.max(img))
        raise Exception("The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def normalise(data, nmax=1., nmin=0.):
    return (data-data.min()) * ((nmax - nmin) / (data.max() - data.min() + 1e-8)) + nmin

def enable_dropout(m):
    counter = 0
    for module in m.modules():
        if module.__class__.__name__.startswith('Dropout'):
            module.train()
            counter += 1
    return counter

class ResampleToMask(object):
    def __init__(self, im_size):
        self.im_size = im_size

    def bbox2(self, img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax

    def __call__(self, sample):
        mask = np.squeeze(sample['mask'])
        rmin, rmax, cmin, cmax = self.bbox2(mask)
        w, h = (rmax-rmin), (cmax-cmin)
        lc = np.max((w, h))
        scale_ratio = (lc + 1.) / (self.im_size / 2.)
        resample = tio.Resample((scale_ratio, scale_ratio, 1))
        sample = resample(sample)
        return sample

class RandomMammoArtefact():
    pass

def dice_loss(logits, true, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        true: a tensor of shape [B, 1, H, W].
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1).to(torch.int64)]
        # to .contigious() to suppress channels_last mixed tensor memory format
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        # true_1_hot = true_1_hot.permute(0, 3, 1, 2).contiguous().float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)

def make_folders(base_path, experiment_name):
    ''' Create experiment folder with subfolders figures, models, segmentations
    Arguments
        :pathlib.Path base_path: where the experiment folder path will be created
        :str experiment_name: all experiment related outputs will be here

    @returns a tuple of pathlib.Path objects
    (models_path, figures_path, seg_out_path)
    '''
    results_path = Path(base_path / experiment_name)
    figures_path = results_path / 'figures'
    models_path = results_path / 'models'
    seg_out_path = results_path / 'segmentations'
    if not results_path.exists():
        results_path.mkdir(parents=True)
        figures_path.mkdir()
        models_path.mkdir()
        seg_out_path.mkdir()

    return models_path, figures_path, seg_out_path


def variable_batch_collate(batch):
    '''Collate function for batches with images of different sizes.
    Arguments
        :list batch: list of tuples (data, target)
    Returns
        :tuple (data, target)
    '''
    data = [item['scan']['data'] for item in batch]
    target = [item['status'] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]

def bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # Save checkpoints if model is a list (Ensemble)
        if isinstance(model, list):
            for i, m in enumerate(model):
                torch.save(m.state_dict(), self.path[i])
        else:
            torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def prepare_batch(batch, device):
    inputs = batch['scan'][tio.DATA].squeeze(-1).to(device)
    targets = batch['mask'][tio.DATA].squeeze(-1).to(device)
    return inputs, targets

def forward(model, inputs):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        logits = model(inputs)
    return logits

class Action(enum.Enum):
    TRAIN = 'Training'
    VALIDATE = 'Validation'

def run_epoch(action, loader, model, criterion, optimiser, device, num_training_subjects=None):
    is_training = action == Action.TRAIN
    epoch_losses = []
    model.train(is_training)
    for batch in tqdm(loader, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
        inputs, targets = prepare_batch(batch, device)
        optimiser.zero_grad()
        with torch.set_grad_enabled(is_training):
            if num_training_subjects is not None:
                if is_training:
                    batch_loss = model.sample_elbo(inputs=inputs,
                                                labels=targets,
                                                criterion=criterion,
                                                sample_nbr=5,
                                                complexity_cost_weight=1./num_training_subjects) 
                    batch_loss.backward()
                    optimiser.step()
                else:
                    logits = forward(model, inputs)
                    batch_loss = criterion(logits, targets)
            else:
                logits = forward(model, inputs)
                batch_loss = criterion(logits, targets)
                if is_training:
                    batch_loss.backward()
                    optimiser.step()
            epoch_losses.append(batch_loss.item())
    epoch_losses = np.array(epoch_losses)
    print(f'{action.value} mean loss: {epoch_losses.mean():0.3f}')
    return epoch_losses.mean()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
