import numpy as np
from scipy.ndimage import label
from scipy.ndimage import labeled_comprehension as lc
from scipy.ndimage.morphology import binary_erosion as imerode
from sklearn.neighbors import NearestNeighbors


def regionprops(mask):
    """
    Get region properties

    Inputs:
    - mask: 3D np.ndarray, input MRI mask

    Output:
    - regions: 3D np.ndarray, labeled version of the input mask array
      where each region is labeled with a unique label
    - labels: list of unique labels
    - volumes: list with region volumes
    """
    regions, num_regions = label(as_logical(mask))
    labels = np.arange(1, num_regions+1)
    volumes = lc(regions > 0, regions, labels, np.sum, int, 0)
    return regions, labels, volumes


def as_logical(mask):
    """
    convert to Boolean

    Inputs:
    - mask: 3D np.ndarray, input MRI mask

    Output:
    - Same mask binarized
    """
    return np.array(mask).astype(dtype=np.bool)


def num_regions(mask):
    """
    compute the number of regions from an input mask

    Inputs:
    - mask: 3D np.ndarray, input MRI mask

    Output:
    - num_regions: (int) number of candidate regions
    """
    regions, num_regions = label(as_logical(mask))
    return num_regions


def min_region(mask):
    """
    compute the volume of the smallest region from an input mask

    Inputs:
    - mask: 3D np.ndarray, input MRI mask

    Output:
    - min_region: (int) the size of the minimum region volume.
    """
    regions, num_regions = label(as_logical(mask))
    labels = np.arange(1, num_regions+1)
    mask = as_logical(mask)
    return np.min(lc(mask, regions, labels, np.sum, int, 0)) \
        if num_regions > 0 else 0


def max_region(mask):
    """
    compute the volume of the biggest region from an input mask

    Inputs:
    - mask: 3D np.ndarray, input MRI mask

    Output:
    - max_region: (int) the size of the maximum region volume.

    """
    regions, num_regions = label(as_logical(mask))
    labels = np.arange(1, num_regions+1)
    mask = as_logical(mask)
    return np.max(lc(mask, regions, labels, np.sum, int, 0)) \
        if num_regions > 0 else 0


def num_voxels(mask):
    """
    compute the number of voxels from an input mask

    Inputs:
    - mask: 3D np.ndarray, input MRI mask

    Output:
    - region volumen: (int) volume of the input mask (in voxels)

    """
    return np.sum(as_logical(mask))


def true_positive_seg(gt, mask):
    """
    compute the number of true positive voxels between an input mask an
    a ground truth (GT) mask

    Inputs:
    - gt: 3D np.ndarray, reference image (ground truth)
    - mask: 3D np.ndarray, input MRI mask

    Output:
    - (int) number of true positive voxels between the input and gt mask
    """
    a = as_logical(gt)
    b = as_logical(mask)

    return np.count_nonzero(np.logical_and(a, b))


def true_positive_det(gt, mask):
    """
    compute the number of positive regions between an input mask an
    a ground truth (GT) mask

    Inputs:
    - gt: 3D np.ndarray, reference image (ground truth)
    - mask: 3D np.ndarray, input MRI mask

    Output:
    - (int) number of true positive regions between the input and gt mask

    """
    regions, num_regions = label(as_logical(gt))
    labels = np.arange(1, num_regions+1)
    mask = as_logical(mask)
    tpr = lc(mask, regions, labels, np.sum, int, 0)

    return np.sum(tpr > 0)


def false_negative_seg(gt, mask):
    """
    compute the number of false negative voxels

    Inputs:
    - gt: 3D np.ndarray, reference image (ground truth)
    - mask: 3D np.ndarray, input MRI mask

    Output:
    - (int) number of false negative voxels between the input and gt mask

    """
    a = as_logical(gt)
    b = as_logical(mask)

    return np.count_nonzero(np.logical_and(a, np.logical_not(b)))


def false_negative_det(gt, mask):
    """
    compute the number of false negative regions between a input mask an
    a ground truth (GT) mask

    Inputs:
    - gt: 3D np.ndarray, reference image (ground truth)
    - mask: 3D np.ndarray, input MRI mask

    Output:
    - (int) number of false negative regions between the input and gt mask

    """
    regions, num_regions = label(as_logical(gt))
    labels = np.arange(1, num_regions+1)
    mask = as_logical(mask)
    tpr = lc(mask, regions, labels, np.sum, int, 0)

    return np.sum(tpr == 0)


def false_positive_seg(gt, mask):
    """
    compute the number of false positive voxels between a input mask an
    a ground truth (GT) mask

    Inputs:
    - gt: 3D np.ndarray, reference image (ground truth)
    - mask: 3D np.ndarray, input MRI mask

    Output:
    - (int) number of false positive voxels between the input and gt mask

    """
    a = as_logical(gt)
    b = as_logical(mask)

    return np.count_nonzero(np.logical_and(np.logical_not(a), b))


def false_positive_det(gt, mask):
    """
    compute the number of false positive regions between a input mask an
    a ground truth (GT) mask

    Inputs:
    - gt: 3D np.ndarray, reference image (ground truth)
    - mask: 3D np.ndarray, input MRI mask

    Output:
    - (int) number of false positive regions between the input and gt mask

    """
    regions, num_regions = label(as_logical(mask))
    labels = np.arange(1, num_regions+1)
    gt = as_logical(gt)

    return np.sum(lc(gt, regions, labels, np.sum, int, 0) == 0) \
        if num_regions > 0 else 0


def true_negative_seg(gt, mask):
    """
    compute the number of true negative samples between an input mask and
    a ground truth mask

    Inputs:
    - gt: 3D np.ndarray, reference image (ground truth)
    - mask: 3D np.ndarray, input MRI mask

    Output:
    - (int) number of true negative voxels between the input and gt mask

    """
    a = as_logical(gt)
    b = as_logical(mask)

    return np.count_nonzero(np.logical_and(np.logical_not(a),
                                           np.logical_not(b)))


def TPF_seg(gt, mask):
    """
    compute the True Positive Fraction (Sensitivity) between an input mask and
    a ground truth mask

    Inputs:
    - gt: 3D np.ndarray, reference image (ground truth)
    - mask: 3D np.ndarray, input MRI mask

    Output:
    - (float) Voxelwise true positive fraction between the input and gt mask

    """
    TP = true_positive_seg(gt, mask)
    GT_voxels = np.sum(as_logical(gt)) if np.sum(as_logical(gt)) > 0 else 0

    return float(TP) / GT_voxels


def TPF_det(gt, mask):
    """
    Compute the TPF (sensitivity) detecting candidate regions between an
    input mask and a ground truth mask

    Inputs:
    - gt: 3D np.ndarray, reference image (ground truth)
    - mask: 3D np.ndarray, input MRI mask

    Output:
    - (float) Regionwise true positive fraction between the input and gt mask

    """

    TP = true_positive_det(gt, mask)
    number_of_regions = num_regions(gt)

    return float(TP) / number_of_regions


def FPF_seg(gt, mask):
    """
    Compute the False positive fraction between an input mask and a ground
    truth mask

    Inputs:
    - gt: 3D np.ndarray, reference image (ground truth)
    - mask: 3D np.ndarray, input MRI mask

    Output:
    - (float) Voxelwise false positive fraction between the input and gt mask

    """
    b = float(num_voxels(mask))
    fpf = false_positive_seg(gt, mask) / b if b > 0 else 0

    return fpf


def FPF_det(gt, mask):
    """
    Compute the FPF detecting candidate regions between an
    input mask and a ground truth mask

    Inputs:
    - gt: 3D np.ndarray, reference image (ground truth)
    - mask: 3D np.ndarray, input MRI mask

    Output:
    - (float) Regionwise false positive fraction between the input and gt mask

    """

    FP = false_positive_det(gt, mask)
    number_of_regions = num_regions(mask)

    return float(FP) / number_of_regions if number_of_regions > 0 else 0


def DSC_seg(gt, mask):
    """
    Compute the Dice (DSC) coefficient betweeen an input mask and a ground
    truth mask. The DSC score is equal to the f1-score.

    Inputs:
    - gt: 3D np.ndarray, reference image (ground truth)
    - mask: 3D np.ndarray, input MRI mask

    Output:
    - (float) Voxelwise Dice coefficient between the input and gt mask

    """
    A = num_voxels(gt)
    B = num_voxels(mask)

    return 2.0 * true_positive_seg(gt, mask) / (A + B) \
        if (A + B) > 0 else 0


def AOV_seg(gt, mask):
    """
    Compute the Area Overlap coefficient betweeen an input mask and a ground
    truth mask == TPF. This is equivalent to the TPF fraction score.

    Inputs:
    - gt: 3D np.ndarray, reference image (ground truth)
    - mask: 3D np.ndarray, input MRI mask

    Output:
    - (float) Area overlap fraction between the input and gt mask

    """
    a = np.sum(np.logical_and(as_logical(gt), as_logical(mask)))
    b = num_voxels(as_logical(gt))

    return float(a) / b


def DSC_det(gt, mask):
    """
    Compute the Dice (DSC) coefficient betweeen an input mask and a ground
    truth mask

    Inputs:
    - gt: 3D np.ndarray, reference image (ground truth)
    - mask: 3D np.ndarray, input MRI mask

    Output:
    - (float) Regionwise Dice coefficient between the input and gt mask

    """
    A = num_regions(gt)
    B = num_regions(mask)

    return 2.0 * true_positive_det(gt, mask) / (A + B) \
        if (A + B) > 0 else 0


def PVE(gt, mask, type='absolute'):
    """
    Compute the volume difference error betweeen an input mask and a ground
    truth mask

    Inputs:
    - gt: 3D np.ndarray, reference image (ground truth)
    - mask: 3D np.ndarray, input MRI mask
    - type: controls if the error is relative or absolute (def: absolute)

    Output:
    - (float) relative / absolute difference in volume  between the input
       and gt mask

    """
    A = num_voxels(gt)
    B = num_voxels(mask)

    if type == 'absolute':
        pve = np.abs(float(B - A) / A)
    else:
        pve = float(B - A) / A

    return pve

def VSI(gt, mask, type='absolute'):
    """
    Compute the volumetrics similarity betweeen an input mask and a ground
    truth mask

    Inputs:
    - gt: 3D np.ndarray, reference image (ground truth)
    - mask: 3D np.ndarray, input MRI mask
    - type: controls if the error is relative or absolute (def: absolute)

    Output:
    - (float) relative / absolute difference in volume  between the input
       and gt mask

    """
    A = num_voxels(gt)
    B = num_voxels(mask)

    if type == 'absolute':
        pve = 1- np.abs(float(B - A) / (A + B))
    else:
        pve = float(B - A) / (A + B)

    return pve


def PPV_det(gt, mask):
    """
    Compute the positive predictive value (recall) for the detected
    regions between an input mask and a ground truth mask

    Inputs:
    - gt: 3D np.ndarray, reference image (ground truth)
    - mask: 3D np.ndarray, input MRI mask

    Output:
    - (float) Regionwise PPV coefficient between the input and gt mask

    """

    a = TPF_det(gt, mask)
    b = TPF_det(gt, mask) + FPF_det(gt, mask)

    return a / b if a > 0 else 0


def PPV_seg(gt, mask):
    """
    Compute the positive predictive value (recall) for the detected
    regions between an input mask and a ground truth mask

    Inputs:
    - gt: 3D np.ndarray, reference image (ground truth)
    - mask: 3D np.ndarray, input MRI mask

    Output:
    - (float) Voxelwise PPV coefficient between the input and gt mask

    """
    a = TPF_seg(gt, mask)
    b = TPF_seg(gt, mask) + FPF_seg(gt, mask)

    return a / b if a > 0 else 0


def f_score(gt, mask):
    """
    Compute a custom score between an input mask and a ground truth mask:

    F = 3 * DSC_s + TPF_d + (1- FPF) / DSC_s +TPF_d + (1-FPF)

    This score can be useful to optimize the best hyper-parameters based on
    different evaluation scores.

    Inputs:
    - gt: 3D np.ndarray, reference image (ground truth)
    - mask: 3D np.ndarray, input MRI mask

    Output:
    - (float) Custom f-score

    """

    a = 3.0 * DSC_seg(gt, mask) * TPF_det(gt, mask) * (1 - FPF_det(gt, mask))
    b = DSC_seg(gt, mask) + TPF_det(gt, mask) + (1 - FPF_det(gt, mask))

    return a / b if a > 0 else 0


def eucl_distance(a, b):
    """
    Euclidian distance between Region a and b

    Inputs:
    - a: 3D np.ndarray
    - b: 3D np.ndarray

    Output:
    - (float) Euclidian distances between A and B

    """
    nbrs_a = NearestNeighbors(n_neighbors=1,
                              algorithm='kd_tree').fit(a) if a.size > 0 else None
    nbrs_b = NearestNeighbors(n_neighbors=1,
                              algorithm='kd_tree').fit(b) if b.size > 0 else None
    distances_a, _ = nbrs_a.kneighbors(b) if nbrs_a and b.size > 0 else ([np.inf], None)
    distances_b, _ = nbrs_b.kneighbors(a) if nbrs_b and a.size > 0 else ([np.inf], None)

    return [distances_a, distances_b]


def surface_distance(gt, mask, spacing=list((1, 1, 1))):
    """
    Compute the surface distance between the input mask and a
    ground truth mask

    Inputs:
    - gt: 3D np.ndarray, reference image (ground truth)
    - mask: 3D np.ndarray, input MRI mask
    - spacing: sets the input resolution (def: (1, 1, 1))

    Output:
    - (float) Euclidian distance between gt and mask

    """
    a = as_logical(gt)
    b = as_logical(mask)
    a_bound = np.stack(np.where(
        np.logical_and(a, np.logical_not(imerode(a)))), axis=1) * spacing
    b_bound = np.stack(np.where(
        np.logical_and(b, np.logical_not(imerode(b)))), axis=1) * spacing
    return eucl_distance(a_bound, b_bound)


def mask_distance(gt, mask, spacing=list((1, 1, 1))):
    """
    Compute the mask distance between the input mask and the
    ground truth mask

    Inputs:
    - gt: 3D np.ndarray, reference image (ground truth)
    - mask: 3D np.ndarray, input MRI mask
    - spacing: sets the input resolution (def: (1, 1, 1))

    Output:
    - (float) mask distance between gt and mask

    """
    a = as_logical(gt)
    b = as_logical(mask)
    a_full = np.stack(np.where(a), axis=1) * spacing
    b_full = np.stack(np.where(b), axis=1) * spacing
    return eucl_distance(a_full, b_full)


def ASD(gt, mask, spacing=(1, 1, 1)):
    """
    Compute the average_surface_distance (ASD) between an input mask and a
    ground truth mask

    Inputs:
    - gt: 3D np.ndarray, reference image (ground truth)
    - mask: 3D np.ndarray, input MRI mask
    - spacing: sets the input resolution (def: (1, 1, 1))

    Output:
    - (float) Average surface distance between gt and mask

    - spacing: sets the input resolution
    """
    distances = np.concatenate(surface_distance(gt, mask, spacing))
    return np.mean(distances)


def HD(gt, mask, spacing=(1, 1, 1)):
    """
    Compute the Haursdoff distance between an input mask and a
    groud truth mask

    Inputs:
    - gt: 3D np.ndarray, reference image (ground truth)
    - mask: 3D np.ndarray, input MRI mask
    - spacing: sets the input resolution (def: (1, 1, 1))

    Output:
    - (float) Haursdoff distance betwwen Gt and mask


    - spacing: sets the input resolution
    """
    distances = surface_distance(gt, mask, spacing)
    return np.max([np.max(distances[0]), np.max(distances[1])])


def MHD(gt, mask, spacing=(1, 1, 1)):
    """
    Compute the modified Haursdoff distance between an input mask and a
    groud truth mask using the spacing parameter

    Inputs:
    - gt: 3D np.ndarray, reference image (ground truth)
    - mask: 3D np.ndarray, input MRI mask
    - spacing: sets the input resolution (def: (1, 1, 1))

    Output:
    - (float) Modified Haursdoff distance between mask and gt
    """

    distances = mask_distance(gt, mask, spacing)
    return np.max([np.mean(distances[0]), np.mean(distances[1])])


def mad(gt, mask):
    """
    Compute the median absolute error between two input masks

    Inputs:
    - gt: 3D np.ndarray, reference image (ground truth)
    - mask: 3D np.ndarray, input MRI mask

    Output:
    - (float) median absolute error between mask and gt
    """

    intersect = np.multiply(gt != 0, mask != 0)
    max_gt = gt[intersect].max()
    max_mask = mask[intersect].max()
    return np.median(
        np.abs(gt[intersect] / max_gt - mask[intersect] / max_mask))


def ssim(gt, mask):
    """
    Compute the structural similarity index between two input masks

    Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image
    quality assessment: from error visibility to structural similarity. IEEE
    transactions on image processing, 13, 600-612.

    Inputs:
    - gt: 3D np.ndarray, reference image (ground truth)
    - mask: 3D np.ndarray, input MRI mask

    Output:
    - (float) ssim coefficient between the two masks
    """

    A = gt / gt.max()
    B = mask / mask.max()
    intersect = np.multiply(A != 0, B != 0)
    ua = A[intersect].mean()
    ub = B[intersect].mean()
    oa = A[intersect].std() ** 2
    ob = B[intersect].std() ** 2
    oab = np.sum(
        np.multiply(
            A[intersect] - ua, B[intersect] - ub)) / (np.sum(intersect) - 1)

    # coeffients
    k1 = 0.01
    k2 = 0.03
    L = 1
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    num = (2*ua*ub + c1) * (2*oab + c2)
    den = (ua**2 + ub**2 + c1) * (oa + ob + c2)
    return num / den


def get_evaluations(gt, mask, spacing=(1, 1, 1)):
    """
    Helper function to compute all the evaluation metrics:

    - Segmentation volume, number of regions, min and max vol for each region
    - TPF (segmentation and detection)
    - FPF (segmentation and detection)
    - DSC (segmentation and detection)
    - PPV (segmentation and detection)
    - Volume difference
    - Haursdoff distance (standard and modified)
    - Custom f-score

    Inputs:
    - gt: 3D np.ndarray, reference image (ground truth)
    - mask: 3D np.ndarray, input MRI mask
    - spacing: sets the input resolution (def: (1, 1, 1))

    Output:
    - (dict) containing each of the evaluated results

    """

    metrics = {}
    # metrics['vol_mask'] = num_voxels(mask)
    # metrics['vol_gt'] = num_voxels(gt)
    # metrics['regions_mask'] = num_regions(mask)
    # metrics['regions_gt'] = num_regions(gt)
    # metrics['min_region_mask'] = min_region(mask)
    # metrics['max_region_mask'] = max_region(mask)
    # metrics['min_region_gt'] = min_region(gt)
    # metrics['max_region_gt'] = max_region(gt)
    # metrics['tpf_seg'] = TPF_seg(gt, mask)
    # metrics['tpf_det'] = TPF_det(gt, mask)
    # metrics['fpf_seg'] = FPF_seg(gt, mask)
    # metrics['fpf_det'] = FPF_det(gt, mask)
    metrics['dsc_seg'] = DSC_seg(gt, mask)
    # metrics['dsc_det'] = DSC_det(gt, mask)
    # metrics['ppv_seg'] = PPV_seg(gt, mask)
    # metrics['ppv_det'] = PPV_det(gt, mask)
    # metrics['vd'] = PVE(gt, mask)
    # metrics['vsi'] = VSI(gt, mask)
    metrics['hd'] = HD(gt, mask, spacing)
    metrics['mhd'] = MHD(gt, mask, spacing)
    # metrics['f_score'] = f_score(gt, mask)
    # metrics['mad'] = mad(gt, mask)
    # metrics['ssim'] = ssim(gt, mask)

    return metrics

