import SimpleITK as sitk
import os 
from typing import Tuple, List, Union, Optional
import numpy as np
def region_or_label_to_mask(segmentation: np.ndarray, region_or_label: Union[int, Tuple[int, ...]]) -> np.ndarray:
    if np.isscalar(region_or_label):
        return segmentation == region_or_label
    else:
        mask = np.zeros_like(segmentation, dtype=bool)
        for r in region_or_label:
            mask[segmentation == r] = True
    return mask

def compute_tp_fp_fn_tn(mask_ref: np.ndarray, mask_pred: np.ndarray, ignore_mask: np.ndarray = None):
    if ignore_mask is None:
        use_mask = np.ones_like(mask_ref, dtype=bool)
    else:
        use_mask = ~ignore_mask
    tp = np.sum((mask_ref & mask_pred) & use_mask)
    fp = np.sum(((~mask_ref) & mask_pred) & use_mask)
    fn = np.sum((mask_ref & (~mask_pred)) & use_mask)
    tn = np.sum(((~mask_ref) & (~mask_pred)) & use_mask)
    return tp, fp, fn, tn
    
def compute_metrics(reference_file: str, prediction_file: str, 
    labels_or_regions: Union[List[int], List[Union[int, Tuple[int, ...]]]],ignore_label: int = None) -> dict:
    # load images
    seg_ref = sitk.ReadImage(reference_file)
    seg_ref =sitk.GetArrayFromImage(seg_ref)
    seg_pred = sitk.ReadImage(prediction_file)
    seg_pred = sitk.GetArrayFromImage(seg_pred)

    ignore_mask = seg_ref == ignore_label if ignore_label is not None else None

    results = {}
    results['reference_file'] = reference_file
    results['prediction_file'] = prediction_file
    results['metrics'] = {}
    for r in labels_or_regions:
        results['metrics'][r] = {}
        mask_ref = region_or_label_to_mask(seg_ref, r)
        mask_pred = region_or_label_to_mask(seg_pred, r)
        tp, fp, fn, tn = compute_tp_fp_fn_tn(mask_ref, mask_pred, ignore_mask)
        if tp + fp + fn == 0:
            results['metrics'][r]['Dice'] = np.nan
            results['metrics'][r]['IoU'] = np.nan
        else:
            results['metrics'][r]['Dice'] = 2 * tp / (2 * tp + fp + fn)
            results['metrics'][r]['IoU'] = tp / (tp + fp + fn)
        results['metrics'][r]['FP'] = fp
        results['metrics'][r]['TP'] = tp
        results['metrics'][r]['FN'] = fn
        results['metrics'][r]['TN'] = tn
        results['metrics'][r]['n_pred'] = fp + tp
        results['metrics'][r]['n_ref'] = fn + tp
    return results

file_path = '/cpfs01/projects-SSD/cfff-282dafecea22_SSD/pantan/code/research-contributions/SwinUNETR/AUTOPET/outputs/cropped_autopet_nii_ct'
from tqdm import tqdm
files = os.listdir(file_path)
results = []
for fi in tqdm(files):
    if 'seg' not in fi:
        continue
    fi_pre = os.path.join(file_path,fi)
    fi_ref = os.path.join(file_path,fi.replace('_seg', '_label'))
    r = compute_metrics(fi_ref,fi_pre,labels_or_regions=[1])
    results.append(r['metrics'][1]['Dice'])
import pdb;pdb.set_trace()