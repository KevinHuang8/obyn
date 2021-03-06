import numpy as np
from ..utils import constants as C

def compute_iou(label1, label2, group_id1, group_id2):
    '''
    Computes the IOU between two point groups: those with 'label1' equal to
    'group_id1' and those with 'label2' equal to 'group_id2'.
    '''

    pts1 = label1 == group_id1
    pts2 = label2 == group_id2

    return np.sum(pts1 & pts2) / np.sum(pts1 | pts2)

def get_counts(gt_labels, pred_labels, iou_thresh):
    '''
    Gets the TP, FP, and FN counts for a single sample with ground truth
    'gt_labels' and predictions 'pred_labels'
    '''
    TP, FP, FN = 0, 0, 0

    associated = set()
    for i in np.unique(gt_labels):
        closest_iou = -1
        closest = -1
        for j in np.unique(pred_labels):
            if j in associated:
                continue

            iou = compute_iou(gt_labels, pred_labels, i, j)

            if iou < iou_thresh:
                continue

            if iou > closest_iou:
                closest = j
                closest_iou = iou
        
        if closest == -1:
            FN += 1
        else:
            associated.add(closest)
            TP += 1

    for j in np.unique(pred_labels):
        if j not in associated:
            FP += 1

    return TP, FP, FN

def get_counts_all(gt_all, predictions_all, iou_thresh):
    '''
    Gets the TP, FP, and FN counts for all samples.
    '''
    TP, FP, FN = 0, 0, 0
    for i, pred in enumerate(predictions_all):
        dTP, dFP, dFN = get_counts(gt_all[i], pred, iou_thresh)
        TP += dTP
        FP += dFP
        FN += dFN

    return TP, FP, FN

def precision(TP, FP):
    return TP / (TP + FP)

def recall(TP, FN):
    return TP / (TP + FN)

def f1(TP, FP, FN):
    return TP / (TP + 0.5*(FP + FN))

def f1_PR(precision, recall):
    return 2*(precision*recall)/(precision + recall)
