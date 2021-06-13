import numpy as np
from tqdm import tqdm
from .predict import predict, model_output
from . import metrics
from .calculate_ths import calculate_ths
from ..utils import constants as C

def evaluate(X, y, name, iou_thresh=[C.DEFAULT_IOU_THRESH],
    num=C.NUM_ITER_PR_CURVE, return_conf_list=False):
    '''
    Return average precision (AP) scores on validation data (X, y) using
    model 'name'.

    'num' is number of different confidence levels to test
    '''

    model_path = str(C.CHECKPOINT_DIR / (name + '.ckpt'))
    model_outputs = model_output(X, model_path)

    confidences = []
    for i in model_outputs:
        confidences.extend([j for j in np.unique(i['conf'])])

    confidences = np.array(confidences)

    precision_lists = {}
    recall_lists = {}
    ap_dict = {}
    conf_threshs = np.linspace(min(confidences), max(confidences), num)
    
    for iou_t in iou_thresh:
        precision_lists[iou_t] = []
        recall_lists[iou_t] = []

    print("Evaluating PR for {} different confidence levels".format(len(conf_threshs)))
    for conf_thresh in conf_threshs:
        print(f'Evaluating confidence threshold {conf_thresh}')
        
        predictions = predict(name, model_outputs=model_outputs,
            confidence_threshold=conf_thresh, use_outputs=True)

        for iou_t in iou_thresh:
            TP, FP, FN = metrics.get_counts_all(y, predictions, iou_t)

            prec = metrics.precision(TP, FP)
            precision_lists[iou_t].append(prec)
            recall = metrics.recall(TP, FN)
            recall_lists[iou_t].append(recall)

    for iou_t in iou_thresh:
        # Calculate AP (code from: https://blog.paperspace.com/mean-average-precision/)
        recalls = np.array(recall_lists[iou_t] + [0.0])
        precisions = np.array(precision_lists[iou_t] + [1.0])
        ap = np.sum((recalls[:-1] - recalls[1:]) * precisions[:-1])
        ap_dict[iou_t] = ap

    if return_conf_list:
        return ap_dict, precision_lists, recall_lists, conf_threshs
    return ap_dict, precision_lists, recall_lists
