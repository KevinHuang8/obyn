import numpy as np
from tqdm import tqdm
from .predict import predict, model_output
from . import metrics
from ..utils import constants as C

def evaluate(X, y, model_path, iou_thresh=C.DEFAULT_IOU_THRESH, 
    num=C.NUM_ITER_PR_CURVE):
    '''
    Return average precision (AP) scores on validation data (X, y) using
    model checkpoint saved at 'model_path'.

    'num' is number of different confidence levels to test
    '''

    model_outputs = model_output(X, model_path)

    confidences = []
    for i in model_outputs:
        confidences.extend([j for j in np.unique(i['conf'])])

    confidences = np.array(confidences)

    precision_list = []
    recall_list = []
    total_prec = 0
    for conf_thresh in np.linspace(min(confidences), max(confidences), num):
        print(f'Evaluating confidence threshold {conf_thresh}')
        predictions = predict(model_outputs=model_outputs, 
            confidence_threshold=conf_thresh, use_outputs=True)

        TP, FP, FN = 0, 0, 0
        for i, pred in enumerate(predictions):
            dTP, dFP, dFN = metrics.get_counts(y[i], pred, iou_thresh)
            TP += dTP
            FP += dFP
            FN += dFN

        prec = metrics.precision(TP, FP)
        precision_list.append(prec)
        total_prec += prec
        recall = metrics.recall(TP, FN)
        recall_list.append(recall)

    ap = total_prec / num

    return ap, precision_list, recall_list
