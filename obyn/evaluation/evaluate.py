import numpy as np
from tqdm import tqdm
from .predict import predict, model_output
from . import metrics
from .calculate_ths import calculate_ths
from ..utils import constants as C

def evaluate(X, y, name, iou_thresh=C.DEFAULT_IOU_THRESH,
    num=C.NUM_ITER_PR_CURVE):
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

    precision_list = []
    recall_list = []
    conf_threshs = np.linspace(min(confidences), max(confidences), num)
    print("Evaluating PR for {} different confidence levels".format(len(conf_threshs)))
    for conf_thresh in conf_threshs:
        print(f'Evaluating confidence threshold {conf_thresh}')
        predictions = predict(name, model_outputs=model_outputs,
            confidence_threshold=conf_thresh, use_outputs=True)

        TP, FP, FN = 0, 0, 0
        for i, pred in enumerate(predictions):
            dTP, dFP, dFN = metrics.get_counts(y[i], pred, iou_thresh)
            TP += dTP
            FP += dFP
            FN += dFN

        prec = metrics.precision(TP, FP)
        precision_list.append(prec)
        recall = metrics.recall(TP, FN)
        recall_list.append(recall)

    # Calculate AP (code from: https://blog.paperspace.com/mean-average-precision/)
    recalls = np.array(recall_list + [0.0])
    precisions = np.array(precision_list + [1.0])
    ap = np.sum((recalls[:-1] - recalls[1:]) * precisions[:-1])

    return ap, precision_list, recall_list
