import numpy as np
import matplotlib.pyplot as plt
import obyn.training.train_tree as t
from obyn.evaluation.evaluate import evaluate
from obyn.evaluation.metrics import *
from obyn.evaluation.calculate_ths import calculate_ths
from obyn.utils.data_augmentation.artificial_labelling import create_artificial_labels
from obyn.utils import read_data
from obyn.utils import constants as C

if __name__ == '__main__':
    data_name = 'augmented'
    model_name = 'model3'

    train = False
    # Training
    if train:
        data = read_data.LidarDataAugmented(data_name, category='data_neon', force_reload=True,
            group_threshold=0)
        t.train(data, model_name)
        data.save_indices(model_name)

    # After training, we can refer to the model by 'model_name'
    # Note everything after here can be done separately from training.
    data = read_data.LidarDataAugmented(data_name)
    train_idx, valid_idx = data.load_indices(model_name)

    # Need to update Ths on the training data if not updated already
    if not (C.THS_DIR / f'{model_name}.npy').is_file():
        calculate_ths(data.x[train_idx], data.y[train_idx], model_name)

    # iou = 0.25, 0.5, 0.75
    ap_dict, precision_dict, recall_dict = evaluate(data.x[valid_idx], data.y[valid_idx], model_name, iou_thresh=[0.25, 0.5, 0.75])

    print(f'AP25: {ap_dict[0.25]}')
    print(f'AP50: {ap_dict[0.50]}')
    print(f'AP75: {ap_dict[0.75]}')
    plt.figure()
    plt.plot(recall_dict[0.25], precision_dict[0.25], label='IOU: 25%')
    plt.plot(recall_dict[0.5], precision_dict[0.5], label='IOU: 50%')
    plt.plot(recall_dict[0.75], precision_dict[0.75], label='IOU: 75%')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'PR Curve')
    plt.legend()

    plt.savefig(C.FIGURES_DIR / f'PR_curve_{model_name}.png')

    # Get precision/recall for k-means clustering
    # data = read_data.LidarData(data_name)
    # y_kmeans = create_artificial_labels(data.x, data.y)
    # k_iou = 0.5
    # TP, FP, FN = get_counts_all(data.y, y_kmeans, k_iou)
    # k_prec = precision(TP, FP)
    # k_recall = recall(TP, FN)
    # print(f'KMeans precision (50%): {k_prec}, recall: {k_recall}')