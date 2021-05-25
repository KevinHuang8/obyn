import numpy as np
import matplotlib.pyplot as plt
import obyn.training.train_tree as t
from obyn.evaluation.evaluate import evaluate
from obyn.evaluation.calculate_ths import calculate_ths
from obyn.utils import read_data
from obyn.utils import constants as C

if __name__ == '__main__':
    data_name = 'standard4096'
    model_name = 'model2'

    train = True
    # Training
    if train:
        data = read_data.LidarData(data_name, category='data_neon', force_reload=True)
        t.train(data, model_name)
        data.save_indices(model_name)

    # After training, we can refer to the model by 'model_name'
    # Note everything after here can be done separately from training.
    data = read_data.LidarData(data_name)
    train_idx, valid_idx = data.load_indices(model_name)

    # Need to update Ths on the training data if not updated already
    if not (C.THS_DIR / f'{model_name}.npy').is_file():
        calculate_ths(data.x[train_idx], data.y[train_idx], model_name)

    # iou = 0.25, 0.5, 0.75
    ap, prec, recall = evaluate(data.x[valid_idx], data.y[valid_idx], model_name)

    print(f'AP: {ap}')
    plt.figure()
    plt.plot(recall, prec)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'PR Curve (AP: {ap:.4f})')
    plt.savefig(C.FIGURES_DIR / f'PR_curve_{model_name}.png')
