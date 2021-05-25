import numpy as np
import matplotlib.pyplot as plt
import obyn.training.train_tree as t
from obyn.evaluation.evaluate import evaluate
from obyn.utils import read_data
from obyn.utils import constants as C

if __name__ == '__main__':
    # Note: make sure to force reload whenever changing data size
    # t.train(data_category='data_neon', force_reload=False, artificial_labels=False)

    data = read_data.LidarData()

    ap, prec, recall = evaluate(data.x, data.y, str(C.CHECKPOINT_DIR / 'epoch_20.ckpt'))

    print(ap)
    plt.plot(prec, recall)
    plt.savefig('PR.png')
