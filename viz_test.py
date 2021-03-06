import obyn.training.viz_predict_tree as v
from obyn.utils import read_data
from obyn.utils import constants as C
import numpy as np
from obyn.utils.visualization import show3d_balls as viz

if __name__ == '__main__':
    # Note: make sure to force reload whenever changing data size

    # Note: RUN compile_render_balls_so.sh in obyn/utils/visualiztion
    # to compile the render_balls_so.so file

    data = read_data.LidarData('standard', category='data_neon', force_reload=False)

    X = data.x[122:130]
    Y = data.y[122:130]

    assert len(X) == len(Y)

    print("Predicting on {} points".format(len(X)))
    print("Each one will open its own visualiztion window")
    print("Press P to see predicted colors, Press T to see true colors, Press Q to move on to next")


    v.predict(X, Y, model_path=str(C.CHECKPOINT_DIR / 'model6.ckpt'))
