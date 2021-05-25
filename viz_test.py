import obyn.training.viz_predict_tree as v
from obyn.utils import read_data
from obyn.utils import constants as C

if __name__ == '__main__':
    # Note: make sure to force reload whenever changing data size

    # Note: RUN compile_render_balls_so.sh in obyn/utils/visualiztion
    # to compile the render_balls_so.so file

    data = read_data.LidarData(category='data_neon', force_reload=False)
    X = data.x[12:15]
    Y = data.y[12:15]

    assert len(X) == len(Y)

    print("Predicting on {} points".format(len(X)))
    print("Each one will open its own visualiztion window")
    print("Press P to see predicted colors, Press T to see true colors, Press Q to move on to next")

    v.predict(X, Y, model_path=str(C.CHECKPOINT_DIR / 'epoch_20.ckpt'))
