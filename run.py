import os
import obyn.training.train_tree as t

# Disable INFO and WARNING messages (0 is default)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

if __name__ == '__main__':
    # Note: make sure to force reload whenever changing data size
    t.train(data_category='data_neon', force_reload=False, artificial_labels=False)
