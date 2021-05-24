from pathlib import Path

MAIN_DIR = Path(__file__).parent.parent.parent
DATA_DIR = MAIN_DIR / 'data'
TRAINING_DIR = MAIN_DIR / 'obyn' / 'training'
NEON_DIR_RAW = DATA_DIR / 'NeonTrees'
IDTREES_DIR_RAW = DATA_DIR / 'train'

## Data parameters ##


NUM_POINTS = 1024
NONZERO_POINT_THRESHOLD = 50

# when using artificial labels, take only every x samples to save space/time
ARTIFICIAL_LABEL_SKIP = 10 


## Model parameters ##


# Maximum number of instances (trees) in a point cloud
NUM_GROUPS = 230 


## Training parameters ##


BATCH_SIZE = 32
TRAINING_EPOCHES = 20

# Perecent of training data to use for validation
VALIDATION_SIZE = 0.15

CHECKPOINT_DIR = TRAINING_DIR / 'checkpoints'