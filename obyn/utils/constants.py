from pathlib import Path

MAIN_DIR = Path(__file__).parent.parent.parent
DATA_DIR = MAIN_DIR / 'data'
TRAINING_DIR = MAIN_DIR / 'obyn' / 'training'
EVALUATION_DIR = MAIN_DIR / 'obyn' / 'evaluation'
NEON_DIR_RAW = DATA_DIR / 'NeonTrees'
IDTREES_DIR_RAW = DATA_DIR / 'train'

THS_SAVE_FILE = EVALUATION_DIR / 'ths.npy'

## Data parameters ##


NUM_POINTS = 1024
NONZERO_POINT_THRESHOLD = 50

# when using artificial labels, take only every x samples to save space/time
ARTIFICIAL_LABEL_SKIP = 10 


## Model parameters ##


# Maximum number of instances (trees) in a point cloud
NUM_GROUPS = 100


## Training parameters ##


BATCH_SIZE = 32
TRAINING_EPOCHES = 20

# Perecent of training data to use for validation
VALIDATION_SIZE = 0.15

CHECKPOINT_DIR = TRAINING_DIR / 'checkpoints'

MIN_ALPHA = 100


## Test parameters ##

DEFAULT_CONFIDENCE_THRESHOLD = 0.1
MIN_POINTS_IN_GROUP_PROPOSAL = 10
DEFAULT_IOU_THRESH = 0.5

# Number of different confidence levels to test when calculating PR curve/AP
NUM_ITER_PR_CURVE = 20
