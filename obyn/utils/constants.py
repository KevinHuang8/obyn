from pathlib import Path

#################
## Directories ##
#################

MAIN_DIR = Path(__file__).parent.parent.parent
DATA_DIR = MAIN_DIR / 'data'
# Where generated data is cached
DATA_GENERATED_DIR = DATA_DIR / 'generated'
# Raw data location
NEON_DIR_RAW = DATA_DIR / 'NeonTrees'
IDTREES_DIR_RAW = DATA_DIR / 'train'

TRAINING_DIR = MAIN_DIR / 'obyn' / 'training'
EVALUATION_DIR = MAIN_DIR / 'obyn' / 'evaluation'
# Where model checkpoints are stored
CHECKPOINT_DIR = TRAINING_DIR / 'checkpoints'
THS_DIR = EVALUATION_DIR / 'ths'
IDX_DIR = TRAINING_DIR / 'saved_idx'
FIGURES_DIR = MAIN_DIR / 'figures'

#####################
## Data parameters ##
#####################

# Number of points per point cloud
NUM_POINTS = 1024
# Point clouds with less than this many nonzero points are discarded
NONZERO_POINT_THRESHOLD = 50

# When using artificial labels, take only every x samples to save space/time
ARTIFICIAL_LABEL_SKIP = 10 
# How many extra samples to add to the augmented dataset by rotations per original
# sample
NUM_EXTRA_AUGMENTED = 4

# Groups with less than this number of points are removed
GROUP_THRESHOLD = 0

######################
## Model parameters ##
######################

# Maximum number of instances (trees) in a point cloud
NUM_GROUPS = 88
# Batch normalization exponential moving average decay
# Lower if overfitting
BN_DECAY = 0.9

#########################
## Training parameters ##
#########################

BATCH_SIZE = 32
TRAINING_EPOCHES = 30

# Perecent of training data to use for validation
VALIDATION_SIZE = 0.15

# alpha value for the loss
# a higher alpha penalizes points that are in the same segmentation class
# but are incorrectly grouped more
MIN_ALPHA = 5

# Adam optimizer parameters
DECAY_STEP = 800000.
DECAY_RATE = 0.5

LEARNING_RATE_CLIP = 1e-6
BASE_LEARNING_RATE = 1e-4
MOMENTUM = 0.9

#####################
## Test parameters ##
#####################

DEFAULT_CONFIDENCE_THRESHOLD = 0.1
# groups with less than this number of points are discarded
# Basically, we think that any trees with less than this number of points
# aren't really trees. Th_M2 in the paper
MIN_POINTS_IN_GROUP_PROPOSAL = 10
DEFAULT_IOU_THRESH = 0.5

# Number of different confidence levels to test when calculating PR curve/AP
NUM_ITER_PR_CURVE = 20
