from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / 'data'
NEON_DIR_RAW = DATA_DIR / 'NeonTrees'
IDTREES_DIR_RAW = DATA_DIR / 'train'

# Data parameters

NUM_POINTS = 1024
NONZERO_POINT_THRESHOLD = 50

# when using artificial labels, take only every x samples to save space/time
ARTIFICIAL_LABEL_SKIP = 10 

# Model parameters
NUM_GROUPS = 230 # Maximum number of instances (trees) in a point cloud

# Training parameters
BATCH_SIZE = 32
TRAINING_EPOCHES = 20