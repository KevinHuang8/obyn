from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / 'data'
NEON_DIR_RAW = DATA_DIR / 'NeonTrees'
IDTREES_DIR_RAW = DATA_DIR / 'train'

NUM_POINTS = 1024
NONZERO_POINT_THRESHOLD = 50

# Model parameters
NUM_GROUPS = 230 # Maximum number of instances (trees) in a point cloud

# Training parameters
BATCH_SIZE = 16
TRAINING_EPOCHES = 20