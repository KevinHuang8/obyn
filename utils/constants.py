from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'data'
NEON_DIR_RAW = DATA_DIR / 'NeonTrees'
IDTREES_DIR_RAW = DATA_DIR / 'train'

NUM_POINTS = 1024
NONZERO_POINT_THRESHOLD = 50