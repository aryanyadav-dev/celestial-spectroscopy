# Directory where data is saved
SAVE_DIR = './spectral_data'

# Model parameters
INPUT_SHAPE = (1024, 1)  # Input shape for the model (1024 spectral points, 1 feature per point)
NUM_CLASSES = 2  # Two classes: 'star' and 'galaxy'

# Number of samples for data collection
NUM_SAMPLES = 500

# Model training parameters
EPOCHS = 20
BATCH_SIZE = 32

# Checkpoints and model saving
MODEL_SAVE_PATH = 'spectral_model_best.keras'

# Logging parameters
LOG_DIR = './logs'

# SDSS query and data fetching settings
SDSS_QUERY_LIMIT = 1000
