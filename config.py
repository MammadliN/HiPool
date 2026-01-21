# Configuration for HiPool WSSED pipeline

# Random seed
SEED = 42

# Dataset roots
ANURASET_ROOT = "/ds-iml/Bioacoustics/AnuraSet/raw_data"
FNJV_ROOT = "/ds-iml/Bioacoustics/FNJV/458"
# /ds-iml/Bioacoustics/FNJV/578

# Dataset selection
DATASET_TRAIN = "FNJV"
DATASET_VAL = "AnuraSet"
DATASET_TEST = "AnuraSet"

# POOLING options: max, mean, linear, exp, att, auto, power, hi, hi_plus, hi_fixed
POOLING = "mean"
BAG_SECONDS = 10

# MODEL_NAME options: Baseline, CNN-biGRU, CNN-Transformer, CDur, TALNet
MODEL_NAME = "CDur"

EPOCHS = 100
BATCH_SIZE = 8
NUM_WORKERS = 4
LEARNING_RATE = 1e-3

VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

TARGET_SPECIES = [
    "DENMIN",
    "LEPLAT",
    "PHYCUV",
    "SPHSUR",
    "SCIPER",
    "BOABIS",
    "BOAFAB",
    "LEPPOD",
    "PHYALB",
]
# TARGET_SPECIES = ["DENMIN", "BOARAN", "DENNAN", "LEPFUS", "SCIFUS"]

sample_rate = 16000
n_mels = 64
n_fft = 1024
hop_length = 664

threshold = 0.5
