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
# BAG_SECONDS can be an integer (e.g., 3, 10, 30) to cut fixed-length bags,
# or "full" to use the entire recording as a single bag.
BAG_SECONDS = 10

# FULL_BAG_METHOD is used only when BAG_SECONDS == "full".
# Options: "batch" (batch_size=1, no padding) or "pad" (pad to max length).
FULL_BAG_METHOD = "batch"

# PAD_MODE is used only when BAG_SECONDS == "full" and FULL_BAG_METHOD == "pad".
# Options: "repeat" (repeat audio to max length) or "silence" (zero-pad + mask).
PAD_MODE = "silence"

# MODEL_NAME options: Baseline, CNN-biGRU, CNN-Transformer, CDur, TALNet
MODEL_NAME = "CDur"

EPOCHS = 100
BATCH_SIZE = 8
NUM_WORKERS = 4
LEARNING_RATE = 1e-3

VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1
APPLY_VALIDATION_SPLIT = True
APPLY_TEST_SPLIT = True

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

OVERLAP_BAGS = False
HOP_SECONDS = 1
