"""
Central configuration for the Indian Pines hyperspectral classification project.

All paths are configurable via environment variables so the code runs the
same way locally, on a server, or in a notebook -- no more hardcoded
'/content/...' Colab paths.

Override any of these by exporting the corresponding environment variable, e.g.:
    export IP_DATA_DIR=/path/to/dataset
    export IP_SAVE_DIR=/path/to/outputs
"""
import os

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------
DATA_DIR = os.environ.get("IP_DATA_DIR", "./data")
DATASET_DATA_PATH = os.environ.get(
    "IP_DATASET_DATA_PATH", os.path.join(DATA_DIR, "Indian_pines_corrected.mat")
)
DATASET_LABELS_PATH = os.environ.get(
    "IP_DATASET_LABELS_PATH", os.path.join(DATA_DIR, "Indian_pines_gt.mat")
)

# ---------------------------------------------------------------------------
# Output paths (models, results, plots, splits)
# ---------------------------------------------------------------------------
SAVE_DIR = os.environ.get("IP_SAVE_DIR", "./results")

MODELS_DIR = os.path.join(SAVE_DIR, "models")
RESULTS_DIR = os.path.join(SAVE_DIR, "metrics")
PLOTS_DIR = os.path.join(SAVE_DIR, "plots")
SPLITS_DIR = os.path.join(SAVE_DIR, "splits")

for _d in (MODELS_DIR, RESULTS_DIR, PLOTS_DIR, SPLITS_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = int(os.environ.get("IP_SEED", 42))

# ---------------------------------------------------------------------------
# Class configuration (7 agricultural classes selected from the 16 Indian
# Pines land-cover classes)
# ---------------------------------------------------------------------------
SELECTED_CLASSES = [2, 3, 4, 10, 11, 12, 13]
CLASS_NAMES = [
    "Corn-notill",
    "Corn-mintill",
    "Corn",
    "Soybean-notill",
    "Soybean-mintill",
    "Soybean-clean",
    "Wheat",
]
