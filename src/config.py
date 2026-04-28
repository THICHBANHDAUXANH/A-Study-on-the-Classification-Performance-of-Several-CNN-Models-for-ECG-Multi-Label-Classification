from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PTBXL_DIR = PROJECT_ROOT/"data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"
PTBXL_META_CSV = PTBXL_DIR/"ptbxl_database.csv"
PTBXL_SCP_CSV = PTBXL_DIR/"scp_statements.csv"
PTBXL_RECORD_BASE = PTBXL_DIR

OUTPUT_DIR = PROJECT_ROOT/"outputs"
PREPROCESSED_SIGNAL_DIR = OUTPUT_DIR/"preprocessed_signals"
IMAGE_DIR = OUTPUT_DIR/"ecg_images"
LABEL_DIR = OUTPUT_DIR/"labels"
ARRAY_DIR = OUTPUT_DIR/"arrays"
MODEL_DIR = OUTPUT_DIR/"models"
FIGURE_DIR = OUTPUT_DIR/"figures"
TABLE_DIR = OUTPUT_DIR/"tables"
GRADCAM_DIR = OUTPUT_DIR/"gradcam"

ARRAY_X_PATH = ARRAY_DIR/"ecg_images_array.npy"
ARRAY_Y_PATH = ARRAY_DIR/"ecg_labels_array.npy"
ARRAY_IDS_PATH = ARRAY_DIR/"ecg_array_ids.npy"

REPORT_CLASSES = ["NORM", "MI", "CD", "HYP", "STTC"]

IMAGE_SIZE = 224
RANDOM_STATE = 48
TEST_SIZE = 0.20
