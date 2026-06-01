import torch

TRAIN_FILE = 'mitbih_train.csv'
TEST_FILE = 'mitbih_test.csv'

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
N_CLASSES = 5
INPUT_CHANNELS = 1
INPUT_LENGTH = 187 # 187 features per sample
BATCH_SIZE = 64
NUM_EPOCHS = 50 # You had 50 in your notebook
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
LR_PATIENCE = 5 # For ReduceLROnPlateau scheduler
LR_FACTOR = 0.5 # For ReduceLROnPlateau scheduler

# --- File Paths ---
MODEL_PATH = 'best_ecg_model.pth'
SCALER_PATH = 'scaler.joblib' # Path to save the scaler (My Improvement #1)
TRAIN_PLOTS_PATH = 'training_plots.png'
EVAL_CM_PATH = 'confusion_matrix.png'
EVAL_ROC_PATH = 'roc_curves.png'
