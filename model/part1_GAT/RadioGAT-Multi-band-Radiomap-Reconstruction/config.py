"""
Configuration Module
======================================
Centralized management of all training and data processing configurations.
All paths obtained via get_config(); hardcoded paths in other modules are forbidden.
"""

import os
from typing import List


class Config:
    """GAT Model Training Configuration"""

    PROJECT_ROOT = r"D:\3_document\4_research\NLOS Signal Identification and Correction"

    DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "dataset")

    PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processedData")

    RESULT_DIR = os.path.join(PROJECT_ROOT, "model", "part1_GAT", "result")

    SOURCE_DIR = os.path.join(PROJECT_ROOT, "model", "part1_GAT",
                              "RadioGAT-Multi-band-Radiomap-Reconstruction")

    DATASETS = [
        'berlin1_potsdamer_platz',
    ]

    # ========== Data Processing Config ==========
    NEED_PROCESS_DATA = True
    PROCESSED_DATA_FORMAT = '{dataset_name}_processed.pkl'
    TIME_SYNC_TOLERANCE = 1.0
    USE_ESTIMATED_GEOMETRY = True

    ERROR_CLIP_KM = 100.0

    # ========== Model Architecture Config ==========
    IN_FEATURES = 11
    HIDDEN_FEATURES = 128
    NUM_HEADS = 8
    NUM_LAYERS = 2
    DROPOUT = 0.1
    SIGMA_MIN = 5.0
    SIGMA_MAX = 8.0

    # ========== Training Config ==========
    LEARNING_RATE = 5e-5
    NUM_EPOCHS = 100
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.2
    GRADIENT_CLIP = 1.0

    GRADIENT_ACCUMULATION = 1

    USE_LR_SCHEDULER = True
    SCHEDULER_PATIENCE = 10
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_MIN_LR = 1e-6

    EARLY_STOPPING_PATIENCE = 20  # default; overridden to 60 for MoG training

    CHECKPOINT_INTERVAL = 1

    # ========== Loss Function Config ==========
    POS_WEIGHT = 1.07
    LABEL_SMOOTHING = 0.05
    LAMBDA_BCE = 0.6
    P_LOS_SMOOTHING = 0.2
    LAMBDA_ENTROPY = 0.03
    LAMBDA_UNC = 0.08
    LAMBDA_ELEVATION_PRIOR = 0.1
    USE_MIXTURE_GAUSSIAN = True

    # ========== MoG (Mixture of Gaussians) Config ==========
    MOG_PURE_BCE_EPOCHS = 10  # shortened: 30→10
    MOG_BLEND_EPOCHS = 40  # extended: 25→40
    # SIGMA_LOS_FIXED = 2.0  # deprecated: sigma_los now learnable
    MU_NLOS_MIN = 0.0
    MU_NLOS_MAX = 500.0
    SIGMA_NLOS_MIN = 0.05  # lowered to match new clamp
    SIGMA_NLOS_MAX = 200.0
    LAMBDA_MU_REG = 0.005  # 5x stronger: 0.001→0.005
    LAMBDA_SIGMA_REG = 0.01  # 10x stronger: 0.001→0.01
    SIGMA_GAP_TARGET = 0.5  # increased: 0.3→0.5 km
    LAMBDA_SIGMA_SEP = 5.0  # stronger: 2.0→5.0

    # ========== Graph Construction Config ==========
    AZIMUTH_THRESHOLD = 90

    # ========== Device Config ==========
    DEVICE = 'auto'

    # ========== Logging Config ==========
    LOG_INTERVAL = 5
    USE_TENSORBOARD = False  # sandbox blocks TB write
    TENSORBOARD_DIR = None
    LOG_GRADIENTS = True
    LOG_HISTOGRAM_EPOCHS = 5

    # ========== Prediction Output Config ==========
    SAVE_PREDICTIONS = True
    PREDICTION_FORMAT = 'prediction_{gps_week}_{gps_seconds}.json'

    # ========== Other Config ==========
    RANDOM_SEED = 42

    # ========== Block-Diagonal Batching ==========
    USE_BLOCK_DIAGONAL = True
    NUM_WORKERS = 8
    VAL_NUM_WORKERS = 4
    USE_AMP = True
    

    def __init__(self, **kwargs):
        """Initialize config, supports overriding defaults"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_device(self):
        """Get training device"""
        if self.DEVICE == 'auto':
            import torch
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            import torch
            return torch.device(self.DEVICE)

    def ensure_dirs(self):
        """Ensure all necessary directories exist"""
        dirs_to_create = [
            self.PROCESSED_DATA_DIR,
            self.RESULT_DIR
        ]
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)

    def get_processed_data_path(self, dataset_name: str) -> str:
        """Get processed data save path"""
        return os.path.join(
            self.PROCESSED_DATA_DIR,
            self.PROCESSED_DATA_FORMAT.format(dataset_name=dataset_name)
        )

    def get_data_dir(self, dataset_name: str) -> str:
        """Get raw dataset directory"""
        return os.path.join(self.DATA_ROOT, dataset_name)

    def __str__(self):
        """Print configuration info"""
        info = []
        info.append("=" * 60)
        info.append("GAT Model Configuration")
        info.append("=" * 60)
        info.append(f"  DATA_ROOT: {self.DATA_ROOT}")
        info.append(f"  PROCESSED_DATA_DIR: {self.PROCESSED_DATA_DIR}")
        info.append(f"  RESULT_DIR: {self.RESULT_DIR}")
        info.append(f"  DATASETS: {', '.join(self.DATASETS)}")
        info.append(f"  IN_FEATURES: {self.IN_FEATURES}")
        info.append(f"  HIDDEN_FEATURES: {self.HIDDEN_FEATURES}")
        info.append(f"  NUM_HEADS: {self.NUM_HEADS}")
        info.append(f"  NUM_LAYERS: {self.NUM_LAYERS}")
        info.append(f"  LEARNING_RATE: {self.LEARNING_RATE}")
        info.append(f"  NUM_EPOCHS: {self.NUM_EPOCHS}")
        info.append(f"  BATCH_SIZE: {self.BATCH_SIZE}")
        info.append(f"  GRADIENT_ACCUMULATION: {self.GRADIENT_ACCUMULATION}")
        info.append(f"  AZIMUTH_THRESHOLD: {self.AZIMUTH_THRESHOLD} deg")
        info.append(f"  USE_MIXTURE_GAUSSIAN: {self.USE_MIXTURE_GAUSSIAN}")
        info.append(f"  LAMBDA_BCE: {self.LAMBDA_BCE}")
        info.append(f"  LAMBDA_ENTROPY: {self.LAMBDA_ENTROPY}")
        info.append(f"  LAMBDA_ELEVATION_PRIOR: {self.LAMBDA_ELEVATION_PRIOR}")
        info.append("=" * 60)
        return '\n'.join(info)


def get_config(**overrides) -> Config:
    """Factory function, supports command-line config overrides"""
    config = Config()
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


