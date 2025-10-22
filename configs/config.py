import os
from pathlib import Path
# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DATASET_PATH = PROCESSED_DATA_DIR / "news(U).xlsx"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
# Model paths
MODELS_DIR = PROJECT_ROOT / "models"
FINE_TUNED_DIR = MODELS_DIR / "fine_tuned"
BERT_MODELS_DIR = MODELS_DIR / "bert_models"
ROBERTA_MODELS_DIR = MODELS_DIR / "roberta_models"
ONNX_DIR = MODELS_DIR / "onnx"
# Fine-tuning hyperparameters (optimized for free resources)
FINE_TUNING_CONFIG = {
    'bert': {
        'model_name': 'bert-base-uncased',
        'num_labels': 2,
        'max_length': 512,
        'batch_size': 8,
        'learning_rate': 2e-5,
        'num_epochs': 3,
        'warmup_ratio': 0.1,
        'weight_decay': 0.01,
        'seed': 42
    },
    'roberta': {
        'model_name': 'roberta-base',
        'num_labels': 2,
        'max_length': 512,
        'batch_size': 8,
        'learning_rate': 1e-5,
        'num_epochs': 3,
        'warmup_ratio': 0.1,
        'weight_decay': 0.01,
        'seed': 42
    },
    'distilbert': {
        'model_name': 'distilbert-base-uncased',
        'num_labels': 2,
        'max_length': 512,
        'batch_size': 16,
        'learning_rate': 5e-5,
        'num_epochs': 4,
        'warmup_ratio': 0.1,
        'weight_decay': 0.01,
        'seed': 42
    }
}
# Environment mapping for different tasks
ENVIRONMENT_MAP = {
    'data_processing': 'env_nlp',
    'fine_tuning': 'env_nlp',
    'explainability': 'env_explain',
    'api': 'env_api',
    'frontend': 'env_frontend',
    'pipeline': 'env_pipeline',
    "preprocessing": "env_preprocessing"
}