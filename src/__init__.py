"""
Aerial Vehicle Detection System

A comprehensive deep learning system for detecting and classifying vehicles
from aerial or drone images and videos using YOLOv8.

Main Components:
- dataset_loader: Dataset loading and preprocessing
- model_trainer: Model training functionality  
- model_evaluator: Model evaluation and metrics
- inference_engine: Real-time inference capabilities
- config: Configuration management
- utils: Utility functions and helpers

Usage:
    from src.dataset_loader import load_dataset
    from src.model_trainer import train_model
    from src.model_evaluator import evaluate_model
    from src.inference_engine import detect_from_image
"""

__version__ = "1.0.0"
__author__ = "Aerial VH Detection Team"
__email__ = "contact@aerialvh.com"

# Import main functions for easy access
from .dataset_loader import load_dataset, AerialDatasetLoader
from .model_trainer import train_model, ModelTrainer
from .model_evaluator import evaluate_model, ModelEvaluator
from .inference_engine import detect_from_image, detect_from_video, InferenceEngine
from .config import Config, get_config
from .utils import setup_logging, check_gpu, get_system_info

__all__ = [
    # Dataset functionality
    'load_dataset',
    'AerialDatasetLoader',
    
    # Training functionality
    'train_model', 
    'ModelTrainer',
    
    # Evaluation functionality
    'evaluate_model',
    'ModelEvaluator',
    
    # Inference functionality
    'detect_from_image',
    'detect_from_video', 
    'InferenceEngine',
    
    # Configuration
    'Config',
    'get_config',
    
    # Utilities
    'setup_logging',
    'check_gpu',
    'get_system_info'
]