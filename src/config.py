"""
Configuration settings for Aerial Vehicle Detection System

This module contains all configuration parameters, model settings, and paths
used throughout the aerial vehicle detection system.
"""

import os
from pathlib import Path
from typing import Dict, List, Any


class Config:
    """
    Central configuration class for the aerial vehicle detection system.
    Contains all hyperparameters, paths, and settings.
    """
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / 'data'
    MODELS_DIR = PROJECT_ROOT / 'models'
    RESULTS_DIR = PROJECT_ROOT / 'results'
    
    # Dataset configuration
    DATASET_CONFIG = {
        'default_classes': {
            0: 'car',
            1: 'truck',
            2: 'bus', 
            3: 'motorcycle',
            4: 'bicycle',
            5: 'van',
            6: 'trailer'
        },
        'image_extensions': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
        'label_extension': '.txt'
    }
    
    # Model configuration
    MODEL_CONFIG = {
        'available_models': [
            'yolov8n.pt',  # Nano - fastest, least accurate
            'yolov8s.pt',  # Small - balanced
            'yolov8m.pt',  # Medium - more accurate
            'yolov8l.pt',  # Large - very accurate
            'yolov8x.pt'   # Extra Large - most accurate, slowest
        ],
        'default_model': 'yolov8n.pt',
        'input_sizes': [320, 416, 512, 640, 736, 832, 896, 960, 1024, 1280],
        'default_input_size': 640
    }
    
    # Training hyperparameters
    TRAINING_CONFIG = {
        'epochs': 100,
        'batch_size': 16,
        'learning_rate': 0.01,
        'patience': 50,
        'save_period': 10,
        'optimizer': 'AdamW',
        'cos_lr': True,
        'amp': True,  # Automatic Mixed Precision
        'seed': 42,
        
        # Data augmentation
        'augmentation': {
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0
        }
    }
    
    # Inference configuration
    INFERENCE_CONFIG = {
        'conf_thresh': 0.25,
        'iou_thresh': 0.45,
        'max_det': 1000,
        'agnostic_nms': False,
        'retina_masks': False
    }
    
    # Evaluation configuration
    EVALUATION_CONFIG = {
        'iou_thresholds': [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
        'save_plots': True,
        'save_json': True,
        'save_hybrid': False
    }
    
    # Visualization configuration
    VISUALIZATION_CONFIG = {
        'colors': [
            (255, 0, 0),     # Red - car
            (0, 255, 0),     # Green - truck
            (0, 0, 255),     # Blue - bus
            (255, 255, 0),   # Yellow - motorcycle
            (255, 0, 255),   # Magenta - bicycle
            (0, 255, 255),   # Cyan - van
            (128, 0, 128)    # Purple - trailer
        ],
        'line_thickness': 2,
        'font_size': 0.6,
        'font_thickness': 2
    }
    
    # Hardware configuration
    HARDWARE_CONFIG = {
        'device': 'auto',  # 'auto', 'cpu', 'cuda', '0', '1', etc.
        'workers': 4,
        'pin_memory': True,
        'persistent_workers': True
    }
    
    # Logging configuration
    LOGGING_CONFIG = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'log_file': 'aerial_detection.log'
    }
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist."""
        directories = [
            cls.DATA_DIR,
            cls.MODELS_DIR,
            cls.RESULTS_DIR,
            cls.DATA_DIR / 'raw',
            cls.DATA_DIR / 'processed',
            cls.RESULTS_DIR / 'training',
            cls.RESULTS_DIR / 'evaluation',
            cls.RESULTS_DIR / 'inference'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_model_config(cls, model_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific model.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            Dict[str, Any]: Model-specific configuration
        """
        base_config = cls.MODEL_CONFIG.copy()
        
        # Model-specific optimizations
        if 'yolov8n' in model_name:
            base_config.update({
                'batch_size': 32,
                'learning_rate': 0.01,
                'workers': 8
            })
        elif 'yolov8s' in model_name:
            base_config.update({
                'batch_size': 24,
                'learning_rate': 0.01,
                'workers': 6
            })
        elif 'yolov8m' in model_name:
            base_config.update({
                'batch_size': 16,
                'learning_rate': 0.01,
                'workers': 4
            })
        elif 'yolov8l' in model_name:
            base_config.update({
                'batch_size': 12,
                'learning_rate': 0.008,
                'workers': 4
            })
        elif 'yolov8x' in model_name:
            base_config.update({
                'batch_size': 8,
                'learning_rate': 0.008,
                'workers': 2
            })
        
        return base_config
    
    @classmethod
    def get_device_config(cls, device: str = None) -> Dict[str, Any]:
        """
        Get device-specific configuration.
        
        Args:
            device (str): Device identifier
            
        Returns:
            Dict[str, Any]: Device-specific configuration
        """
        import torch
        
        if device is None or device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        config = {
            'device': device,
            'amp': True,  # Automatic Mixed Precision
            'half': False  # FP16 precision
        }
        
        if device == 'cpu':
            config.update({
                'workers': 2,
                'batch_size': 8,
                'amp': False,
                'half': False
            })
        elif 'cuda' in device:
            # Get GPU memory
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                
                if gpu_memory < 4:  # Less than 4GB
                    config.update({
                        'workers': 2,
                        'batch_size': 8,
                        'half': True
                    })
                elif gpu_memory < 8:  # Less than 8GB
                    config.update({
                        'workers': 4,
                        'batch_size': 16,
                        'half': False
                    })
                else:  # 8GB or more
                    config.update({
                        'workers': 8,
                        'batch_size': 32,
                        'half': False
                    })
        
        return config
    
    @classmethod
    def validate_config(cls) -> bool:
        """
        Validate configuration parameters.
        
        Returns:
            bool: True if configuration is valid
        """
        try:
            # Check if required directories can be created
            cls.create_directories()
            
            # Validate model names
            for model in cls.MODEL_CONFIG['available_models']:
                if not model.endswith('.pt'):
                    print(f"Warning: Model {model} doesn't have .pt extension")
            
            # Validate training parameters
            training = cls.TRAINING_CONFIG
            if training['epochs'] <= 0:
                raise ValueError("Epochs must be positive")
            if training['batch_size'] <= 0:
                raise ValueError("Batch size must be positive")
            if not 0 < training['learning_rate'] < 1:
                raise ValueError("Learning rate must be between 0 and 1")
            
            # Validate inference parameters
            inference = cls.INFERENCE_CONFIG
            if not 0 < inference['conf_thresh'] < 1:
                raise ValueError("Confidence threshold must be between 0 and 1")
            if not 0 < inference['iou_thresh'] < 1:
                raise ValueError("IoU threshold must be between 0 and 1")
            
            print("✅ Configuration validation passed")
            return True
            
        except Exception as e:
            print(f"❌ Configuration validation failed: {e}")
            return False
    
    @classmethod
    def save_config(cls, filepath: str):
        """
        Save current configuration to a file.
        
        Args:
            filepath (str): Path to save configuration
        """
        import json
        
        config_dict = {
            'dataset': cls.DATASET_CONFIG,
            'model': cls.MODEL_CONFIG,
            'training': cls.TRAINING_CONFIG,
            'inference': cls.INFERENCE_CONFIG,
            'evaluation': cls.EVALUATION_CONFIG,
            'visualization': cls.VISUALIZATION_CONFIG,
            'hardware': cls.HARDWARE_CONFIG,
            'logging': cls.LOGGING_CONFIG
        }
        
        # Convert Path objects to strings for JSON serialization
        def path_converter(obj):
            if isinstance(obj, Path):
                return str(obj)
            return obj
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2, default=path_converter)
        
        print(f"Configuration saved to: {filepath}")
    
    @classmethod
    def load_config(cls, filepath: str):
        """
        Load configuration from a file.
        
        Args:
            filepath (str): Path to configuration file
        """
        import json
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Update class attributes
        cls.DATASET_CONFIG.update(config_dict.get('dataset', {}))
        cls.MODEL_CONFIG.update(config_dict.get('model', {}))
        cls.TRAINING_CONFIG.update(config_dict.get('training', {}))
        cls.INFERENCE_CONFIG.update(config_dict.get('inference', {}))
        cls.EVALUATION_CONFIG.update(config_dict.get('evaluation', {}))
        cls.VISUALIZATION_CONFIG.update(config_dict.get('visualization', {}))
        cls.HARDWARE_CONFIG.update(config_dict.get('hardware', {}))
        cls.LOGGING_CONFIG.update(config_dict.get('logging', {}))
        
        print(f"Configuration loaded from: {filepath}")


# Environment-specific configurations
class DevelopmentConfig(Config):
    """Development environment configuration."""
    DEBUG = True
    TRAINING_CONFIG = Config.TRAINING_CONFIG.copy()
    TRAINING_CONFIG.update({
        'epochs': 10,  # Fewer epochs for testing
        'batch_size': 4,  # Smaller batch for development
    })


class ProductionConfig(Config):
    """Production environment configuration."""
    DEBUG = False
    TRAINING_CONFIG = Config.TRAINING_CONFIG.copy()
    TRAINING_CONFIG.update({
        'epochs': 300,  # More epochs for production
        'patience': 100,  # More patience
    })


class TestingConfig(Config):
    """Testing environment configuration."""
    DEBUG = True
    TRAINING_CONFIG = Config.TRAINING_CONFIG.copy()
    TRAINING_CONFIG.update({
        'epochs': 2,  # Minimal epochs for testing
        'batch_size': 2,  # Small batch for testing
    })


# Configuration factory
def get_config(environment: str = 'development') -> Config:
    """
    Get configuration based on environment.
    
    Args:
        environment (str): Environment name ('development', 'production', 'testing')
        
    Returns:
        Config: Configuration class instance
    """
    configs = {
        'development': DevelopmentConfig,
        'production': ProductionConfig,
        'testing': TestingConfig
    }
    
    config_class = configs.get(environment, Config)
    return config_class


if __name__ == "__main__":
    # Test configuration
    config = Config()
    
    # Validate and create directories
    if config.validate_config():
        config.create_directories()
        print("Configuration setup completed successfully!")
    
    # Save default configuration
    config.save_config('config.json')
    
    # Test device configuration
    device_config = config.get_device_config()
    print(f"Device configuration: {device_config}")
    
    # Test model configuration
    model_config = config.get_model_config('yolov8n.pt')
    print(f"Model configuration: {model_config}")