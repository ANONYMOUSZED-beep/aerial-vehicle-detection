"""
Model Trainer for Aerial Vehicle Detection using YOLOv8

This module handles training YOLOv8 models for aerial vehicle detection.
It includes functionality for:
- Loading pretrained YOLOv8 models
- Fine-tuning on aerial vehicle datasets
- GPU support and automatic device selection
- Model checkpointing and saving
- Training progress monitoring and logging
"""

import os
import time
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import yaml
from ultralytics import YOLO
from ultralytics.utils import LOGGER
import matplotlib.pyplot as plt
import numpy as np


class ModelTrainer:
    """
    Handles training of YOLOv8 models for aerial vehicle detection.
    Supports various YOLOv8 model sizes and configurations.
    """
    
    def __init__(self,
                 model_name: str = 'yolov8n.pt',
                 dataset_path: str = None,
                 epochs: int = 100,
                 batch_size: int = 16,
                 img_size: int = 640,
                 lr: float = 0.01,
                 save_dir: str = './models',
                 device: str = 'auto',
                 patience: int = 50,
                 save_period: int = 10):
        """
        Initialize the model trainer.
        
        Args:
            model_name (str): YOLOv8 model variant ('yolov8n.pt', 'yolov8s.pt', etc.)
            dataset_path (str): Path to dataset directory with data.yaml
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
            img_size (int): Input image size
            lr (float): Learning rate
            save_dir (str): Directory to save trained models
            device (str): Device to use ('auto', 'cpu', 'cuda', or specific GPU)
            patience (int): Early stopping patience
            save_period (int): Save checkpoint every N epochs
        """
        self.model_name = model_name
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_size = img_size
        self.lr = lr
        self.save_dir = Path(save_dir)
        self.device = device
        self.patience = patience
        self.save_period = save_period
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = None
        self.training_results = {}
        
        # Setup logging
        self._setup_training_logging()
    
    def _setup_training_logging(self):
        """Setup logging configuration for training."""
        # YOLOv8 handles most logging internally
        # We'll add custom logging for our specific needs
        pass
    
    def _validate_dataset(self) -> str:
        """
        Validate dataset structure and return path to data.yaml.
        
        Returns:
            str: Path to data.yaml file
        """
        if not self.dataset_path or not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")
        
        # Look for data.yaml
        yaml_path = self.dataset_path / 'data.yaml'
        if not yaml_path.exists():
            raise FileNotFoundError(f"data.yaml not found in {self.dataset_path}")
        
        # Validate yaml content
        with open(yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        required_keys = ['train', 'val', 'nc', 'names']
        for key in required_keys:
            if key not in data_config:
                raise ValueError(f"Missing required key '{key}' in data.yaml")
        
        # Check if image directories exist
        base_path = Path(data_config.get('path', self.dataset_path))
        train_path = base_path / data_config['train']
        val_path = base_path / data_config['val']
        
        if not train_path.exists():
            raise FileNotFoundError(f"Training images directory not found: {train_path}")
        if not val_path.exists():
            print(f"Warning: Validation images directory not found: {val_path}")
        
        print(f"✅ Dataset validation passed")
        print(f"   Classes: {data_config['nc']}")
        print(f"   Class names: {data_config['names']}")
        print(f"   Train images: {train_path}")
        print(f"   Val images: {val_path}")
        
        return str(yaml_path)
    
    def _load_model(self) -> YOLO:
        """
        Load YOLOv8 model (pretrained or from checkpoint).
        
        Returns:
            YOLO: Loaded model instance
        """
        print(f"Loading model: {self.model_name}")
        
        # Check if it's a path to existing model or a model name
        if os.path.exists(self.model_name):
            print(f"Loading from checkpoint: {self.model_name}")
            # PyTorch 2.6+ enforces safe globals for torch.load when weights_only=True.
            # Allowlist known Ultralytics and torch container classes if available.
            try:
                # Allowlist common Ultralytics classes used in checkpoints so torch.load (weights_only)
                # can unpickle model objects on PyTorch 2.6+.
                safe_globals = []

                # DetectionModel
                try:
                    from ultralytics.nn.tasks import DetectionModel
                    safe_globals.append(DetectionModel)
                except Exception:
                    pass

                # Add torch Sequential container
                try:
                    safe_globals.append(torch.nn.modules.container.Sequential)
                except Exception:
                    pass
                # Add common torch conv classes
                try:
                    safe_globals.append(torch.nn.modules.conv.Conv2d)
                except Exception:
                    pass
                try:
                    safe_globals.append(torch.nn.Conv2d)
                except Exception:
                    pass

                # Collect classes from ultralytics.nn.modules (e.g., conv, head, etc.)
                try:
                    import ultralytics.nn.modules as umod
                    for name in dir(umod):
                        obj = getattr(umod, name)
                        if isinstance(obj, type):
                            safe_globals.append(obj)
                except Exception:
                    # Try the modules package more generally
                    try:
                        import ultralytics.nn as unn
                        if hasattr(unn, 'modules'):
                            mod = unn.modules
                            for name in dir(mod):
                                obj = getattr(mod, name)
                                if isinstance(obj, type):
                                    safe_globals.append(obj)
                    except Exception:
                        pass

                # Deduplicate and add
                if safe_globals:
                    unique = []
                    seen = set()
                    for cls in safe_globals:
                        key = (getattr(cls, '__module__', ''), getattr(cls, '__name__', ''))
                        if key not in seen:
                            seen.add(key)
                            unique.append(cls)
                    if unique:
                        torch.serialization.add_safe_globals(unique)
            except Exception:
                # If anything unexpected happens, continue — YOLO will surface an informative error
                pass

            # PyTorch 2.6+ default changed weights_only; to load existing checkpoints that
            # contain module objects we temporarily call torch.load with weights_only=False.
            # This is only done when loading a trusted checkpoint file (user-provided).
            _orig_torch_load = torch.load
            def _patched_torch_load(f, *args, **kwargs):
                if 'weights_only' not in kwargs:
                    kwargs['weights_only'] = False
                return _orig_torch_load(f, *args, **kwargs)

            torch.load = _patched_torch_load
            try:
                model = YOLO(self.model_name)
            finally:
                # restore original
                torch.load = _orig_torch_load
        else:
            print(f"Loading pretrained model: {self.model_name}")
            model = YOLO(self.model_name)
        
        # Print model info
        print(f"Model architecture: {model.model}")
        print(f"Model parameters: {sum(p.numel() for p in model.model.parameters()):,}")
        
        return model
    
    def _setup_training_config(self) -> Dict[str, Any]:
        """
        Setup training configuration parameters.
        
        Returns:
            Dict[str, Any]: Training configuration
        """
        config = {
            'epochs': self.epochs,
            'batch': self.batch_size,
            'imgsz': self.img_size,
            'lr0': self.lr,
            'device': self.device,
            'patience': self.patience,
            'save_period': self.save_period,
            'project': str(self.save_dir),
            'name': f'aerial_vehicle_detection_{int(time.time())}',
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'AdamW',
            'verbose': True,
            'seed': 42,
            'deterministic': True,
            'single_cls': False,
            'rect': False,
            'cos_lr': True,
            'close_mosaic': 10,
            'resume': False,
            'amp': True,  # Automatic Mixed Precision
            'fraction': 1.0,
            'profile': False,
            'freeze': None,
            # Data augmentation parameters
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
            'copy_paste': 0.0,
        }
        
        return config
    
    def train(self) -> str:
        """
        Train the YOLOv8 model on the aerial vehicle dataset.
        
        Returns:
            str: Path to the best trained model
        """
        print("🚀 Starting YOLOv8 training for aerial vehicle detection...")
        print(f"Configuration:")
        print(f"  Model: {self.model_name}")
        print(f"  Dataset: {self.dataset_path}")
        print(f"  Epochs: {self.epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Image size: {self.img_size}")
        print(f"  Learning rate: {self.lr}")
        print(f"  Device: {self.device}")
        
        try:
            # Validate dataset
            data_yaml_path = self._validate_dataset()
            
            # Load model
            self.model = self._load_model()
            
            # Setup training configuration
            train_config = self._setup_training_config()
            
            # Start training
            print(f"\n🎯 Starting training with {self.epochs} epochs...")
            start_time = time.time()
            
            # Train the model
            results = self.model.train(
                data=data_yaml_path,
                **train_config
            )
            
            training_time = time.time() - start_time
            print(f"⏱️  Training completed in {training_time/3600:.2f} hours")
            
            # Store training results
            self.training_results = results
            
            # Get best model path
            best_model_path = self.model.trainer.best
            print(f"✅ Best model saved at: {best_model_path}")
            
            # Generate training summary
            self._generate_training_summary(results, training_time)
            
            return str(best_model_path)
            
        except Exception as e:
            print(f"❌ Training failed: {str(e)}")
            raise e
    
    def _generate_training_summary(self, results, training_time: float):
        """
        Generate and save training summary with metrics and plots.
        
        Args:
            results: Training results from YOLOv8
            training_time (float): Total training time in seconds
        """
        print("\n📊 Generating training summary...")
        
        try:
            # Create summary directory
            summary_dir = self.save_dir / f"training_summary_{int(time.time())}"
            summary_dir.mkdir(exist_ok=True)
            
            # Training summary text
            summary_text = f"""
Aerial Vehicle Detection Training Summary
========================================

Model Configuration:
- Architecture: {self.model_name}
- Input Size: {self.img_size}x{self.img_size}
- Batch Size: {self.batch_size}
- Learning Rate: {self.lr}
- Epochs: {self.epochs}
- Device: {self.device}

Training Results:
- Training Time: {training_time/3600:.2f} hours
- Best Model: {self.model.trainer.best}
- Last Model: {self.model.trainer.last}

Dataset Information:
- Dataset Path: {self.dataset_path}
- Training Config: {self._validate_dataset()}

Performance Metrics:
(Detailed metrics available in YOLOv8 results)

Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            # Save summary
            with open(summary_dir / 'training_summary.txt', 'w') as f:
                f.write(summary_text)
            
            print(f"📋 Training summary saved to: {summary_dir}")
            
        except Exception as e:
            print(f"Warning: Could not generate training summary: {e}")
    
    def resume_training(self, checkpoint_path: str, additional_epochs: int = None) -> str:
        """
        Resume training from a checkpoint.
        
        Args:
            checkpoint_path (str): Path to checkpoint file
            additional_epochs (int): Additional epochs to train (optional)
            
        Returns:
            str: Path to the best trained model
        """
        print(f"📂 Resuming training from checkpoint: {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Update epochs if specified
        if additional_epochs:
            self.epochs = additional_epochs
            print(f"Training for {additional_epochs} additional epochs")
        
        # Update model path to checkpoint
        self.model_name = checkpoint_path
        
        # Train (resume) the model
        return self.train()
    
    def validate_model(self, model_path: str = None, data_path: str = None) -> Dict:
        """
        Validate a trained model on the validation set.
        
        Args:
            model_path (str): Path to model file (optional, uses last trained model)
            data_path (str): Path to validation dataset (optional, uses training dataset)
            
        Returns:
            Dict: Validation results
        """
        if model_path is None:
            if self.model is None:
                raise ValueError("No model available. Train a model first or provide model_path")
            model = self.model
        else:
            model = YOLO(model_path)
        
        if data_path is None:
            data_path = self._validate_dataset()
        
        print(f"🧪 Validating model on dataset...")
        
        # Run validation
        results = model.val(
            data=data_path,
            imgsz=self.img_size,
            batch=self.batch_size,
            device=self.device,
            verbose=True
        )
        
        print(f"✅ Validation completed")
        return results
    
    def export_model(self, 
                    model_path: str = None, 
                    format: str = 'onnx',
                    optimize: bool = True) -> str:
        """
        Export trained model to different formats.
        
        Args:
            model_path (str): Path to model file (optional)
            format (str): Export format ('onnx', 'torchscript', 'coreml', etc.)
            optimize (bool): Optimize for inference
            
        Returns:
            str: Path to exported model
        """
        if model_path is None:
            if self.model is None:
                raise ValueError("No model available. Train a model first or provide model_path")
            model = self.model
        else:
            model = YOLO(model_path)
        
        print(f"📦 Exporting model to {format} format...")
        
        # Export model
        exported_path = model.export(
            format=format,
            imgsz=self.img_size,
            optimize=optimize,
            half=False,  # FP16 quantization
            int8=False,  # INT8 quantization
            dynamic=False,  # Dynamic input shapes
            simplify=True,  # Simplify ONNX model
            opset=None,  # ONNX opset version
        )
        
        print(f"✅ Model exported to: {exported_path}")
        return str(exported_path)


def train_model(dataset_path: str,
               model_name: str = 'yolov8n.pt',
               epochs: int = 100,
               batch_size: int = 16,
               img_size: int = 640,
               lr: float = 0.01,
               save_dir: str = './models',
               device: str = 'auto') -> str:
    """
    Main function to train an aerial vehicle detection model.
    
    Args:
        dataset_path (str): Path to dataset directory
        model_name (str): YOLOv8 model variant
        epochs (int): Number of training epochs
        batch_size (int): Training batch size
        img_size (int): Input image size
        lr (float): Learning rate
        save_dir (str): Directory to save models
        device (str): Device to use for training
        
    Returns:
        str: Path to the best trained model
    """
    trainer = ModelTrainer(
        model_name=model_name,
        dataset_path=dataset_path,
        epochs=epochs,
        batch_size=batch_size,
        img_size=img_size,
        lr=lr,
        save_dir=save_dir,
        device=device
    )
    
    return trainer.train()


if __name__ == "__main__":
    # Example usage
    dataset_path = "./data/aerial_vehicles"
    
    try:
        # Train a model
        best_model = train_model(
            dataset_path=dataset_path,
            model_name='yolov8n.pt',
            epochs=50,
            batch_size=16,
            img_size=640,
            lr=0.01,
            device='auto'
        )
        
        print(f"Training completed! Best model: {best_model}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()