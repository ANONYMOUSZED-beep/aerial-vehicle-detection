#!/usr/bin/env python3
"""
Example script demonstrating the Aerial Vehicle Detection System

This script shows how to use the different components of the system:
1. Dataset preparation and validation
2. Model training
3. Model evaluation
4. Inference on images and videos

Run this script to see a complete workflow example.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.dataset_loader import AerialDatasetLoader, load_dataset
from src.model_trainer import ModelTrainer
from src.model_evaluator import ModelEvaluator
from src.inference_engine import InferenceEngine
from src.config import Config
from src.utils import setup_logging, check_gpu, get_system_info, Timer


def main():
    """Run complete example workflow."""
    
    print("🚀 Aerial Vehicle Detection System - Example Workflow")
    print("=" * 60)
    
    # Setup logging
    logger = setup_logging('INFO')
    
    # Check system capabilities
    print("\n1. 🔍 System Check")
    print("-" * 30)
    device = check_gpu()
    system_info = get_system_info()
    print(f"Device: {device}")
    print(f"CPUs: {system_info['cpu_count']}")
    print(f"RAM: {system_info['memory_total_gb']:.1f} GB")
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    example_dataset_path = project_root / 'data' / 'example_aerial_dataset'
    models_dir = project_root / 'models'
    results_dir = project_root / 'results' / 'example'
    
    # Create directories
    models_dir.mkdir(exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n2. 📁 Project Structure")
    print("-" * 30)
    print(f"Project root: {project_root}")
    print(f"Dataset path: {example_dataset_path}")
    print(f"Models dir: {models_dir}")
    print(f"Results dir: {results_dir}")
    
    # Example 1: Dataset Preparation (if dataset exists)
    if example_dataset_path.exists():
        print(f"\n3. 📊 Dataset Analysis")
        print("-" * 30)
        
        try:
            # Load and analyze dataset
            dataset_loader = AerialDatasetLoader(str(example_dataset_path))
            dataset_info = dataset_loader.get_dataset_info()
            
            print(f"Dataset classes: {dataset_info['class_names']}")
            print(f"Number of classes: {dataset_info['num_classes']}")
            
            for split, info in dataset_info['splits'].items():
                if info['num_images'] > 0:
                    print(f"{split.upper()}: {info['num_images']} images, {info['total_objects']} objects")
            
            # Create data.yaml if needed
            yaml_path = dataset_loader.create_yolo_dataset_yaml()
            print(f"YOLO config: {yaml_path}")
            
        except Exception as e:
            print(f"Dataset analysis failed: {e}")
            print("Creating dummy dataset structure for demonstration...")
            create_dummy_dataset(example_dataset_path)
    else:
        print(f"\n3. 📊 Creating Example Dataset Structure")
        print("-" * 30)
        create_dummy_dataset(example_dataset_path)
    
    # Example 2: Training (demonstration - use small parameters)
    print(f"\n4. 🎯 Model Training Example")
    print("-" * 30)
    
    try:
        # Initialize trainer with small parameters for demonstration
        trainer = ModelTrainer(
            model_name='yolov8n.pt',  # Smallest model for demo
            dataset_path=str(example_dataset_path),
            epochs=2,  # Very few epochs for demo
            batch_size=4,  # Small batch size
            img_size=320,  # Smaller image size
            lr=0.01,
            save_dir=str(models_dir),
            device=device
        )
        
        print("Training configuration:")
        print(f"  Model: yolov8n.pt (nano - fastest)")
        print(f"  Epochs: 2 (demo only)")
        print(f"  Batch size: 4")
        print(f"  Image size: 320")
        print(f"  Device: {device}")
        
        # Start training (with timer)
        with Timer("Model training"):
            if example_dataset_path.exists() and (example_dataset_path / 'data.yaml').exists():
                best_model_path = trainer.train()
                print(f"✅ Training completed! Model saved: {best_model_path}")
            else:
                print("⚠️  Skipping training - no valid dataset found")
                best_model_path = None
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        best_model_path = None
    
    # Example 3: Evaluation (if model was trained)
    if best_model_path and Path(best_model_path).exists():
        print(f"\n5. 🧪 Model Evaluation")
        print("-" * 30)
        
        try:
            evaluator = ModelEvaluator(
                model_path=best_model_path,
                device=device,
                conf_thresh=0.25,
                iou_thresh=0.45
            )
            
            # Evaluate on validation set
            with Timer("Model evaluation"):
                metrics = evaluator.evaluate_dataset(
                    str(example_dataset_path),
                    str(results_dir / 'evaluation')
                )
            
            print("📈 Evaluation Results:")
            print(f"  mAP@50: {metrics.get('map50', 0):.3f}")
            print(f"  mAP@50-95: {metrics.get('map', 0):.3f}")
            print(f"  Precision: {metrics.get('precision', 0):.3f}")
            print(f"  Recall: {metrics.get('recall', 0):.3f}")
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
    else:
        print(f"\n5. 🧪 Model Evaluation")
        print("-" * 30)
        print("⚠️  Skipping evaluation - no trained model available")
    
    # Example 4: Inference (use pretrained model if available)
    print(f"\n6. 🔍 Inference Example")
    print("-" * 30)
    
    # Try to use trained model or fall back to pretrained
    model_path = best_model_path if best_model_path and Path(best_model_path).exists() else 'yolov8n.pt'
    
    try:
        inference_engine = InferenceEngine(
            model_path=model_path,
            device=device,
            conf_thresh=0.25,
            iou_thresh=0.45
        )
        
        print(f"Using model: {model_path}")
        print(f"Model classes: {list(inference_engine.class_names.values())}")
        
        # Create dummy test image if none exists
        test_image_path = create_dummy_test_image(project_root / 'examples' / 'test_image.jpg')
        
        if test_image_path.exists():
            print(f"Processing test image: {test_image_path}")
            
            with Timer("Image inference"):
                detections = inference_engine.detect_from_image(
                    str(test_image_path),
                    str(results_dir / 'inference'),
                    show=False
                )
            
            print(f"✅ Found {len(detections)} detections")
            for det in detections[:3]:  # Show first 3 detections
                print(f"  - {det['class_name']}: {det['confidence']:.3f}")
        else:
            print("⚠️  No test image available for inference demo")
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
    
    # Example 5: Configuration Management
    print(f"\n7. ⚙️  Configuration Management")
    print("-" * 30)
    
    try:
        # Save current configuration
        config_path = results_dir / 'example_config.json'
        Config.save_config(str(config_path))
        print(f"Configuration saved: {config_path}")
        
        # Display key settings
        print("Key configuration settings:")
        print(f"  Training epochs: {Config.TRAINING_CONFIG['epochs']}")
        print(f"  Batch size: {Config.TRAINING_CONFIG['batch_size']}")
        print(f"  Learning rate: {Config.TRAINING_CONFIG['learning_rate']}")
        print(f"  Confidence threshold: {Config.INFERENCE_CONFIG['conf_thresh']}")
        print(f"  Available models: {Config.MODEL_CONFIG['available_models'][:3]}...")
        
    except Exception as e:
        print(f"Configuration management failed: {e}")
    
    # Summary
    print(f"\n8. 📋 Summary")
    print("-" * 30)
    print("✅ Example workflow completed!")
    print(f"📁 Results saved to: {results_dir}")
    print(f"📖 Check the README.md for detailed usage instructions")
    print(f"🚀 You can now use the main.py CLI for full functionality")
    
    print(f"\n💡 Next Steps:")
    print("1. Prepare your own aerial vehicle dataset")
    print("2. Train a model: python main.py train --dataset your_data --epochs 100")
    print("3. Evaluate: python main.py test --model models/best.pt --dataset your_data")
    print("4. Run inference: python main.py inference --model models/best.pt --input test.jpg")


def create_dummy_dataset(dataset_path: Path):
    """Create a dummy dataset structure for demonstration."""
    
    dataset_path.mkdir(parents=True, exist_ok=True)
    
    # Create directory structure
    dirs_to_create = [
        'images/train',
        'images/val', 
        'labels/train',
        'labels/val'
    ]
    
    for dir_path in dirs_to_create:
        (dataset_path / dir_path).mkdir(parents=True, exist_ok=True)
    
    # Create dummy data.yaml
    yaml_content = """# Aerial Vehicle Detection Dataset Configuration
path: {}
train: images/train
val: images/val

# Number of classes
nc: 7

# Class names
names: ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'van', 'trailer']
""".format(str(dataset_path.absolute()).replace('\\', '/'))
    
    with open(dataset_path / 'data.yaml', 'w') as f:
        f.write(yaml_content)
    
    print(f"✅ Created dummy dataset structure at: {dataset_path}")
    print("⚠️  This is just a structure - add your own images and labels for actual training")


def create_dummy_test_image(image_path: Path):
    """Create a dummy test image for inference demonstration."""
    try:
        import numpy as np
        import cv2
        
        # Create a simple test image (aerial view simulation)
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Add some background (road-like texture)
        image[:] = (50, 50, 50)  # Dark background
        
        # Add some rectangles to simulate vehicles
        cv2.rectangle(image, (100, 100), (150, 180), (0, 255, 0), -1)  # Green car
        cv2.rectangle(image, (200, 150), (280, 220), (255, 0, 0), -1)  # Blue truck
        cv2.rectangle(image, (400, 300), (440, 360), (0, 0, 255), -1)  # Red car
        
        # Add some noise
        noise = np.random.randint(0, 50, image.shape, dtype=np.uint8)
        image = cv2.add(image, noise)
        
        # Save image
        image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(image_path), image)
        
        print(f"✅ Created dummy test image: {image_path}")
        return image_path
        
    except Exception as e:
        print(f"❌ Could not create dummy test image: {e}")
        return Path("nonexistent.jpg")


if __name__ == "__main__":
    main()