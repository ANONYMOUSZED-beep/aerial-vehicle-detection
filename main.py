#!/usr/bin/env python3
"""
Aerial Vehicle Detection System using YOLOv8

This is the main entry point for the Aerial Vehicle Detection system that can:
1. Train a vehicle detection model on aerial/drone images
2. Evaluate model performance with mAP, precision, recall metrics
3. Perform inference on images and videos with real-time detection

Usage:
    python main.py train --dataset path/to/dataset --epochs 100
    python main.py test --model path/to/model.pt --dataset path/to/test_data
    python main.py inference --model path/to/model.pt --input path/to/image_or_video
"""

import argparse
import sys
import os
import torch
from pathlib import Path

# Fix for Windows ultralytics Git issue
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
os.environ['YOLO_VERBOSE'] = 'False'

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from src.dataset_loader import AerialDatasetLoader
from src.model_trainer import ModelTrainer
from src.model_evaluator import ModelEvaluator
from src.inference_engine import InferenceEngine
from src.config import Config
from src.utils import setup_logging, check_gpu


def parse_arguments():
    """Parse command line arguments for different modes of operation."""
    parser = argparse.ArgumentParser(
        description="Aerial Vehicle Detection System using YOLOv8",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train a model:
    python main.py train --dataset ./data/aerial_vehicles --epochs 100 --batch-size 16
    
  Evaluate a model:
    python main.py test --model ./models/best.pt --dataset ./data/test
    
  Run inference on image:
    python main.py inference --model ./models/best.pt --input ./test_image.jpg
    
  Run inference on video:
    python main.py inference --model ./models/best.pt --input ./test_video.mp4 --output ./results/
        """
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Training arguments
    train_parser = subparsers.add_parser('train', help='Train a vehicle detection model')
    train_parser.add_argument('--dataset', type=str, required=True,
                            help='Path to dataset directory (YOLO format)')
    train_parser.add_argument('--epochs', type=int, default=100,
                            help='Number of training epochs (default: 100)')
    train_parser.add_argument('--batch-size', type=int, default=16,
                            help='Batch size for training (default: 16)')
    train_parser.add_argument('--img-size', type=int, default=640,
                            help='Input image size (default: 640)')
    train_parser.add_argument('--model', type=str, default='yolov8n.pt',
                            help='Pre-trained model to start from (default: yolov8n.pt)')
    train_parser.add_argument('--lr', type=float, default=0.01,
                            help='Learning rate (default: 0.01)')
    train_parser.add_argument('--save-dir', type=str, default='./models',
                            help='Directory to save trained models (default: ./models)')
    train_parser.add_argument('--resume', action='store_true',
                            help='Resume training from a checkpoint')
    train_parser.add_argument('--checkpoint', type=str, default=None,
                            help='Path to checkpoint (.pt) to resume from')
    
    # Testing/Evaluation arguments
    test_parser = subparsers.add_parser('test', help='Evaluate a trained model')
    test_parser.add_argument('--model', type=str, required=True,
                           help='Path to trained model file (.pt)')
    test_parser.add_argument('--dataset', type=str, required=True,
                           help='Path to test dataset directory')
    test_parser.add_argument('--img-size', type=int, default=640,
                           help='Input image size (default: 640)')
    test_parser.add_argument('--conf-thresh', type=float, default=0.25,
                           help='Confidence threshold (default: 0.25)')
    test_parser.add_argument('--iou-thresh', type=float, default=0.45,
                           help='IoU threshold for NMS (default: 0.45)')
    test_parser.add_argument('--save-results', type=str, default='./results',
                           help='Directory to save evaluation results (default: ./results)')
    
    # Inference arguments
    inference_parser = subparsers.add_parser('inference', help='Run inference on images or videos')
    inference_parser.add_argument('--model', type=str, required=True,
                                help='Path to trained model file (.pt)')
    inference_parser.add_argument('--input', type=str, required=True,
                                help='Path to input image, video, or directory')
    inference_parser.add_argument('--output', type=str, default='./results',
                                help='Output directory for results (default: ./results)')
    inference_parser.add_argument('--conf-thresh', type=float, default=0.25,
                                help='Confidence threshold (default: 0.25)')
    inference_parser.add_argument('--iou-thresh', type=float, default=0.45,
                                help='IoU threshold for NMS (default: 0.45)')
    inference_parser.add_argument('--save-txt', action='store_true',
                                help='Save results in YOLO format text files')
    inference_parser.add_argument('--save-conf', action='store_true',
                                help='Include confidence scores in saved results')
    inference_parser.add_argument('--show', action='store_true',
                                help='Display results in real-time')
    
    return parser.parse_args()


def main():
    """Main function to handle different operation modes."""
    args = parse_arguments()
    
    if not args.mode:
        print("Error: Please specify a mode (train, test, or inference)")
        print("Use --help for more information")
        sys.exit(1)
    
    # Setup logging
    setup_logging()
    
    # Check GPU availability
    device = check_gpu()
    print(f"Using device: {device}")
    
    # Create output directories if they don't exist
    os.makedirs(args.save_dir if hasattr(args, 'save_dir') else './models', exist_ok=True)
    if hasattr(args, 'output'):
        os.makedirs(args.output, exist_ok=True)
    if hasattr(args, 'save_results'):
        os.makedirs(args.save_results, exist_ok=True)
    
    try:
        if args.mode == 'train':
            print(f"🚀 Starting training on dataset: {args.dataset}")
            print(f"📊 Training parameters:")
            print(f"   - Epochs: {args.epochs}")
            print(f"   - Batch size: {args.batch_size}")
            print(f"   - Image size: {args.img_size}")
            print(f"   - Learning rate: {args.lr}")
            print(f"   - Pre-trained model: {args.model}")
            
            # Load dataset (validation done inside trainer)
            dataset_loader = AerialDatasetLoader(args.dataset)

            # If resuming, require a checkpoint path or fail
            if args.resume or args.checkpoint:
                checkpoint = args.checkpoint
                if not checkpoint:
                    print("❌ To resume training you must provide --checkpoint PATH")
                    sys.exit(1)

                print(f"🔁 Resuming training from checkpoint: {checkpoint}")
                trainer = ModelTrainer(
                    model_name=checkpoint,
                    dataset_path=args.dataset,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    img_size=args.img_size,
                    lr=args.lr,
                    save_dir=args.save_dir,
                    device=device
                )

                best_model_path = trainer.resume_training(checkpoint_path=checkpoint, additional_epochs=args.epochs)
                print(f"✅ Resumed training completed! Best model saved at: {best_model_path}")
            else:
                # Initialize trainer and start fresh training
                trainer = ModelTrainer(
                    model_name=args.model,
                    dataset_path=args.dataset,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    img_size=args.img_size,
                    lr=args.lr,
                    save_dir=args.save_dir,
                    device=device
                )

                # Train the model
                best_model_path = trainer.train()
                print(f"✅ Training completed! Best model saved at: {best_model_path}")
            
        elif args.mode == 'test':
            print(f"🧪 Starting evaluation on model: {args.model}")
            print(f"📊 Test dataset: {args.dataset}")
            
            # Initialize evaluator
            evaluator = ModelEvaluator(
                model_path=args.model,
                device=device,
                conf_thresh=args.conf_thresh,
                iou_thresh=args.iou_thresh,
                img_size=args.img_size
            )
            
            # Evaluate the model
            metrics = evaluator.evaluate_dataset(args.dataset, args.save_results)
            
            print(f"✅ Evaluation completed! Results saved to: {args.save_results}")
            print(f"📈 Performance metrics:")
            print(f"   - mAP@50: {metrics.get('map50', 'N/A'):.3f}")
            print(f"   - mAP@50-95: {metrics.get('map', 'N/A'):.3f}")
            print(f"   - Precision: {metrics.get('precision', 'N/A'):.3f}")
            print(f"   - Recall: {metrics.get('recall', 'N/A'):.3f}")
            
        elif args.mode == 'inference':
            print(f"🔍 Starting inference with model: {args.model}")
            print(f"📁 Input: {args.input}")
            print(f"💾 Output: {args.output}")
            
            # Initialize inference engine
            inference_engine = InferenceEngine(
                model_path=args.model,
                device=device,
                conf_thresh=args.conf_thresh,
                iou_thresh=args.iou_thresh
            )
            
            # Determine input type and run inference
            input_path = Path(args.input)
            
            if input_path.is_file():
                if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                    # Single image inference
                    print("🖼️  Processing single image...")
                    results = inference_engine.detect_from_image(
                        str(input_path), 
                        args.output,
                        save_txt=args.save_txt,
                        save_conf=args.save_conf,
                        show=args.show
                    )
                    print(f"✅ Detection completed! Found {len(results)} objects")
                    
                elif input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                    # Video inference
                    print("🎥 Processing video...")
                    inference_engine.detect_from_video(
                        str(input_path),
                        args.output,
                        save_txt=args.save_txt,
                        save_conf=args.save_conf,
                        show=args.show
                    )
                    print("✅ Video processing completed!")
                    
                else:
                    print(f"❌ Unsupported file format: {input_path.suffix}")
                    sys.exit(1)
                    
            elif input_path.is_dir():
                # Directory of images
                print("📁 Processing directory of images...")
                total_detections = inference_engine.detect_from_directory(
                    str(input_path),
                    args.output,
                    save_txt=args.save_txt,
                    save_conf=args.save_conf
                )
                print(f"✅ Directory processing completed! Total detections: {total_detections}")
                
            else:
                print(f"❌ Input path does not exist: {args.input}")
                sys.exit(1)
                
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("🎉 Operation completed successfully!")


if __name__ == "__main__":
    main()