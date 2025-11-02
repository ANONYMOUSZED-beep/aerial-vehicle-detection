"""
Model Evaluator for Aerial Vehicle Detection

This module handles evaluation of trained YOLOv8 models for aerial vehicle detection.
It provides comprehensive evaluation metrics including:
- mAP (mean Average Precision) at different IoU thresholds
- Precision and Recall metrics
- Per-class performance analysis
- Confusion matrix generation
- Detection visualization and analysis
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from ultralytics.utils.metrics import ConfusionMatrix
import torch
from sklearn.metrics import classification_report
import pandas as pd


class ModelEvaluator:
    """
    Comprehensive evaluation system for aerial vehicle detection models.
    Provides detailed metrics and visualizations for model performance analysis.
    """
    
    def __init__(self,
                 model_path: str,
                 device: str = 'auto',
                 conf_thresh: float = 0.25,
                 iou_thresh: float = 0.45,
                 img_size: int = 640):
        """
        Initialize the model evaluator.
        
        Args:
            model_path (str): Path to trained model file
            device (str): Device to use for evaluation ('auto', 'cpu', 'cuda')
            conf_thresh (float): Confidence threshold for detections
            iou_thresh (float): IoU threshold for NMS
            img_size (int): Input image size
        """
        self.model_path = Path(model_path)
        self.device = device
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.img_size = img_size
        
        # Load model
        self.model = self._load_model()
        self.class_names = self.model.names
        self.num_classes = len(self.class_names)
        
        print(f"✅ Model loaded: {model_path}")
        print(f"📊 Classes: {list(self.class_names.values())}")
        print(f"🎯 Confidence threshold: {conf_thresh}")
        print(f"🔗 IoU threshold: {iou_thresh}")
    
    def _load_model(self) -> YOLO:
        """Load the trained YOLO model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        model = YOLO(str(self.model_path))
        return model
    
    def evaluate_dataset(self, 
                        dataset_path: str, 
                        save_dir: str = './results',
                        split: str = 'val') -> Dict[str, float]:
        """
        Evaluate model on a complete dataset.
        
        Args:
            dataset_path (str): Path to dataset directory
            save_dir (str): Directory to save evaluation results
            split (str): Dataset split to evaluate ('val', 'test')
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        print(f"🧪 Evaluating model on {split} dataset...")
        
        dataset_path = Path(dataset_path)
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Find data.yaml file
        data_yaml = dataset_path / 'data.yaml'
        if not data_yaml.exists():
            raise FileNotFoundError(f"data.yaml not found in {dataset_path}")
        
        # Run validation using YOLOv8's built-in evaluation
        start_time = time.time()
        results = self.model.val(
            data=str(data_yaml),
            split=split,
            imgsz=self.img_size,
            batch=1,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            device=self.device,
            plots=True,
            save_json=True,
            save_hybrid=False,
            verbose=True,
            project=str(save_dir),
            name=f'evaluation_{split}_{int(time.time())}'
        )
        
        eval_time = time.time() - start_time
        print(f"⏱️  Evaluation completed in {eval_time:.2f} seconds")
        
        # Extract metrics
        metrics = self._extract_metrics(results)
        
        # Generate detailed evaluation report
        self._generate_evaluation_report(results, metrics, save_dir, eval_time)
        
        # Generate visualizations
        self._generate_evaluation_plots(results, save_dir)
        
        return metrics
    
    def _extract_metrics(self, results) -> Dict[str, float]:
        """
        Extract key metrics from YOLOv8 validation results.
        
        Args:
            results: YOLOv8 validation results object
            
        Returns:
            Dict[str, float]: Key evaluation metrics
        """
        metrics = {}
        
        # Main metrics
        metrics['map50'] = float(results.box.map50) if hasattr(results.box, 'map50') else 0.0
        metrics['map'] = float(results.box.map) if hasattr(results.box, 'map') else 0.0
        metrics['precision'] = float(results.box.mp) if hasattr(results.box, 'mp') else 0.0
        metrics['recall'] = float(results.box.mr) if hasattr(results.box, 'mr') else 0.0
        metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0.0
        
        # Per-class metrics if available
        if hasattr(results.box, 'ap_class_index') and hasattr(results.box, 'ap'):
            for i, class_idx in enumerate(results.box.ap_class_index):
                class_name = self.class_names.get(int(class_idx), f'class_{class_idx}')
                if i < len(results.box.ap):
                    metrics[f'ap_{class_name}'] = float(results.box.ap[i])
        
        return metrics
    
    def _generate_evaluation_report(self, 
                                  results, 
                                  metrics: Dict[str, float], 
                                  save_dir: Path,
                                  eval_time: float):
        """
        Generate a comprehensive evaluation report.
        
        Args:
            results: YOLOv8 validation results
            metrics (Dict[str, float]): Evaluation metrics
            save_dir (Path): Directory to save the report
            eval_time (float): Evaluation time in seconds
        """
        report_path = save_dir / 'evaluation_report.txt'
        
        report_content = f"""
Aerial Vehicle Detection Model Evaluation Report
==============================================

Model Information:
- Model Path: {self.model_path}
- Model Type: YOLOv8
- Input Size: {self.img_size}x{self.img_size}
- Device: {self.device}

Evaluation Configuration:
- Confidence Threshold: {self.conf_thresh}
- IoU Threshold: {self.iou_thresh}
- Evaluation Time: {eval_time:.2f} seconds

Overall Performance Metrics:
- mAP@50: {metrics.get('map50', 0):.4f}
- mAP@50-95: {metrics.get('map', 0):.4f}
- Precision: {metrics.get('precision', 0):.4f}
- Recall: {metrics.get('recall', 0):.4f}
- F1-Score: {metrics.get('f1', 0):.4f}

Per-Class Performance:
"""
        
        # Add per-class metrics
        for key, value in metrics.items():
            if key.startswith('ap_'):
                class_name = key.replace('ap_', '')
                report_content += f"- {class_name}: {value:.4f}\n"
        
        report_content += f"""
Class Names: {list(self.class_names.values())}
Number of Classes: {self.num_classes}

Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        # Save report
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        # Also save metrics as JSON for programmatic access
        metrics_path = save_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"📋 Evaluation report saved to: {report_path}")
        print(f"📊 Metrics JSON saved to: {metrics_path}")
    
    def _generate_evaluation_plots(self, results, save_dir: Path):
        """
        Generate evaluation plots and visualizations.
        
        Args:
            results: YOLOv8 validation results
            save_dir (Path): Directory to save plots
        """
        print("📈 Generating evaluation plots...")
        
        try:
            # Create plots directory
            plots_dir = save_dir / 'plots'
            plots_dir.mkdir(exist_ok=True)
            
            # Precision-Recall curve
            if hasattr(results, 'box') and hasattr(results.box, 'p') and hasattr(results.box, 'r'):
                self._plot_precision_recall_curve(results, plots_dir)
            
            # Confusion matrix
            if hasattr(results, 'confusion_matrix') or hasattr(results.box, 'confusion_matrix'):
                self._plot_confusion_matrix(results, plots_dir)
            
            # Class distribution plot
            self._plot_class_distribution(plots_dir)
            
            print(f"📊 Plots saved to: {plots_dir}")
            
        except Exception as e:
            print(f"Warning: Could not generate some plots: {e}")
    
    def _plot_precision_recall_curve(self, results, plots_dir: Path):
        """Plot Precision-Recall curve."""
        try:
            plt.figure(figsize=(10, 8))
            
            # Extract precision and recall arrays
            if hasattr(results.box, 'p') and hasattr(results.box, 'r'):
                precision = results.box.p
                recall = results.box.r
                
                # Plot curves for each class
                for i, class_name in self.class_names.items():
                    if i < len(precision):
                        plt.plot(recall[i], precision[i], label=f'{class_name}')
                
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curves by Class')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.xlim([0, 1])
                plt.ylim([0, 1])
                
                plt.tight_layout()
                plt.savefig(plots_dir / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            print(f"Warning: Could not generate PR curve: {e}")
    
    def _plot_confusion_matrix(self, results, plots_dir: Path):
        """Plot confusion matrix."""
        try:
            if hasattr(results, 'confusion_matrix'):
                cm = results.confusion_matrix.matrix
            elif hasattr(results.box, 'confusion_matrix'):
                cm = results.box.confusion_matrix
            else:
                return
            
            plt.figure(figsize=(10, 8))
            
            # Create class labels
            labels = list(self.class_names.values()) + ['background']
            
            # Plot heatmap
            sns.heatmap(cm, 
                       annot=True, 
                       fmt='d', 
                       cmap='Blues',
                       xticklabels=labels,
                       yticklabels=labels)
            
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            plt.savefig(plots_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Warning: Could not generate confusion matrix: {e}")
    
    def _plot_class_distribution(self, plots_dir: Path):
        """Plot class distribution."""
        try:
            plt.figure(figsize=(12, 6))
            
            class_names = list(self.class_names.values())
            class_counts = [1] * len(class_names)  # Placeholder
            
            plt.bar(class_names, class_counts)
            plt.title('Class Distribution in Dataset')
            plt.xlabel('Vehicle Classes')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(plots_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Warning: Could not generate class distribution plot: {e}")
    
    def evaluate_single_image(self, 
                            image_path: str, 
                            save_result: bool = True,
                            save_dir: str = './results') -> Dict:
        """
        Evaluate model on a single image and return detailed results.
        
        Args:
            image_path (str): Path to image file
            save_result (bool): Whether to save visualization
            save_dir (str): Directory to save results
            
        Returns:
            Dict: Detection results with bounding boxes, confidence scores, and classes
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Run inference
        results = self.model(
            str(image_path),
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            imgsz=self.img_size,
            device=self.device
        )
        
        # Extract results
        detections = {
            'image_path': str(image_path),
            'detections': [],
            'num_detections': 0
        }
        
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                
                for i in range(len(boxes)):
                    detection = {
                        'bbox': boxes[i].tolist(),
                        'confidence': float(scores[i]),
                        'class_id': int(classes[i]),
                        'class_name': self.class_names.get(classes[i], f'class_{classes[i]}')
                    }
                    detections['detections'].append(detection)
                
                detections['num_detections'] = len(boxes)
        
        # Save visualization if requested
        if save_result:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save annotated image
            if results and len(results) > 0:
                annotated_img = results[0].plot()
                cv2.imwrite(str(save_dir / f'{image_path.stem}_detected.jpg'), annotated_img)
        
        return detections
    
    def benchmark_inference_speed(self, 
                                 test_images: List[str], 
                                 warmup_runs: int = 10,
                                 benchmark_runs: int = 100) -> Dict[str, float]:
        """
        Benchmark inference speed on test images.
        
        Args:
            test_images (List[str]): List of test image paths
            warmup_runs (int): Number of warmup inference runs
            benchmark_runs (int): Number of benchmark runs
            
        Returns:
            Dict[str, float]: Speed benchmarking results
        """
        print(f"🏁 Benchmarking inference speed...")
        
        if not test_images:
            raise ValueError("No test images provided")
        
        # Warmup
        print(f"Warming up with {warmup_runs} runs...")
        for i in range(warmup_runs):
            img_path = test_images[i % len(test_images)]
            _ = self.model(img_path, conf=self.conf_thresh, iou=self.iou_thresh, verbose=False)
        
        # Benchmark
        print(f"Running {benchmark_runs} benchmark iterations...")
        times = []
        
        for i in range(benchmark_runs):
            img_path = test_images[i % len(test_images)]
            
            start_time = time.time()
            _ = self.model(img_path, conf=self.conf_thresh, iou=self.iou_thresh, verbose=False)
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        # Calculate statistics
        times = np.array(times)
        results = {
            'mean_inference_time': float(np.mean(times)),
            'std_inference_time': float(np.std(times)),
            'min_inference_time': float(np.min(times)),
            'max_inference_time': float(np.max(times)),
            'fps': float(1.0 / np.mean(times)),
            'total_benchmark_time': float(np.sum(times))
        }
        
        print(f"⚡ Benchmark Results:")
        print(f"   Mean inference time: {results['mean_inference_time']*1000:.2f} ms")
        print(f"   FPS: {results['fps']:.2f}")
        print(f"   Min/Max time: {results['min_inference_time']*1000:.2f}/{results['max_inference_time']*1000:.2f} ms")
        
        return results


def evaluate_model(model_path: str,
                  dataset_path: str,
                  save_dir: str = './results',
                  conf_thresh: float = 0.25,
                  iou_thresh: float = 0.45,
                  device: str = 'auto') -> Dict[str, float]:
    """
    Main function to evaluate a trained aerial vehicle detection model.
    
    Args:
        model_path (str): Path to trained model file
        dataset_path (str): Path to dataset directory
        save_dir (str): Directory to save evaluation results
        conf_thresh (float): Confidence threshold
        iou_thresh (float): IoU threshold
        device (str): Device to use for evaluation
        
    Returns:
        Dict[str, float]: Evaluation metrics
    """
    evaluator = ModelEvaluator(
        model_path=model_path,
        device=device,
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh
    )
    
    return evaluator.evaluate_dataset(dataset_path, save_dir)


if __name__ == "__main__":
    # Example usage
    model_path = "./models/best.pt"
    dataset_path = "./data/aerial_vehicles"
    
    try:
        # Evaluate model
        metrics = evaluate_model(
            model_path=model_path,
            dataset_path=dataset_path,
            save_dir="./results/evaluation",
            conf_thresh=0.25,
            iou_thresh=0.45
        )
        
        print("✅ Evaluation completed!")
        print(f"📊 Key metrics:")
        print(f"   mAP@50: {metrics.get('map50', 0):.3f}")
        print(f"   mAP@50-95: {metrics.get('map', 0):.3f}")
        print(f"   Precision: {metrics.get('precision', 0):.3f}")
        print(f"   Recall: {metrics.get('recall', 0):.3f}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()