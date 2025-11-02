"""
Inference Engine for Aerial Vehicle Detection

This module handles real-time inference for aerial vehicle detection using trained YOLOv8 models.
It provides functionality for:
- Single image detection with visualization
- Video processing with frame-by-frame detection
- Directory batch processing
- Real-time display and result saving
- Custom visualization with bounding boxes and labels
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torch
from PIL import Image, ImageDraw, ImageFont


class InferenceEngine:
    """
    Real-time inference engine for aerial vehicle detection.
    Handles various input formats and provides comprehensive visualization options.
    """
    
    def __init__(self,
                 model_path: str,
                 device: str = 'auto',
                 conf_thresh: float = 0.25,
                 iou_thresh: float = 0.45,
                 img_size: int = 640):
        """
        Initialize the inference engine.
        
        Args:
            model_path (str): Path to trained model file
            device (str): Device to use ('auto', 'cpu', 'cuda')
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
        
        # Color palette for visualization
        self.colors = self._generate_colors()
        
        print(f"🚀 Inference engine initialized")
        print(f"📁 Model: {model_path}")
        print(f"🎯 Confidence threshold: {conf_thresh}")
        print(f"🔗 IoU threshold: {iou_thresh}")
        print(f"📊 Classes: {list(self.class_names.values())}")
    
    def _load_model(self) -> YOLO:
        """Load the trained YOLO model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        model = YOLO(str(self.model_path))
        model.to(self.device)
        return model
    
    def _generate_colors(self) -> List[Tuple[int, int, int]]:
        """Generate distinct colors for each class."""
        np.random.seed(42)  # For consistent colors
        colors = []
        for i in range(self.num_classes):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            colors.append(color)
        return colors
    
    def detect_from_image(self, 
                         image_path: str, 
                         output_dir: str = './results',
                         save_txt: bool = False,
                         save_conf: bool = False,
                         show: bool = False) -> List[Dict]:
        """
        Perform detection on a single image.
        
        Args:
            image_path (str): Path to input image
            output_dir (str): Directory to save results
            save_txt (bool): Save results in YOLO format text files
            save_conf (bool): Include confidence scores in saved results
            show (bool): Display results in real-time
            
        Returns:
            List[Dict]: List of detections with bounding boxes and metadata
        """
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        print(f"🖼️  Processing image: {image_path.name}")
        
        # Load and process image
        start_time = time.time()
        results = self.model(
            str(image_path),
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            imgsz=self.img_size,
            device=self.device,
            verbose=False
        )
        inference_time = time.time() - start_time
        
        # Extract detections
        detections = []
        if results and len(results) > 0:
            result = results[0]
            detections = self._extract_detections(result, inference_time)
        
        print(f"✅ Found {len(detections)} objects in {inference_time*1000:.2f}ms")
        
        # Load original image for visualization
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create visualization
        annotated_image = self._draw_detections(image, detections)
        
        # Save results
        output_name = image_path.stem
        
        # Save annotated image
        annotated_path = output_dir / f"{output_name}_detected.jpg"
        plt.figure(figsize=(12, 8))
        plt.imshow(annotated_image)
        plt.axis('off')
        plt.title(f'Aerial Vehicle Detection - {len(detections)} objects detected')
        plt.tight_layout()
        plt.savefig(annotated_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        if show:
            plt.show()
        else:
            plt.close()
        
        # Save text results if requested
        if save_txt:
            self._save_detections_txt(detections, output_dir / f"{output_name}.txt", 
                                    image.shape, save_conf)
        
        # Save JSON results
        self._save_detections_json(detections, output_dir / f"{output_name}.json", 
                                 str(image_path), inference_time)
        
        print(f"💾 Results saved to: {output_dir}")
        
        return detections
    
    def detect_from_video(self, 
                         video_path: str, 
                         output_dir: str = './results',
                         save_txt: bool = False,
                         save_conf: bool = False,
                         show: bool = False,
                         save_video: bool = True) -> Dict:
        """
        Perform detection on a video file.
        
        Args:
            video_path (str): Path to input video
            output_dir (str): Directory to save results
            save_txt (bool): Save frame-by-frame results in text files
            save_conf (bool): Include confidence scores
            show (bool): Display results in real-time
            save_video (bool): Save annotated video
            
        Returns:
            Dict: Video processing statistics
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        print(f"🎥 Processing video: {video_path.name}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📊 Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup video writer if saving
        video_writer = None
        if save_video:
            output_video_path = output_dir / f"{video_path.stem}_detected.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        
        # Process frames
        frame_count = 0
        total_detections = 0
        processing_times = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Convert BGR to RGB for processing
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Run detection
                start_time = time.time()
                results = self.model(
                    frame_rgb,
                    conf=self.conf_thresh,
                    iou=self.iou_thresh,
                    imgsz=self.img_size,
                    device=self.device,
                    verbose=False
                )
                inference_time = time.time() - start_time
                processing_times.append(inference_time)
                
                # Extract detections
                detections = []
                if results and len(results) > 0:
                    result = results[0]
                    detections = self._extract_detections(result, inference_time)
                
                total_detections += len(detections)
                
                # Create annotated frame
                annotated_frame = self._draw_detections(frame_rgb, detections)
                annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                
                # Add frame info
                info_text = f"Frame: {frame_count}/{total_frames} | Objects: {len(detections)} | FPS: {1/inference_time:.1f}"
                cv2.putText(annotated_frame_bgr, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Save frame if requested
                if save_video and video_writer:
                    video_writer.write(annotated_frame_bgr)
                
                # Display frame if requested
                if show:
                    cv2.imshow('Aerial Vehicle Detection', annotated_frame_bgr)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Save text results if requested
                if save_txt:
                    txt_path = output_dir / f"frame_{frame_count:06d}.txt"
                    self._save_detections_txt(detections, txt_path, frame_rgb.shape, save_conf)
                
                # Progress update
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    avg_fps = 1 / np.mean(processing_times[-30:])
                    print(f"Progress: {progress:.1f}% | Avg FPS: {avg_fps:.1f}")
        
        finally:
            cap.release()
            if video_writer:
                video_writer.release()
            if show:
                cv2.destroyAllWindows()
        
        # Calculate statistics
        avg_inference_time = np.mean(processing_times)
        avg_fps = 1 / avg_inference_time
        
        stats = {
            'total_frames': frame_count,
            'total_detections': total_detections,
            'avg_detections_per_frame': total_detections / max(frame_count, 1),
            'avg_inference_time': avg_inference_time,
            'avg_fps': avg_fps,
            'video_duration': frame_count / fps,
            'processing_time': sum(processing_times)
        }
        
        print(f"✅ Video processing completed!")
        print(f"📊 Statistics:")
        print(f"   Processed frames: {frame_count}")
        print(f"   Total detections: {total_detections}")
        print(f"   Avg detections/frame: {stats['avg_detections_per_frame']:.2f}")
        print(f"   Avg FPS: {avg_fps:.2f}")
        
        # Save statistics
        import json
        stats_path = output_dir / f"{video_path.stem}_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        return stats
    
    def detect_from_directory(self, 
                             input_dir: str, 
                             output_dir: str = './results',
                             save_txt: bool = False,
                             save_conf: bool = False) -> int:
        """
        Process all images in a directory.
        
        Args:
            input_dir (str): Directory containing input images
            output_dir (str): Directory to save results
            save_txt (bool): Save results in text format
            save_conf (bool): Include confidence scores
            
        Returns:
            int: Total number of detections across all images
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_dir.glob(f'*{ext}'))
            image_files.extend(input_dir.glob(f'*{ext.upper()}'))
        
        if not image_files:
            raise ValueError(f"No image files found in {input_dir}")
        
        print(f"📁 Processing {len(image_files)} images from: {input_dir}")
        
        total_detections = 0
        processing_times = []
        
        for i, image_path in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {image_path.name}")
            
            try:
                detections = self.detect_from_image(
                    str(image_path),
                    output_dir,
                    save_txt=save_txt,
                    save_conf=save_conf,
                    show=False
                )
                total_detections += len(detections)
                
            except Exception as e:
                print(f"Error processing {image_path.name}: {e}")
                continue
        
        print(f"✅ Directory processing completed!")
        print(f"📊 Total detections: {total_detections}")
        print(f"💾 Results saved to: {output_dir}")
        
        return total_detections
    
    def _extract_detections(self, result, inference_time: float) -> List[Dict]:
        """Extract detection results from YOLOv8 output."""
        detections = []
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            
            for i in range(len(boxes)):
                detection = {
                    'bbox': boxes[i].tolist(),  # [x1, y1, x2, y2]
                    'confidence': float(scores[i]),
                    'class_id': int(classes[i]),
                    'class_name': self.class_names.get(classes[i], f'class_{classes[i]}'),
                    'inference_time': inference_time
                }
                detections.append(detection)
        
        return detections
    
    def _draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes and labels on image."""
        annotated_image = image.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            class_id = detection['class_id']
            
            # Get color for this class
            color = self.colors[class_id % len(self.colors)]
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            
            # Create label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Draw label background
            cv2.rectangle(annotated_image, 
                         (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1),
                         color, -1)
            
            # Draw label text
            cv2.putText(annotated_image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_image
    
    def _save_detections_txt(self, 
                           detections: List[Dict], 
                           output_path: Path,
                           image_shape: Tuple[int, int, int],
                           save_conf: bool = False):
        """Save detections in YOLO format text file."""
        height, width = image_shape[:2]
        
        with open(output_path, 'w') as f:
            for detection in detections:
                bbox = detection['bbox']
                class_id = detection['class_id']
                confidence = detection['confidence']
                
                # Convert to YOLO format (normalized center coordinates)
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) / 2 / width
                center_y = (y1 + y2) / 2 / height
                bbox_width = (x2 - x1) / width
                bbox_height = (y2 - y1) / height
                
                if save_conf:
                    f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f} {confidence:.6f}\n")
                else:
                    f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")
    
    def _save_detections_json(self, 
                            detections: List[Dict], 
                            output_path: Path,
                            image_path: str,
                            inference_time: float):
        """Save detections in JSON format."""
        import json
        
        result_data = {
            'image_path': image_path,
            'inference_time': inference_time,
            'num_detections': len(detections),
            'detections': detections,
            'model_config': {
                'conf_thresh': self.conf_thresh,
                'iou_thresh': self.iou_thresh,
                'model_path': str(self.model_path)
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(result_data, f, indent=2)


def detect_from_image(model_path: str,
                     image_path: str,
                     output_dir: str = './results',
                     conf_thresh: float = 0.25,
                     iou_thresh: float = 0.45,
                     device: str = 'auto') -> List[Dict]:
    """
    Main function to perform detection on a single image.
    
    Args:
        model_path (str): Path to trained model
        image_path (str): Path to input image
        output_dir (str): Output directory
        conf_thresh (float): Confidence threshold
        iou_thresh (float): IoU threshold
        device (str): Device to use
        
    Returns:
        List[Dict]: Detection results
    """
    engine = InferenceEngine(
        model_path=model_path,
        device=device,
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh
    )
    
    return engine.detect_from_image(image_path, output_dir)


def detect_from_video(model_path: str,
                     video_path: str,
                     output_dir: str = './results',
                     conf_thresh: float = 0.25,
                     iou_thresh: float = 0.45,
                     device: str = 'auto') -> Dict:
    """
    Main function to perform detection on a video.
    
    Args:
        model_path (str): Path to trained model
        video_path (str): Path to input video
        output_dir (str): Output directory
        conf_thresh (float): Confidence threshold
        iou_thresh (float): IoU threshold
        device (str): Device to use
        
    Returns:
        Dict: Processing statistics
    """
    engine = InferenceEngine(
        model_path=model_path,
        device=device,
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh
    )
    
    return engine.detect_from_video(video_path, output_dir)


if __name__ == "__main__":
    # Example usage
    model_path = "./models/best.pt"
    
    try:
        # Test single image detection
        image_path = "./test_image.jpg"
        if os.path.exists(image_path):
            detections = detect_from_image(
                model_path=model_path,
                image_path=image_path,
                conf_thresh=0.25
            )
            print(f"Detected {len(detections)} vehicles")
        
    except Exception as e:
        print(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()