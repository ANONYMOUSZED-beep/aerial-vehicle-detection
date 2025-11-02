#!/usr/bin/env python3
"""
RF-DETR Aerial Vehicle Detection System
Using the rebotnix/rb_vehicle model with the rfdetr library
"""

import os
import torch
import supervision as sv
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from rfdetr import RFDETRBase
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt
import cv2


class RFDETRAerialDetector:
    """
    RF-DETR based aerial vehicle detector using rebotnix/rb_vehicle model
    """
    
    def __init__(self, confidence_threshold: float = 0.15, device: str = 'auto'):
        """
        Initialize the RF-DETR aerial vehicle detector
        
        Args:
            confidence_threshold: Minimum confidence for detections
            device: Device to run inference on ('auto', 'cpu', 'cuda')
        """
        self.confidence_threshold = confidence_threshold
        self.device = self._setup_device(device)
        self.model = None
        self.model_path = None
        
        # Vehicle class names (based on rebotnix model)
        self.class_names = ["vehicle"]  # Single class model
        
        print(f"🚁 Initializing RF-DETR Aerial Vehicle Detector")
        print(f"   Confidence Threshold: {confidence_threshold}")
        print(f"   Device: {self.device}")
    
    def _setup_device(self, device: str) -> str:
        """Setup computation device"""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def load_model(self):
        """Load the RF-DETR model from Hugging Face"""
        try:
            print("📥 Downloading RF-DETR model from Hugging Face...")
            
            # Download model weights
            self.model_path = hf_hub_download(
                repo_id="rebotnix/rb_vehicle",
                filename="rb_vehicle.pth"
            )
            
            print(f"   Model downloaded to: {self.model_path}")
            print("   Loading RF-DETR model...")
            
            # Initialize RF-DETR model
            self.model = RFDETRBase(
                pretrain_weights=self.model_path,
                num_classes=len(self.class_names)
            )
            
            print("✅ RF-DETR model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading RF-DETR model: {e}")
            return False
    
    def detect_objects(self, image_input: Union[str, Path, Image.Image], 
                      confidence_threshold: float = None) -> sv.Detections:
        """
        Detect vehicles in aerial imagery
        
        Args:
            image_input: Image path or PIL Image
            confidence_threshold: Override default confidence threshold
            
        Returns:
            supervision.Detections: Detection results
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
        
        # Load image
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input).convert('RGB')
        else:
            image = image_input.convert('RGB') if image_input.mode != 'RGB' else image_input
        
        # Run detection
        detections = self.model.predict(image, threshold=confidence_threshold)
        
        return detections
    
    def detect_and_format(self, image_input: Union[str, Path, Image.Image], 
                         confidence_threshold: float = None) -> List[Dict]:
        """
        Detect vehicles and return formatted results
        
        Args:
            image_input: Image path or PIL Image  
            confidence_threshold: Override default confidence threshold
            
        Returns:
            List[Dict]: Formatted detection results
        """
        detections = self.detect_objects(image_input, confidence_threshold)
        
        # Format results
        results = []
        if len(detections) > 0:
            for i in range(len(detections)):
                bbox = detections.xyxy[i]  # [x1, y1, x2, y2]
                confidence = detections.confidence[i]
                class_id = detections.class_id[i] if detections.class_id is not None else 0
                
                result = {
                    'bbox': bbox.tolist(),
                    'confidence': float(confidence),
                    'class_id': int(class_id),
                    'class_name': self.class_names[class_id]
                }
                results.append(result)
        
        return results
    
    def visualize_detections(self, image_input: Union[str, Path, Image.Image], 
                           detections: sv.Detections = None,
                           output_path: Optional[str] = None) -> Image.Image:
        """
        Visualize detections on the image
        
        Args:
            image_input: Image path or PIL Image
            detections: Detection results (if None, will run detection)
            output_path: Path to save annotated image
            
        Returns:
            PIL.Image: Annotated image
        """
        # Load image
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input).convert('RGB')
        else:
            image = image_input.convert('RGB') if image_input.mode != 'RGB' else image_input
        
        # Get detections if not provided
        if detections is None:
            detections = self.detect_objects(image)
        
        # Create labels
        labels = []
        if len(detections) > 0:
            labels = [
                f"{self.class_names[0]} {confidence:.2f}"
                for confidence in detections.confidence
            ]
        
        # Annotate image using supervision
        annotated_image = image.copy()
        if len(detections) > 0:
            annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
            annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)
        
        # Save if output path provided
        if output_path:
            annotated_image.save(output_path)
            print(f"   Saved annotated image to: {output_path}")
        
        return annotated_image
    
    def process_directory(self, input_dir: Union[str, Path], 
                         output_dir: Union[str, Path],
                         confidence_threshold: float = None) -> Dict:
        """
        Process all images in a directory
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save results  
            confidence_threshold: Override default confidence threshold
            
        Returns:
            Dict: Processing statistics
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
        
        # Find image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        print(f"🔍 Processing {len(image_files)} images from {input_dir}")
        print(f"   Confidence threshold: {confidence_threshold}")
        
        stats = {
            'total_images': len(image_files),
            'processed_images': 0,
            'total_detections': 0,
            'failed_images': 0
        }
        
        for image_file in image_files:
            try:
                # Detect vehicles
                detections = self.detect_objects(image_file, confidence_threshold)
                num_detections = len(detections)
                
                # Save annotated image
                output_file = output_path / f"detected_{image_file.name}"
                self.visualize_detections(image_file, detections, output_file)
                
                stats['processed_images'] += 1
                stats['total_detections'] += num_detections
                
                print(f"✅ {image_file.name}: {num_detections} vehicles detected")
                
            except Exception as e:
                print(f"❌ Failed to process {image_file.name}: {e}")
                stats['failed_images'] += 1
        
        print(f"\\n📊 Processing Summary:")
        print(f"   Total images: {stats['total_images']}")
        print(f"   Successfully processed: {stats['processed_images']}")
        print(f"   Total detections: {stats['total_detections']}")
        print(f"   Failed: {stats['failed_images']}")
        
        return stats


def main():
    """Test the RF-DETR detector"""
    print("🚁 RF-DETR Aerial Vehicle Detection System")
    print("=" * 50)
    
    try:
        # Initialize detector
        detector = RFDETRAerialDetector(confidence_threshold=0.15)
        
        # Load model
        if detector.load_model():
            print("\\n✅ RF-DETR system ready!")
            
            # Test with sample images if available
            test_images_dir = Path("dataset_yolo/test/images")
            if test_images_dir.exists():
                print(f"\\n🧪 Testing with sample images from {test_images_dir}")
                
                # Process a few test images
                image_files = list(test_images_dir.glob("*.jpg"))[:3]
                if image_files:
                    output_dir = Path("detection_results")
                    output_dir.mkdir(exist_ok=True)
                    
                    for image_file in image_files:
                        print(f"\\n   Processing: {image_file.name}")
                        
                        # Run detection
                        results = detector.detect_and_format(image_file)
                        
                        if results:
                            print(f"   Found {len(results)} vehicles:")
                            for result in results[:5]:  # Show first 5
                                print(f"     - {result['class_name']}: {result['confidence']:.3f}")
                        else:
                            print("   No vehicles detected")
                        
                        # Save annotated image
                        output_file = output_dir / f"detected_{image_file.name}"
                        detector.visualize_detections(image_file, output_path=output_file)
                else:
                    print("   No test images found")
            else:
                print(f"\\n📁 Test directory not found: {test_images_dir}")
            
            print("\\n📋 Usage Examples:")
            print("   # Single image detection")
            print("   results = detector.detect_and_format('image.jpg')")
            print("   detector.visualize_detections('image.jpg', output_path='result.jpg')")
            print("   ")
            print("   # Process directory")
            print("   detector.process_directory('input_dir', 'output_dir')")
            
        else:
            print("\\n❌ Failed to load model")
            
    except Exception as e:
        print(f"\\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()