#!/usr/bin/env python3
"""
Real-time Traffic Processor
Integrates RF-DETR model with traffic analytics for live monitoring
"""

import cv2
import numpy as np
from datetime import datetime
import threading
import time
from pathlib import Path
from typing import Optional, Callable, Dict, Any
from queue import Queue
import sys
import os

# Add parent directory to path to import RF-DETR detector
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from rfdetr_detector import RFDETRAerialDetector
from traffic_analytics import TrafficAnalytics, TrafficZone


class RealTimeTrafficProcessor:
    """Real-time traffic monitoring and processing system"""
    
    def __init__(self, 
                 confidence_threshold: float = 0.15,
                 processing_interval: float = 1.0,
                 analytics_db_path: str = "traffic_monitoring_system/data/traffic_data.db"):
        """
        Initialize the real-time traffic processor
        
        Args:
            confidence_threshold: Minimum confidence for vehicle detection
            processing_interval: Seconds between processing frames
            analytics_db_path: Path to analytics database
        """
        
        self.confidence_threshold = confidence_threshold
        self.processing_interval = processing_interval
        
        # Initialize RF-DETR detector
        print("🚁 Initializing RF-DETR Vehicle Detector...")
        self.detector = RFDETRAerialDetector(confidence_threshold=confidence_threshold)
        
        # Initialize traffic analytics
        print("🚦 Initializing Traffic Analytics...")
        self.analytics = TrafficAnalytics(db_path=analytics_db_path)
        
        # Processing state
        self.is_processing = False
        self.current_frame = None
        self.latest_results = None
        self.frame_queue = Queue(maxsize=10)
        self.results_callback: Optional[Callable] = None
        
        # Performance tracking
        self.processing_times = []
        self.frames_processed = 0
        
        print("✅ Real-time Traffic Processor Initialized")
    
    def load_model(self) -> bool:
        """Load the RF-DETR model"""
        return self.detector.load_model()
    
    def setup_monitoring_zones(self, zones_config: Dict[str, list]):
        """
        Setup traffic monitoring zones
        
        Args:
            zones_config: Dict with zone names and polygon coordinates
                         Example: {"intersection_1": [(100,100), (300,100), (300,300), (100,300)]}
        """
        print(f"🎯 Setting up {len(zones_config)} monitoring zones...")
        
        for zone_name, polygon in zones_config.items():
            self.analytics.add_zone(zone_name, polygon)
        
        print("✅ Monitoring zones configured")
    
    def set_results_callback(self, callback: Callable[[Dict], None]):
        """Set callback function to receive processing results"""
        self.results_callback = callback
    
    def process_single_image(self, image_path: str) -> Dict:
        """
        Process a single image and return traffic analytics
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dict: Complete analysis results
        """
        try:
            start_time = time.time()
            
            # Detect vehicles using RF-DETR
            print(f"🔍 Processing image: {Path(image_path).name}")
            detections = self.detector.detect_and_format(image_path)
            
            # Run traffic analytics
            analytics_results = self.analytics.process_detections(detections)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # Compile complete results
            results = {
                'image_path': image_path,
                'processing_time': round(processing_time, 3),
                'detections': detections,
                'analytics': analytics_results,
                'zones': self.analytics.get_zone_analytics(),
                'performance': {
                    'avg_processing_time': round(np.mean(self.processing_times[-10:]), 3),
                    'frames_processed': len(self.processing_times)
                }
            }
            
            print(f"✅ Processed in {processing_time:.2f}s - Found {len(detections)} vehicles")
            
            return results
            
        except Exception as e:
            print(f"❌ Error processing image: {e}")
            return {'error': str(e)}
    
    def process_video_file(self, video_path: str, output_dir: str = "traffic_monitoring_system/data/video_results") -> Dict:
        """
        Process video file and generate traffic analytics
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save results
            
        Returns:
            Dict: Processing summary
        """
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {'error': f'Could not open video file: {video_path}'}
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            
            print(f"🎬 Processing video: {Path(video_path).name}")
            print(f"   Duration: {duration:.1f}s, FPS: {fps:.1f}, Frames: {total_frames}")
            
            # Process every N frames based on processing interval
            frame_skip = max(1, int(fps * self.processing_interval))
            frames_to_process = list(range(0, total_frames, frame_skip))
            
            results_summary = {
                'video_path': video_path,
                'total_frames': total_frames,
                'frames_processed': 0,
                'processing_errors': 0,
                'total_vehicles_detected': 0,
                'peak_traffic': 0,
                'congestion_events': 0,
                'processing_start': datetime.now().isoformat(),
                'frame_results': []
            }
            
            for frame_num in frames_to_process:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                try:
                    # Save frame temporarily
                    temp_frame_path = f"{output_dir}/temp_frame_{frame_num}.jpg"
                    cv2.imwrite(temp_frame_path, frame)
                    
                    # Process frame
                    frame_results = self.process_single_image(temp_frame_path)
                    
                    # Update summary
                    if 'error' not in frame_results:
                        results_summary['frames_processed'] += 1
                        results_summary['total_vehicles_detected'] += len(frame_results['detections'])
                        results_summary['peak_traffic'] = max(
                            results_summary['peak_traffic'], 
                            frame_results['analytics']['total_vehicles']
                        )
                        
                        if frame_results['analytics']['congestion']['detected']:
                            results_summary['congestion_events'] += 1
                        
                        # Store key frame data
                        results_summary['frame_results'].append({
                            'frame_number': frame_num,
                            'timestamp': frame_num / fps,
                            'vehicles': len(frame_results['detections']),
                            'congestion': frame_results['analytics']['congestion']['overall_level']
                        })
                    else:
                        results_summary['processing_errors'] += 1
                    
                    # Cleanup temp frame
                    if Path(temp_frame_path).exists():
                        Path(temp_frame_path).unlink()
                    
                    # Progress update
                    progress = (frame_num / total_frames) * 100
                    print(f"   Progress: {progress:.1f}% - Frame {frame_num}/{total_frames}")
                    
                except Exception as e:
                    print(f"❌ Error processing frame {frame_num}: {e}")
                    results_summary['processing_errors'] += 1
            
            cap.release()
            
            # Finalize summary
            results_summary['processing_end'] = datetime.now().isoformat()
            results_summary['success_rate'] = (
                results_summary['frames_processed'] / 
                len(frames_to_process) * 100
            ) if frames_to_process else 0
            
            # Save results
            results_file = f"{output_dir}/video_analysis_{Path(video_path).stem}.json"
            import json
            with open(results_file, 'w') as f:
                json.dump(results_summary, f, indent=2)
            
            print(f"\\n📊 Video Processing Complete:")
            print(f"   Frames processed: {results_summary['frames_processed']}")
            print(f"   Total vehicles detected: {results_summary['total_vehicles_detected']}")
            print(f"   Peak traffic: {results_summary['peak_traffic']} vehicles")
            print(f"   Congestion events: {results_summary['congestion_events']}")
            print(f"   Results saved to: {results_file}")
            
            return results_summary
            
        except Exception as e:
            print(f"❌ Error processing video: {e}")
            return {'error': str(e)}
    
    def start_live_monitoring(self, source: Any = 0):
        """
        Start live traffic monitoring from camera/stream
        
        Args:
            source: Video source (0 for webcam, URL for IP camera, etc.)
        """
        if self.is_processing:
            print("⚠️ Live monitoring already running")
            return
        
        print(f"📹 Starting live traffic monitoring from source: {source}")
        
        try:
            # Try different backends for camera access
            cap = None
            if source == 0:  # Webcam
                backends = [
                    (cv2.CAP_DSHOW, "DirectShow"),
                    (cv2.CAP_MSMF, "Media Foundation"),
                    (cv2.CAP_ANY, "Default")
                ]
                
                for backend_id, backend_name in backends:
                    print(f"   Trying {backend_name} backend...")
                    cap = cv2.VideoCapture(source, backend_id)
                    if cap.isOpened():
                        ret, test_frame = cap.read()
                        if ret:
                            print(f"✅ Successfully connected using {backend_name}")
                            break
                        else:
                            cap.release()
                            cap = None
                    else:
                        if cap:
                            cap.release()
                        cap = None
            else:
                # For video files, use default
                cap = cv2.VideoCapture(source)
            
            if not cap or not cap.isOpened():
                print(f"❌ Could not open video source: {source}")
                if source == 0:
                    print("   No camera accessible with any backend")
                    print("   Try running as Administrator or check camera permissions")
                return
            
            # Test frame read
            ret, test_frame = cap.read()
            if not ret:
                print(f"❌ Could not read from video source: {source}")
                cap.release()
                return
            
            print(f"✅ Successfully connected to camera - Frame size: {test_frame.shape[:2]}")
            
            self.is_processing = True
            last_process_time = 0
            
            while self.is_processing:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to read frame")
                    break
                
                self.current_frame = frame.copy()
                
                # Process frame at specified interval
                current_time = time.time()
                if current_time - last_process_time >= self.processing_interval:
                    
                    # Process in background thread to maintain frame rate
                    threading.Thread(
                        target=self._process_frame_async,
                        args=(frame.copy(),),
                        daemon=True
                    ).start()
                    
                    last_process_time = current_time
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.01)
            
            cap.release()
            print("📹 Live monitoring stopped")
            
        except Exception as e:
            print(f"❌ Error in live monitoring: {e}")
        finally:
            self.is_processing = False
    
    def _process_frame_async(self, frame: np.ndarray):
        """Process frame asynchronously"""
        try:
            # Save frame temporarily
            temp_path = "traffic_monitoring_system/data/temp_live_frame.jpg"
            Path(temp_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(temp_path, frame)
            
            # Process frame
            results = self.process_single_image(temp_path)
            
            # Update latest results
            self.latest_results = results
            
            # Call callback if set
            if self.results_callback:
                self.results_callback(results)
            
            # Cleanup
            if Path(temp_path).exists():
                Path(temp_path).unlink()
                
        except Exception as e:
            print(f"❌ Error in async frame processing: {e}")
    
    def stop_live_monitoring(self):
        """Stop live monitoring"""
        self.is_processing = False
        print("🛑 Stopping live monitoring...")
    
    def get_current_status(self) -> Dict:
        """Get current monitoring status"""
        return {
            'is_processing': self.is_processing,
            'frames_processed': len(self.processing_times),
            'avg_processing_time': round(np.mean(self.processing_times[-10:]), 3) if self.processing_times else 0,
            'latest_results': self.latest_results,
            'zones_configured': len(self.analytics.zones),
            'current_time': datetime.now().isoformat()
        }


def main():
    """Test the real-time traffic processor"""
    print("🚦 Testing Real-time Traffic Processor")
    print("=" * 50)
    
    # Initialize processor
    processor = RealTimeTrafficProcessor()
    
    # Load model
    if not processor.load_model():
        print("❌ Failed to load model")
        return
    
    # Setup monitoring zones
    zones_config = {
        "Main_Intersection": [(200, 200), (400, 200), (400, 400), (200, 400)],
        "Highway_Entry": [(500, 300), (700, 300), (700, 400), (500, 400)]
    }
    processor.setup_monitoring_zones(zones_config)
    
    # Test with sample images if available
    test_images_dir = Path("../dataset_yolo/test/images")
    if test_images_dir.exists():
        test_images = list(test_images_dir.glob("*.jpg"))[:3]
        
        for image_path in test_images:
            print(f"\\n🧪 Testing with: {image_path.name}")
            results = processor.process_single_image(str(image_path))
            
            if 'error' not in results:
                print(f"   Vehicles detected: {len(results['detections'])}")
                print(f"   Processing time: {results['processing_time']}s")
                print(f"   Congestion: {results['analytics']['congestion']['overall_level']}")
    
    print("\\n✅ Real-time Traffic Processor Ready!")
    print("\\n📋 Usage Examples:")
    print("   # Process single image")
    print("   processor.process_single_image('traffic_image.jpg')")
    print("   ")
    print("   # Process video file")
    print("   processor.process_video_file('traffic_video.mp4')")
    print("   ")
    print("   # Start live monitoring")
    print("   processor.start_live_monitoring(0)  # Webcam")


if __name__ == "__main__":
    main()