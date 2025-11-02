#!/usr/bin/env python3
"""
Smart Traffic Monitoring System - Main Launcher
Launch script for the complete traffic monitoring system
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse

def main():
    """Main launcher for the traffic monitoring system"""
    print("🚦 Smart Traffic Monitoring System")
    print("=" * 50)
    
    parser = argparse.ArgumentParser(description='Smart Traffic Monitoring System')
    parser.add_argument('--mode', choices=['dashboard', 'process-image', 'process-video', 'test'], 
                       default='dashboard', help='Operation mode')
    parser.add_argument('--input', help='Input file path (for image/video processing)')
    parser.add_argument('--confidence', type=float, default=0.15, help='Confidence threshold')
    parser.add_argument('--output', help='Output directory', default='traffic_monitoring_system/results')
    
    args = parser.parse_args()
    
    # Ensure we're in the right directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    if args.mode == 'dashboard':
        print("🌐 Starting web dashboard...")
        print("📱 Open your browser and go to: http://localhost:8050")
        print("🛑 Press Ctrl+C to stop")
        
        try:
            subprocess.run([sys.executable, 'traffic_monitoring_system/dashboard.py'])
        except KeyboardInterrupt:
            print("\\n🛑 Dashboard stopped by user")
    
    elif args.mode == 'process-image':
        if not args.input:
            print("❌ Please provide --input path for image processing")
            return
        
        print(f"📸 Processing image: {args.input}")
        
        # Import and run image processing
        sys.path.append('.')
        from traffic_monitoring_system.realtime_processor import RealTimeTrafficProcessor
        
        processor = RealTimeTrafficProcessor(confidence_threshold=args.confidence)
        if processor.load_model():
            # Setup default zones
            zones = {
                "Detection_Zone": [(100, 100), (500, 100), (500, 400), (100, 400)]
            }
            processor.setup_monitoring_zones(zones)
            
            # Process image
            results = processor.process_single_image(args.input)
            
            if 'error' not in results:
                print(f"✅ Processing complete!")
                print(f"   Vehicles detected: {len(results['detections'])}")
                print(f"   Processing time: {results['processing_time']}s")
                print(f"   Congestion level: {results['analytics']['congestion']['overall_level']}")
                
                # Save results
                output_dir = Path(args.output)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                import json
                results_file = output_dir / f"analysis_{Path(args.input).stem}.json"
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)
                
                print(f"   Results saved to: {results_file}")
            else:
                print(f"❌ Error: {results['error']}")
    
    elif args.mode == 'process-video':
        if not args.input:
            print("❌ Please provide --input path for video processing")
            return
        
        print(f"🎬 Processing video: {args.input}")
        
        sys.path.append('.')
        from traffic_monitoring_system.realtime_processor import RealTimeTrafficProcessor
        
        processor = RealTimeTrafficProcessor(confidence_threshold=args.confidence)
        if processor.load_model():
            zones = {
                "Traffic_Zone": [(150, 150), (450, 150), (450, 350), (150, 350)]
            }
            processor.setup_monitoring_zones(zones)
            
            results = processor.process_video_file(args.input, args.output)
            
            if 'error' not in results:
                print(f"✅ Video processing complete!")
                print(f"   Frames processed: {results['frames_processed']}")
                print(f"   Total vehicles: {results['total_vehicles_detected']}")
                print(f"   Peak traffic: {results['peak_traffic']}")
            else:
                print(f"❌ Error: {results['error']}")
    
    elif args.mode == 'test':
        print("🧪 Running system tests...")
        
        # Test RF-DETR detector
        print("\\n1. Testing RF-DETR detector...")
        try:
            subprocess.run([sys.executable, 'rfdetr_detector.py'], check=True)
            print("✅ RF-DETR detector working")
        except:
            print("❌ RF-DETR detector failed")
            return
        
        # Test traffic analytics
        print("\\n2. Testing traffic analytics...")
        try:
            subprocess.run([sys.executable, 'traffic_monitoring_system/traffic_analytics.py'], check=True)
            print("✅ Traffic analytics working")
        except:
            print("❌ Traffic analytics failed")
            return
        
        # Test real-time processor
        print("\\n3. Testing real-time processor...")
        try:
            subprocess.run([sys.executable, 'traffic_monitoring_system/realtime_processor.py'], check=True)
            print("✅ Real-time processor working")
        except:
            print("❌ Real-time processor failed")
            return
        
        print("\\n✅ All system tests passed!")
        print("\\n🚀 Your Smart Traffic Monitoring System is ready!")
        print("\\nNext steps:")
        print("1. Run: python traffic_monitoring_launcher.py --mode dashboard")
        print("2. Open browser to: http://localhost:8050")
        print("3. Start monitoring traffic!")

if __name__ == "__main__":
    main()