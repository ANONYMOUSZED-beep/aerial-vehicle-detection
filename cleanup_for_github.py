#!/usr/bin/env python3
"""
GitHub Cleanup Script
Removes unnecessary files before pushing to GitHub
"""

import os
import shutil
from pathlib import Path

def cleanup_project():
    """Clean up project for GitHub"""
    print("🧹 Cleaning up project for GitHub...")
    print("=" * 50)
    
    base_dir = Path("c:/Users/atusa/OneDrive/Desktop/Aerial VH/aerial_vehicle_detection")
    
    # Files and directories to DELETE
    files_to_delete = [
        # Test and temporary files
        "test_installation.py",
        "test_model_access.py",
        "quick_test.py", 
        "quick_start.py",
        "final_status.py",
        "simple_data_setup.py",
        "setup_demo.py",
        "data_setup_guide.py",
        "launch_labelimg.py",
        "annotation_setup.py",
        "create_demo_data.py",
        "convert_dataset.py",
        "convert_aerial_dataset.py",
        "dataset_conversion_template.py",
        "public_datasets.py",
        "huggingface_detector.py",
        "rebotnix_main.py",
        
        # Camera test files (keep main functionality)
        "traffic_monitoring_system/test_camera.py",
        "traffic_monitoring_system/advanced_camera_test.py", 
        "traffic_monitoring_system/directshow_camera_test.py",
        "traffic_monitoring_system/camera_fix_guide.py",
        
        # Cache and build directories
        "__pycache__",
        "traffic_monitoring_system/__pycache__",
        "src/__pycache__",
        "models/__pycache__",
        
        # Virtual environment
        "venv_aerial",
        
        # Large model files (will be downloaded)
        "yolov8n.pt",
        "../yolov8n.pt",
        
        # Results and data directories (keep structure but clean contents)
        # We'll handle these separately
    ]
    
    # Directories to clean but keep structure
    dirs_to_clean = [
        "detection_results",
        "results", 
        "data/temp",
        "traffic_monitoring_system/data/temp",
        "dataset_yolo"
    ]
    
    deleted_count = 0
    
    # Delete files
    for file_path in files_to_delete:
        full_path = base_dir / file_path
        try:
            if full_path.exists():
                if full_path.is_file():
                    full_path.unlink()
                    print(f"🗑️ Deleted file: {file_path}")
                    deleted_count += 1
                elif full_path.is_dir():
                    shutil.rmtree(full_path)
                    print(f"🗂️ Deleted directory: {file_path}")
                    deleted_count += 1
        except Exception as e:
            print(f"⚠️ Could not delete {file_path}: {e}")
    
    # Clean directories but keep structure
    for dir_path in dirs_to_clean:
        full_path = base_dir / dir_path
        if full_path.exists() and full_path.is_dir():
            try:
                # Remove contents but keep directory
                for item in full_path.iterdir():
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                print(f"🧽 Cleaned directory: {dir_path}")
            except Exception as e:
                print(f"⚠️ Could not clean {dir_path}: {e}")
    
    print(f"\n✅ Cleanup complete! Removed {deleted_count} items")
    print("\n📁 Essential files kept:")
    print("  - rfdetr_detector.py (main detection system)")
    print("  - traffic_monitoring_system/ (complete dashboard)")
    print("  - requirements.txt")
    print("  - README.md")
    print("  - main.py")
    print("  - setup.py")
    print("  - Documentation files")

def create_gitignore():
    """Create .gitignore for the project"""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
venv_aerial/
env/
ENV/

# Model files (downloaded automatically)
*.pth
*.pt
*.onnx
models/downloaded/
.cache/

# Data and results
data/temp/
data/uploads/
results/
detection_results/
dataset_yolo/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db
*.log

# Jupyter
.ipynb_checkpoints/

# Database files
*.db
*.sqlite
*.sqlite3

# Temporary files
*.tmp
temp/
"""
    
    gitignore_path = Path("c:/Users/atusa/OneDrive/Desktop/Aerial VH/aerial_vehicle_detection/.gitignore")
    with open(gitignore_path, 'w') as f:
        f.write(gitignore_content)
    
    print("📄 Created .gitignore file")

def show_final_structure():
    """Show the final project structure"""
    print("\n📂 Final Project Structure:")
    print("aerial_vehicle_detection/")
    print("├── README.md")
    print("├── requirements.txt")
    print("├── setup.py")
    print("├── main.py")
    print("├── rfdetr_detector.py")
    print("├── traffic_monitoring_launcher.py")
    print("├── .gitignore")
    print("├── config.json")
    print("├── QUICK_REFERENCE.json")
    print("├── traffic_monitoring_system/")
    print("│   ├── dashboard.py")
    print("│   ├── traffic_analytics.py")
    print("│   ├── realtime_processor.py")
    print("│   ├── data/ (empty)")
    print("│   ├── static/")
    print("│   └── templates/")
    print("├── src/")
    print("├── examples/")
    print("├── data/ (structure only)")
    print("└── Documentation files")

if __name__ == "__main__":
    print("🚀 Preparing project for GitHub")
    print("=" * 50)
    
    response = input("This will delete test files and clean directories. Continue? (y/N): ")
    if response.lower() == 'y':
        cleanup_project()
        create_gitignore()
        show_final_structure()
        
        print("\n🎯 Ready for GitHub!")
        print("Next steps:")
        print("1. cd to project directory")
        print("2. git init")
        print("3. git add .")
        print("4. git commit -m 'Initial commit: RF-DETR Traffic Monitoring System'")
        print("5. git remote add origin <your-github-repo-url>")
        print("6. git push -u origin main")
    else:
        print("❌ Cleanup cancelled")