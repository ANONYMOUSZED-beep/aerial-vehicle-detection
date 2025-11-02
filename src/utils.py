"""
Utility functions for Aerial Vehicle Detection System

This module contains helper functions for:
- GPU/device detection and setup
- Logging configuration
- File handling and path operations
- Visualization utilities
- Data preprocessing helpers
- Performance monitoring
"""

import os
import sys
import logging
import time
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import psutil
from datetime import datetime


def setup_logging(log_level: str = 'INFO', 
                 log_file: Optional[str] = None,
                 console_output: bool = True) -> logging.Logger:
    """
    Set up logging configuration for the application.
    
    Args:
        log_level (str): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file (Optional[str]): Path to log file (optional)
        console_output (bool): Whether to output to console
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger('aerial_vehicle_detection')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # File gets all messages
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def check_gpu() -> str:
    """
    Check GPU availability and return appropriate device string.
    
    Returns:
        str: Device string ('cuda', 'cpu', or specific GPU like 'cuda:0')
    """
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        return 'cpu'
    
    num_gpus = torch.cuda.device_count()
    print(f"🚀 Found {num_gpus} GPU(s) available")
    
    for i in range(num_gpus):
        gpu_props = torch.cuda.get_device_properties(i)
        memory_gb = gpu_props.total_memory / 1e9
        print(f"   GPU {i}: {gpu_props.name} ({memory_gb:.1f} GB)")
    
    # Return the first GPU by default
    return 'cuda:0' if num_gpus > 0 else 'cpu'


def get_system_info() -> Dict[str, Any]:
    """
    Get comprehensive system information.
    
    Returns:
        Dict[str, Any]: System information including CPU, memory, GPU details
    """
    info = {
        'timestamp': datetime.now().isoformat(),
        'python_version': sys.version,
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cpu_count': psutil.cpu_count(),
        'memory_total_gb': psutil.virtual_memory().total / 1e9,
        'memory_available_gb': psutil.virtual_memory().available / 1e9,
        'platform': sys.platform
    }
    
    # GPU information
    if torch.cuda.is_available():
        info['gpu_count'] = torch.cuda.device_count()
        info['gpus'] = []
        
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            gpu_info = {
                'id': i,
                'name': gpu_props.name,
                'memory_total_gb': gpu_props.total_memory / 1e9,
                'compute_capability': f"{gpu_props.major}.{gpu_props.minor}"
            }
            info['gpus'].append(gpu_info)
    
    return info


def save_system_info(save_path: str):
    """
    Save system information to a JSON file.
    
    Args:
        save_path (str): Path to save the system info
    """
    info = get_system_info()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"💾 System info saved to: {save_path}")


def ensure_dir_exists(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path (Union[str, Path]): Directory path
        
    Returns:
        Path: Path object of the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_size(filepath: Union[str, Path]) -> str:
    """
    Get human-readable file size.
    
    Args:
        filepath (Union[str, Path]): Path to file
        
    Returns:
        str: Human-readable file size (e.g., "1.2 MB")
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return "File not found"
    
    size_bytes = filepath.stat().st_size
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    
    return f"{size_bytes:.1f} PB"


def find_images(directory: Union[str, Path], 
               extensions: List[str] = None) -> List[Path]:
    """
    Find all image files in a directory.
    
    Args:
        directory (Union[str, Path]): Directory to search
        extensions (List[str]): Image extensions to look for
        
    Returns:
        List[Path]: List of image file paths
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    directory = Path(directory)
    image_files = []
    
    for ext in extensions:
        image_files.extend(directory.glob(f'**/*{ext}'))
        image_files.extend(directory.glob(f'**/*{ext.upper()}'))
    
    return sorted(image_files)


def validate_dataset_structure(dataset_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Validate YOLO dataset structure and return analysis.
    
    Args:
        dataset_path (Union[str, Path]): Path to dataset directory
        
    Returns:
        Dict[str, Any]: Dataset structure analysis
    """
    dataset_path = Path(dataset_path)
    analysis = {
        'valid': False,
        'errors': [],
        'warnings': [],
        'structure': {},
        'statistics': {}
    }
    
    # Check main directory exists
    if not dataset_path.exists():
        analysis['errors'].append(f"Dataset directory does not exist: {dataset_path}")
        return analysis
    
    # Check for required directories
    required_dirs = ['images', 'labels']
    for dir_name in required_dirs:
        dir_path = dataset_path / dir_name
        if not dir_path.exists():
            analysis['errors'].append(f"Missing required directory: {dir_name}")
        else:
            analysis['structure'][dir_name] = str(dir_path)
    
    # Check for splits
    splits = ['train', 'val', 'test']
    for split in splits:
        img_dir = dataset_path / 'images' / split
        lbl_dir = dataset_path / 'labels' / split
        
        if img_dir.exists():
            images = find_images(img_dir)
            labels = list(lbl_dir.glob('*.txt')) if lbl_dir.exists() else []
            
            analysis['statistics'][split] = {
                'images': len(images),
                'labels': len(labels),
                'matched': 0
            }
            
            # Check for matching image-label pairs
            matched = 0
            for img_file in images:
                lbl_file = lbl_dir / f"{img_file.stem}.txt"
                if lbl_file.exists():
                    matched += 1
            
            analysis['statistics'][split]['matched'] = matched
            
            if len(images) > 0 and matched == 0:
                analysis['warnings'].append(f"No matching labels found for {split} images")
            elif matched < len(images):
                analysis['warnings'].append(f"Only {matched}/{len(images)} images have labels in {split}")
    
    # Check for data.yaml
    yaml_path = dataset_path / 'data.yaml'
    if yaml_path.exists():
        analysis['structure']['data_yaml'] = str(yaml_path)
    else:
        analysis['warnings'].append("No data.yaml file found")
    
    # Overall validation
    analysis['valid'] = len(analysis['errors']) == 0
    
    return analysis


def create_visualization_grid(images: List[np.ndarray], 
                            titles: List[str] = None,
                            grid_size: Tuple[int, int] = None,
                            figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Create a grid visualization of multiple images.
    
    Args:
        images (List[np.ndarray]): List of images to display
        titles (List[str]): Titles for each image
        grid_size (Tuple[int, int]): Grid dimensions (rows, cols)
        figsize (Tuple[int, int]): Figure size
        
    Returns:
        plt.Figure: Matplotlib figure object
    """
    num_images = len(images)
    
    if grid_size is None:
        cols = int(np.ceil(np.sqrt(num_images)))
        rows = int(np.ceil(num_images / cols))
    else:
        rows, cols = grid_size
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if num_images > 1 else [axes]
    
    for i in range(num_images):
        axes[i].imshow(images[i])
        axes[i].axis('off')
        
        if titles and i < len(titles):
            axes[i].set_title(titles[i])
    
    # Hide empty subplots
    for i in range(num_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1 (List[float]): First bounding box [x1, y1, x2, y2]
        box2 (List[float]): Second bounding box [x1, y1, x2, y2]
        
    Returns:
        float: IoU value between 0 and 1
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def non_max_suppression(boxes: np.ndarray, 
                       scores: np.ndarray, 
                       iou_threshold: float = 0.45) -> List[int]:
    """
    Apply Non-Maximum Suppression to remove overlapping boxes.
    
    Args:
        boxes (np.ndarray): Array of bounding boxes [N, 4] (x1, y1, x2, y2)
        scores (np.ndarray): Array of confidence scores [N]
        iou_threshold (float): IoU threshold for suppression
        
    Returns:
        List[int]: Indices of boxes to keep
    """
    if len(boxes) == 0:
        return []
    
    # Sort by confidence scores
    indices = np.argsort(scores)[::-1]
    keep = []
    
    while len(indices) > 0:
        # Pick the box with highest confidence
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
        
        # Calculate IoU with remaining boxes
        current_box = boxes[current]
        remaining_boxes = boxes[indices[1:]]
        
        ious = np.array([calculate_iou(current_box, box) for box in remaining_boxes])
        
        # Keep boxes with IoU less than threshold
        indices = indices[1:][ious < iou_threshold]
    
    return keep


class Timer:
    """Context manager and decorator for timing code execution."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        print(f"⏱️  {self.name} completed in {elapsed:.3f} seconds")
    
    def __call__(self, func):
        """Use as decorator."""
        def wrapper(*args, **kwargs):
            with Timer(func.__name__):
                return func(*args, **kwargs)
        return wrapper


class ProgressTracker:
    """Track and display progress for long-running operations."""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.description = description
        self.current = 0
        self.start_time = time.time()
    
    def update(self, increment: int = 1):
        """Update progress by increment."""
        self.current += increment
        self._display_progress()
    
    def _display_progress(self):
        """Display current progress."""
        if self.total == 0:
            return
        
        percent = (self.current / self.total) * 100
        elapsed = time.time() - self.start_time
        
        if self.current > 0:
            eta = (elapsed / self.current) * (self.total - self.current)
            eta_str = f"ETA: {eta:.0f}s"
        else:
            eta_str = "ETA: --"
        
        bar_length = 40
        filled_length = int(bar_length * self.current // self.total)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        print(f'\r{self.description}: |{bar}| {percent:.1f}% ({self.current}/{self.total}) {eta_str}', 
              end='', flush=True)
        
        if self.current >= self.total:
            print()  # New line when complete


def load_image(image_path: Union[str, Path], 
              target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Load and preprocess an image.
    
    Args:
        image_path (Union[str, Path]): Path to image file
        target_size (Optional[Tuple[int, int]]): Target size (width, height)
        
    Returns:
        np.ndarray: Loaded image in RGB format
    """
    image_path = Path(image_path)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize if target size specified
    if target_size:
        image = cv2.resize(image, target_size)
    
    return image


def save_results_summary(results: Dict[str, Any], save_path: Union[str, Path]):
    """
    Save results summary to JSON file.
    
    Args:
        results (Dict[str, Any]): Results dictionary
        save_path (Union[str, Path]): Path to save summary
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add timestamp
    results['timestamp'] = datetime.now().isoformat()
    results['system_info'] = get_system_info()
    
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"📊 Results summary saved to: {save_path}")


if __name__ == "__main__":
    # Test utility functions
    print("🧪 Testing utility functions...")
    
    # Test logging setup
    logger = setup_logging('INFO', 'test.log')
    logger.info("Logging test successful")
    
    # Test GPU detection
    device = check_gpu()
    print(f"Detected device: {device}")
    
    # Test system info
    sys_info = get_system_info()
    print(f"System info: {sys_info['cpu_count']} CPUs, {sys_info['memory_total_gb']:.1f}GB RAM")
    
    # Test timer
    with Timer("Test operation"):
        time.sleep(0.1)
    
    # Test progress tracker
    tracker = ProgressTracker(100, "Testing progress")
    for i in range(101):
        tracker.update()
        time.sleep(0.01)
    
    print("✅ All utility tests passed!")