"""
Dataset Loader for Aerial Vehicle Detection

This module handles loading and preprocessing of aerial vehicle datasets in YOLO format.
It supports various dataset formats commonly used for aerial object detection including
UAVDT, DOTA, and custom datasets.

Expected directory structure:
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/  
│   └── test/
└── data.yaml

YOLO label format: class_id center_x center_y width height (normalized 0-1)
"""

import os
import yaml
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class AerialDatasetLoader:
    """
    Handles loading and preprocessing of aerial vehicle datasets.
    Supports YOLO format datasets with proper validation and augmentation.
    """
    
    def __init__(self, dataset_path: str, img_size: int = 640):
        """
        Initialize the dataset loader.
        
        Args:
            dataset_path (str): Path to the dataset directory
            img_size (int): Target image size for resizing (default: 640)
        """
        self.dataset_path = Path(dataset_path)
        self.img_size = img_size
        self.class_names = []
        self.num_classes = 0
        
        # Vehicle class mapping for aerial detection
        self.default_classes = {
            0: 'car',
            1: 'truck', 
            2: 'bus',
            3: 'motorcycle',
            4: 'bicycle',
            5: 'van',
            6: 'trailer'
        }
        
        self._validate_dataset_structure()
        self._load_dataset_config()
    
    def _validate_dataset_structure(self) -> None:
        """Validate that the dataset has the required directory structure."""
        # Support two common YOLO layouts:
        # 1) dataset/images/{train,val,test} and dataset/labels/{train,val,test}
        # 2) dataset/{train,val,test}/images and dataset/{train,val,test}/labels

        images_root = self.dataset_path / 'images'
        labels_root = self.dataset_path / 'labels'

        nested_train_images = self.dataset_path / 'train' / 'images'
        nested_train_labels = self.dataset_path / 'train' / 'labels'

        if images_root.exists() and labels_root.exists():
            # layout type 1
            self._layout = 'flat'
        elif nested_train_images.exists() and nested_train_labels.exists():
            # layout type 2 (nested)
            self._layout = 'nested'
        else:
            # Not a strict error yet — allow loader to create default data.yaml later
            raise FileNotFoundError(f"Required dataset directories not found in {self.dataset_path}. Expected either 'images'/'labels' or nested 'train/images' and 'train/labels'.")
    
    def _load_dataset_config(self) -> None:
        """Load dataset configuration from data.yaml file."""
        config_path = self.dataset_path / 'data.yaml'
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            self.class_names = config.get('names', list(self.default_classes.values()))
            self.num_classes = len(self.class_names)
            
            print(f"Loaded dataset config with {self.num_classes} classes: {self.class_names}")
        else:
            print("No data.yaml found, creating default configuration...")
            self._create_default_config()

    def _img_dir(self, split: str) -> Path:
        """Return the image directory Path for a given split, handling both layouts."""
        if getattr(self, '_layout', None) == 'flat':
            return self.dataset_path / 'images' / split
        else:
            # nested layout uses 'valid' instead of 'val' commonly
            split_name = 'valid' if split == 'val' else split
            return self.dataset_path / split_name / 'images'

    def _lbl_dir(self, split: str) -> Path:
        """Return the label directory Path for a given split, handling both layouts."""
        if getattr(self, '_layout', None) == 'flat':
            return self.dataset_path / 'labels' / split
        else:
            split_name = 'valid' if split == 'val' else split
            return self.dataset_path / split_name / 'labels'
    
    def _create_default_config(self) -> None:
        """Create a default data.yaml configuration file."""
        self.class_names = list(self.default_classes.values())
        self.num_classes = len(self.class_names)
        
        config = {
            'path': str(self.dataset_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': self.num_classes,
            'names': self.class_names
        }
        
        config_path = self.dataset_path / 'data.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"Created default data.yaml with {self.num_classes} vehicle classes")
    
    def get_dataset_info(self) -> Dict:
        """
        Get comprehensive dataset information including file counts and class distribution.
        
        Returns:
            Dict: Dataset statistics and information
        """
        info = {
            'dataset_path': str(self.dataset_path),
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'splits': {}
        }
        
        for split in ['train', 'val', 'test']:
            img_dir = self._img_dir(split)
            lbl_dir = self._lbl_dir(split)
            
            if img_dir.exists():
                # Count images
                img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
                img_files = []
                for ext in img_extensions:
                    img_files.extend(glob.glob(str(img_dir / ext)))
                
                # Count labels
                lbl_files = list(lbl_dir.glob('*.txt')) if lbl_dir.exists() else []
                
                # Analyze class distribution
                class_counts = {name: 0 for name in self.class_names}
                total_objects = 0
                
                for lbl_file in lbl_files:
                    try:
                        with open(lbl_file, 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    class_id = int(parts[0])
                                    if 0 <= class_id < len(self.class_names):
                                        class_counts[self.class_names[class_id]] += 1
                                        total_objects += 1
                    except Exception as e:
                        print(f"Warning: Error reading {lbl_file}: {e}")
                
                info['splits'][split] = {
                    'num_images': len(img_files),
                    'num_labels': len(lbl_files),
                    'total_objects': total_objects,
                    'class_distribution': class_counts,
                    'avg_objects_per_image': total_objects / max(len(img_files), 1)
                }
        
        return info
    
    def create_yolo_dataset_yaml(self) -> str:
        """
        Create or update the data.yaml file required by YOLOv8.
        
        Returns:
            str: Path to the created data.yaml file
        """
        config_path = self.dataset_path / 'data.yaml'
        
        # Determine YAML-relative paths depending on layout
        if getattr(self, '_layout', None) == 'nested':
            train_rel = 'train/images'
            val_rel = 'valid/images'
            test_rel = 'test/images'
        else:
            train_rel = 'images/train'
            val_rel = 'images/val'
            test_rel = 'images/test'

        config = {
            'path': str(self.dataset_path.absolute()).replace('\\', '/'),
            'train': train_rel,
            'val': val_rel,
            'nc': self.num_classes,
            'names': self.class_names
        }

        # Add test split if it exists
        test_dir = self._img_dir('test')
        if test_dir.exists():
            config['test'] = test_rel
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"Created/updated data.yaml at: {config_path}")
        return str(config_path)


class AerialVehicleDataset(Dataset):
    """
    Custom PyTorch Dataset for aerial vehicle detection.
    Handles loading images and labels with proper preprocessing.
    """
    
    def __init__(self, 
                 img_dir: str, 
                 lbl_dir: str, 
                 img_size: int = 640,
                 augment: bool = False,
                 class_names: List[str] = None):
        """
        Initialize the dataset.
        
        Args:
            img_dir (str): Directory containing images
            lbl_dir (str): Directory containing label files
            img_size (int): Target image size
            augment (bool): Apply data augmentation
            class_names (List[str]): List of class names
        """
        self.img_dir = Path(img_dir)
        self.lbl_dir = Path(lbl_dir)
        self.img_size = img_size
        self.augment = augment
        self.class_names = class_names or []
        
        # Get all image files
        img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        self.img_files = []
        
        for ext in img_extensions:
            self.img_files.extend(self.img_dir.glob(f'*{ext}'))
            self.img_files.extend(self.img_dir.glob(f'*{ext.upper()}'))
        
        self.img_files = sorted(self.img_files)
        
        # Verify corresponding label files exist
        self.label_files = []
        for img_file in self.img_files:
            lbl_file = self.lbl_dir / f"{img_file.stem}.txt"
            self.label_files.append(lbl_file if lbl_file.exists() else None)
        
        print(f"Found {len(self.img_files)} images in {img_dir}")
        valid_labels = sum(1 for lbl in self.label_files if lbl is not None)
        print(f"Found {valid_labels} corresponding label files")
        
        # Define transforms
        self.transform = self._get_transforms()
    
    def _get_transforms(self):
        """Get image preprocessing transforms."""
        if self.augment:
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self) -> int:
        return len(self.img_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single item from the dataset.
        
        Args:
            idx (int): Index of the item
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Image tensor and label tensor
        """
        # Load image
        img_path = self.img_files[idx]
        image = Image.open(img_path).convert('RGB')
        original_size = image.size  # (width, height)
        
        # Apply transforms
        image = self.transform(image)
        
        # Load labels
        lbl_path = self.label_files[idx]
        labels = []
        
        if lbl_path and lbl_path.exists():
            try:
                with open(lbl_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            cx, cy, w, h = map(float, parts[1:5])
                            labels.append([class_id, cx, cy, w, h])
            except Exception as e:
                print(f"Warning: Error reading label file {lbl_path}: {e}")
        
        # Convert to tensor
        if labels:
            labels = torch.tensor(labels, dtype=torch.float32)
        else:
            labels = torch.zeros((0, 5), dtype=torch.float32)
        
        return image, labels
    
    def collate_fn(self, batch):
        """Custom collate function for batching variable-length labels."""
        images, labels = zip(*batch)
        
        # Stack images
        images = torch.stack(images, 0)
        
        # Add batch index to labels
        for i, label in enumerate(labels):
            if len(label):
                batch_idx = torch.full((len(label), 1), i, dtype=torch.float32)
                labels[i] = torch.cat([batch_idx, label], dim=1)
        
        # Concatenate all labels
        labels = torch.cat([l for l in labels if len(l)], 0)
        
        return images, labels


def load_dataset(dataset_path: str, 
                batch_size: int = 16, 
                img_size: int = 640,
                num_workers: int = 4) -> Tuple[DataLoader, DataLoader, AerialDatasetLoader]:
    """
    Main function to load aerial vehicle dataset and create data loaders.
    
    Args:
        dataset_path (str): Path to the dataset directory
        batch_size (int): Batch size for data loaders
        img_size (int): Target image size
        num_workers (int): Number of worker processes for data loading
        
    Returns:
        Tuple[DataLoader, DataLoader, AerialDatasetLoader]: Train loader, validation loader, and dataset info
    """
    print(f"Loading aerial vehicle dataset from: {dataset_path}")
    
    # Initialize dataset loader
    dataset_loader = AerialDatasetLoader(dataset_path, img_size)
    
    # Get dataset info
    dataset_info = dataset_loader.get_dataset_info()
    print("\n📊 Dataset Information:")
    for split, info in dataset_info['splits'].items():
        if info['num_images'] > 0:
            print(f"  {split.upper()}:")
            print(f"    Images: {info['num_images']}")
            print(f"    Labels: {info['num_labels']}")
            print(f"    Objects: {info['total_objects']}")
            print(f"    Avg objects/image: {info['avg_objects_per_image']:.2f}")
    
    # Create data loaders
    train_loader = None
    val_loader = None
    
    train_img_dir = dataset_loader._img_dir('train')
    train_lbl_dir = dataset_loader._lbl_dir('train')
    
    if train_img_dir.exists():
        train_dataset = AerialVehicleDataset(
            str(train_img_dir),
            str(train_lbl_dir),
            img_size=img_size,
            augment=True,
            class_names=dataset_loader.class_names
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=train_dataset.collate_fn,
            pin_memory=True
        )
    
    val_img_dir = dataset_loader._img_dir('val')
    val_lbl_dir = dataset_loader._lbl_dir('val')
    
    if val_img_dir.exists():
        val_dataset = AerialVehicleDataset(
            str(val_img_dir),
            str(val_lbl_dir),
            img_size=img_size,
            augment=False,
            class_names=dataset_loader.class_names
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=val_dataset.collate_fn,
            pin_memory=True
        )
    
    print(f"✅ Dataset loaded successfully!")
    print(f"   Classes: {dataset_loader.class_names}")
    print(f"   Training samples: {len(train_loader.dataset) if train_loader else 0}")
    print(f"   Validation samples: {len(val_loader.dataset) if val_loader else 0}")
    
    return train_loader, val_loader, dataset_loader


if __name__ == "__main__":
    # Example usage
    dataset_path = "./data/aerial_vehicles"
    
    try:
        train_loader, val_loader, dataset_info = load_dataset(
            dataset_path=dataset_path,
            batch_size=8,
            img_size=640
        )
        
        print("\n🧪 Testing data loading...")
        if train_loader:
            images, labels = next(iter(train_loader))
            print(f"Batch shape: {images.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Sample labels: {labels[:5]}")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have a properly structured dataset in the specified path.")