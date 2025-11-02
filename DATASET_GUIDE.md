# Dataset Preparation Guide

## 📊 How to Prepare Your Aerial Vehicle Dataset

### Step 1: Collect Your Data
- Aerial/drone images containing vehicles
- Images should be in JPG, PNG, or similar formats
- Recommended resolution: 640x640 or higher
- Diverse conditions: different altitudes, lighting, weather

### Step 2: Create YOLO Format Labels
For each image, create a corresponding .txt file with:
```
class_id center_x center_y width height
```

Where:
- class_id: 0=car, 1=truck, 2=bus, 3=motorcycle, 4=bicycle, 5=van, 6=trailer
- All coordinates are normalized (0.0 to 1.0)
- center_x, center_y: center point of vehicle
- width, height: bounding box dimensions

Example label file (image1.txt):
```
0 0.5 0.3 0.1 0.2
1 0.7 0.6 0.15 0.25
```

### Step 3: Organize Your Dataset
```
your_dataset/
├── images/
│   ├── train/        # 80% of your images
│   ├── val/          # 20% of your images
│   └── test/         # Optional test set
├── labels/
│   ├── train/        # Corresponding labels
│   ├── val/          # Corresponding labels
│   └── test/         # Optional test labels
└── data.yaml         # Dataset configuration
```

### Step 4: Create data.yaml
```yaml
path: /path/to/your_dataset
train: images/train
val: images/val
test: images/test

nc: 7
names: ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'van', 'trailer']
```

### Step 5: Place in Project
Copy your prepared dataset to:
```
aerial_vehicle_detection/data/your_dataset/
```

### Annotation Tools
- **LabelImg**: Popular tool for bounding box annotation
- **Roboflow**: Online annotation platform
- **CVAT**: Computer Vision Annotation Tool
- **VGG Image Annotator**: Web-based tool

### Quality Tips
- ✅ Consistent labeling across all images
- ✅ Include vehicles at different scales
- ✅ Label partially visible vehicles
- ✅ Balance between classes if possible
- ✅ Check for mislabeled or missed vehicles