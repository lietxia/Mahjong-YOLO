# Mahjong Tile Recognition with YOLO

A computer vision project that uses YOLO (You Only Look Once) models to detect and recognize mahjong tiles from real-world photographs. The project includes multiple model variants (nano, small, medium, large, extra-large) optimized for different use cases.

Dataset: https://www.kaggle.com/datasets/shinz114514/mahjong-hand-photos-taken-with-mobile-camera/data

## 🎯 Project Overview

This project implements mahjong tile recognition using YOLOv11, capable of:
- Detecting mahjong tiles in real-world photographs
- Recognizing different tile types and suits
- Processing images with various lighting conditions and backgrounds
- Providing both PyTorch (.pt) and ONNX model formats for deployment

## 📁 Project Structure

```
├── models/                          # Trained models organized by size
│   ├── nano/                        # YOLOv11n models (fastest, lowest accuracy)
│   ├── small/                       # YOLOv11s models (balanced speed/accuracy)
│   ├── medium/                      # YOLOv11m models (good accuracy)
│   ├── large/                       # YOLOv11l models (high accuracy)
│   ├── extra_large/                 # YOLOv11x models (highest accuracy)
│   └── *.onnx                       # ONNX format models for deployment
├── scripts/                         # Utility scripts
│   ├── convert_yolo_to_onnx.py      # Convert PyTorch models to ONNX
│   └── convert_yolo_to_coreml.py    # Convert PyTorch models to CoreML
├── notebooks/                       # Jupyter notebooks for training and analysis
│   ├── data_labeling/               # Data annotation and labeling notebooks
│   ├── data_processing/             # Data preprocessing notebooks
│   ├── yolo.ipynb                   # YOLO training notebook
│   └── yolo_predict.ipynb           # Prediction and evaluation notebook
├── results/                         # Training and evaluation results
│   ├── training/                    # Training logs, metrics, and model checkpoints
│   ├── validation/                  # Validation results
│   └── predictions/                 # Prediction outputs and visualizations
├── data/                            # Dataset organization
│   ├── raw/                         # Original images
│   ├── processed/                   # Preprocessed images
│   └── annotations/                 # Label files
├── docs/                            # Documentation
├── examples/                        # Usage examples
└── README.md                        # This file
```

## 🚀 Model Variants

### Currently Available Trained Models

| Model Size | Base Model | Trained Model | Training Status | Speed | Accuracy | Use Case | Avg Detections |
|------------|------------|---------------|-----------------|-------|----------|----------|----------------|
| Nano | yolo11n.pt | trained_models_v2/yolo11n_best.pt | ✅ Complete | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ | Mobile/Edge devices | 22.3 |
| Small | yolo11s.pt | trained_models_v2/yolo11s_best.pt | ✅ Complete | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | Real-time applications | 24.3 |
| Medium | yolo11m.pt | trained_models_v2/yolo11m_best.pt | ✅ Complete | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | Balanced performance | 28.0 |
| Large | yolo11l.pt | trained_models_v2/yolo11l_best.pt | 🔄 In Progress | ⚡⚡ | ⭐⭐⭐⭐⭐⭐ | High accuracy needs | - |
| Extra Large | yolo11x.pt | - | ⏳ Planned | ⚡ | ⭐⭐⭐⭐⭐⭐⭐ | Maximum accuracy | - |

### Model Performance

- **Nano (YOLOv11n)**: Fastest inference, optimized for mobile deployment
- **Small (YOLOv11s)**: Good balance of speed and accuracy for real-time applications
- **Medium (YOLOv11m)**: Recommended for most use cases, best accuracy/speed trade-off
- **Large (YOLOv11l)**: High accuracy for production applications
- **Extra Large (YOLOv11x)**: Maximum accuracy when speed is not critical

## 🛠️ Installation

### Prerequisites

```bash
pip install ultralytics opencv-python matplotlib torch torchvision
```

### Additional Dependencies for Development

```bash
pip install jupyter notebook albumentations numpy
```

## 💻 Usage

### Quick Start - Inference

```python
from ultralytics import YOLO

# Load a trained model
model = YOLO('trained_models_v2/yolo11m_best.pt')

# Run inference on an image
results = model.predict('path/to/mahjong/image.jpg')

# Display results
results[0].show()
```

### 🎯 Inference Examples

The repository includes visual examples demonstrating detection performance across different model sizes:

![Model Comparison](inference_examples/comparison_58eec28b-000021.png)

**Detection Performance Summary:**
- **YOLOv11n (Nano)**: Average 22.3 detections per image
- **YOLOv11s (Small)**: Average 24.3 detections per image
- **YOLOv11m (Medium)**: Average 28.0 detections per image

**Sample Results:**
- [YOLOv11n Examples](inference_examples/) - Fastest inference for mobile deployment
- [YOLOv11s Examples](inference_examples/) - Balanced speed and accuracy
- [YOLOv11m Examples](inference_examples/) - Best accuracy among completed models

Generated using: `python3 generate_inference_examples.py`

### Using ONNX Models

```python
import onnxruntime as ort
import cv2
import numpy as np

# Load ONNX model
session = ort.InferenceSession('models/mahjong-yolom-best.onnx')

# Preprocess image
img = cv2.imread('path/to/image.jpg')
img_resized = cv2.resize(img, (640, 640))
img_normalized = img_resized.astype(np.float32) / 255.0
img_transposed = np.transpose(img_normalized, (2, 0, 1))
img_batch = np.expand_dims(img_transposed, axis=0)

# Run inference
outputs = session.run(None, {'images': img_batch})
```

### Model Conversion

Convert PyTorch models to ONNX format:

```bash
python scripts/convert_yolo_to_onnx.py models/medium/mahjong-yolom-best.pt
```

Convert PyTorch models to CoreML format:

```bash
python scripts/convert_yolo_to_coreml.py models/medium/mahjong-yolom-best.pt
```

Batch conversion (all models):

```bash
python scripts/convert_yolo_to_onnx.py models/ --batch
python scripts/convert_yolo_to_coreml.py models/ --batch
```

## 🎓 Training

### Data Preparation

1. Organize your dataset in YOLO format:
   ```
   dataset/
   ├── images/
   │   ├── train/
   │   ├── val/
   │   └── test/
   └── labels/
       ├── train/
       ├── val/
       └── test/
   ```

2. Create a data configuration file (`data.yaml`):
   ```yaml
   train: path/to/train/images
   val: path/to/val/images
   test: path/to/test/images
   
   nc: 34  # number of classes (mahjong tile types)
   names: ['1m', '2m', '3m', ..., 'red', 'green', 'white']
   ```

### Training Different Model Sizes

```python
from ultralytics import YOLO

# Train nano model
model = YOLO('models/nano/yolo11n.pt')
model.train(data='data.yaml', epochs=500, batch=24, name='mahjong-yolon')

# Train small model
model = YOLO('models/small/yolo11s.pt')
model.train(data='data.yaml', epochs=500, batch=16, name='mahjong-yolos')

# Train medium model
model = YOLO('models/medium/yolo11m.pt')
model.train(data='data.yaml', epochs=500, batch=12, name='mahjong-yolom')

# Train large model
model = YOLO('models/large/yolo11l.pt')
model.train(data='data.yaml', epochs=500, batch=10, name='mahjong-yolol')
```

## 📊 Evaluation

### Model Validation

```python
# Validate trained model
model = YOLO('models/medium/mahjong-yolom-best.pt')
metrics = model.val()

print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
```

### Performance Metrics

Training results include:
- Precision/Recall curves
- F1 score curves
- Confusion matrices
- Training loss graphs
- Validation metrics

## 🎯 Mahjong Tile Classes

The model recognizes 38 different mahjong tile types:

### Number Tiles (Man/Wan - Characters)
- 1m through 9m, 0m (red five)

### Number Tiles (Pin/Bing - Circles)
- 1p through 9p, 0p (red five)

### Number Tiles (Sou/Tiao - Bamboos)
- 1s through 9s, 0s (red five)

### Honor Tiles (Winds - Z tiles)
- 1z (East), 2z (South), 3z (West), 4z (North)

### Honor Tiles (Dragons - Z tiles)
- 5z (Red Dragon), 6z (Green Dragon), 7z (White Dragon)

### Special Recognition
- UNKNOWN class for unclear or damaged tiles

**Total Classes**: 38 (including red fives and unknown category)

## 🔧 Customization

### Adding New Tile Types

1. Update the data configuration file with new classes
2. Retrain the model with expanded dataset
3. Update the class names in prediction scripts

### Hyperparameter Tuning

Key training parameters to adjust:
- `batch`: Batch size (adjust based on GPU memory)
- `lr0`: Initial learning rate
- `epochs`: Training epochs
- `patience`: Early stopping patience
- `conf`: Confidence threshold for predictions
- `iou`: IoU threshold for NMS

## 📈 Performance Tips

### For Speed
- Use nano or small models
- Convert to ONNX format
- Use TensorRT for NVIDIA GPUs
- Optimize input image size

### For Accuracy
- Use medium, large, or extra-large models
- Increase training epochs
- Use data augmentation
- Ensemble multiple models

### For Deployment
- Use ONNX models for cross-platform compatibility
- Use CoreML models for iOS/macOS deployment
- Implement batch processing for multiple images
- Use GPU acceleration when available

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Zhen Zhang** - zhenz@vt.edu
- **Yiyun Huang** - yiyunh@vt.edu

## 🙏 Acknowledgments

- [Ultralytics](https://ultralytics.com/) for the YOLO implementation
- The computer vision community for datasets and techniques
- Contributors to the mahjong recognition research

## 📞 Support

For questions and support:
- Open an issue on GitHub
- Check the documentation in the `docs/` folder
- Review the example notebooks in `notebooks/`

---

*Built with ❤️ for the mahjong and computer vision communities*
