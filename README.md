# 🌴 Palm Oil Fruit Bunch Detection & Counting — YOLOv8

> Real-time detection and automated counting of palm oil fruit bunches (Tandan Buah Segar) using YOLOv8 — an industry collaboration project with **PT Perkebunan Nusantara (PTPN)**, developed at **Binus University**.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-ff4c2b?logo=ultralytics)](https://github.com/ultralytics/ultralytics)
[![Roboflow](https://img.shields.io/badge/Dataset-Roboflow-purple?logo=roboflow)](https://roboflow.com)
[![ONNX](https://img.shields.io/badge/Export-ONNX-lightgrey?logo=onnx)](https://onnx.ai)
[![Colab](https://img.shields.io/badge/Notebook-Google%20Colab-orange?logo=googlecolab)](https://colab.research.google.com)

---

## 🎯 Project Overview

This project automates the quality grading and counting of **palm oil fruit bunches (Tandan Buah Segar/TBS)** using computer vision. The model detects and classifies each bunch into three categories, enabling faster and more accurate quality assessment in palm oil plantations.

**Industry Partner:** PT Perkebunan Nusantara (PTPN)  
**Institution:** Binus University

---

## 🍇 Detection Classes

| Class | Label | Color | Description |
|---|---|---|---|
| 0 | **Matang** 🟢 | Dark Green | Ripe — ready for harvest |
| 1 | **Mentah** 🔵 | Blue | Unripe — not ready |
| 2 | **Cacat** 🔴 | Red | Defective — damaged or abnormal |

---

## 🏗️ Pipeline

```
Aerial / Field Image
        │
        ▼
  Roboflow Dataset
  (palm-oil-nk2ks / cluster-palm-oil v12)
        │
        ▼
  YOLOv8 Training
  ├─ Model   : yolov8n (nano)
  ├─ Epochs  : 200
  ├─ ImgSz   : 800×800
  ├─ Batch   : Auto (-1)
  └─ Workers : 16
        │
        ▼
  Object Detection & Counting
  ├─ Custom colored bounding boxes per class
  ├─ Numbered labels per detected object
  ├─ Per-class count: Matang / Mentah / Cacat
  └─ Confidence threshold tuning (0.03 – 0.6)
        │
        ▼
  Export to ONNX
  (for deployment / edge device)
        │
        ▼
  TensorBoard Monitoring
```

---

## ✨ Key Features

- **3-class palm oil quality detection** — Ripe, Unripe, and Defective bunches in a single pass
- **Automated object counting** — each detected bunch is labeled and numbered on the output image
- **Color-coded bounding boxes** — green for Matang, red for Cacat, blue for Mentah for instant visual grading
- **Confidence threshold tuning** — configurable from 0.03 to 0.6 for different field conditions
- **ONNX export** — simplified model (opset 13) for edge deployment on plantation hardware
- **TensorBoard integration** — real-time training monitoring
- **Google Drive integration** — persistent model storage across Colab sessions

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Model** | YOLOv8 (Ultralytics) |
| **Dataset Management** | Roboflow API |
| **Computer Vision** | OpenCV (cv2) |
| **Export Format** | ONNX (opset 13, imgsz 416) |
| **Training Monitor** | TensorBoard |
| **Environment** | Google Colab + Google Drive |
| **Language** | Python 3.10+ |

---

## 📁 Project Structure

```
YOLO8/
├── ultralytics/                  # YOLOv8 cloned repository
│   ├── runs/
│   │   └── detect/
│   │       └── train/
│   │           └── weights/
│   │               └── best.pt   # Best trained model weights
│   ├── Data_17 05 2025/          # Field test images
│   │   ├── Percobaan1.jpg
│   │   ├── Percobaan3.jpg
│   │   └── Percobaan8.jpg
│   └── target_prediction/        # Prediction targets
├── cluster-palm-oil-12/          # Roboflow dataset (YOLOv8 OBB format)
├── chip.yaml                     # Dataset config
└── YOLO8_COUNT_OBJECT.ipynb      # Main notebook
```

---

## 🚀 Getting Started

### 1. Setup Environment
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/YOLO8

# Install dependencies
!pip install -U roboflow
!git clone https://github.com/ultralytics/ultralytics
%pip install -qe ultralytics
```

### 2. Download Dataset via Roboflow
```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("palm-oil-nk2ks").project("cluster-palm-oil")
dataset = project.version(12).download("yolov8-obb")
```

### 3. Train the Model
```bash
yolo task=detect mode=train \
     model=yolov8n.pt \
     data=/content/drive/MyDrive/YOLO8/chip.yaml \
     epochs=200 \
     workers=16 \
     batch=-1 \
     imgsz=800 \
     save=True \
     cache=ram \
     val=True \
     plots=True
```

### 4. Run Detection & Counting
```python
from ultralytics import YOLO
import cv2

model = YOLO('runs/detect/train/weights/best.pt')
class_names = {0: 'Cacat', 1: 'Matang', 2: 'Mentah'}

results = model(source='your_image.jpg', conf=0.5)

# Count per class
Total, Matang, Mentah, Cacat = 0, 0, 0, 0
for result in results:
    for class_index in result.boxes.cls.tolist():
        Total += 1
        if class_index == 1: Matang += 1
        elif class_index == 2: Mentah += 1
        elif class_index == 0: Cacat += 1

print(f"Total: {Total} | Matang: {Matang} | Mentah: {Mentah} | Cacat: {Cacat}")
```

### 5. Export to ONNX
```bash
yolo task=detect mode=export \
     model=runs/detect/train/weights/best.pt \
     format=onnx simplify=True opset=13 imgsz=416
```

---

## ⚙️ Training Configuration

| Parameter | Value | Notes |
|---|---|---|
| Base Model | `yolov8n.pt` | Nano — fast inference |
| Epochs | 200 | Transfer learning |
| Image Size | 800×800 | Optimal for cluster detail |
| Batch Size | -1 (auto) | Adapts to GPU VRAM |
| Workers | 16 | Faster data loading |
| Cache | RAM | Faster epoch cycling |
| Confidence | 0.03 – 0.6 | Tunable per condition |

---

## 🌐 Deployment

The model is exported to **ONNX format** (opset 13, 416×416) for potential deployment on:
- Edge devices at plantation sites
- Mobile inspection tools
- Integrated quality grading systems

---

## 🤝 Collaboration

This project is a **university-industry collaboration** between:

| | |
|---|---|
| **University** | Binus University |
| **Industry Partner** | PT Perkebunan Nusantara (PTPN) |
| **Use Case** | Automated TBS quality grading at harvest points |

---

## 📄 License

This project was developed for academic and industrial research purposes in collaboration with PTPN. Dataset and model weights are proprietary.
