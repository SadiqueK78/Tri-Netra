# TriNetra-AMRF — Dual-Modality Surveillance Framework

> **Tri**-spectral **Ne**ural **Tra**cking with **A**daptive **M**ulti-modal **R**easoning **F**usion

TriNetra is a dual-modality (RGB + Thermal) surveillance framework built on a 7-layer architecture:

1. **Data Acquisition** — Synchronized capture from visible and thermal sensors
2. **Preprocessing & Reliability (TRC)** — Alignment, denoising, normalization with Thermal Reliability Confidence scoring
3. **Dynamic Cross-Modal Fusion** — Adaptive fusion of RGB and thermal feature maps
4. **Multi-Scale Object Detection** — YOLO-based detection across multiple spatial scales
5. **Tracking & Behaviour Analysis** — ByteTrack/DeepSORT tracking with behavioural anomaly detection
6. **Hierarchical Threat Reasoning** — Multi-level risk assessment and threat classification
7. **Explainable Alert Generation** — Human-interpretable alerts with visual explanations

---

## Project Structure

```
TriNetra/
├── datasets/              # Raw and processed datasets
│   ├── visible/           # RGB imagery
│   ├── thermal/           # Thermal/IR imagery
│   ├── annotations/       # COCO-format labels
│   └── calibration/       # Camera calibration matrices
├── preprocessing/         # Data preprocessing pipeline
│   ├── alignment.py       # Cross-modal spatial alignment
│   ├── denoise.py         # Noise reduction filters
│   └── normalize.py       # Intensity/contrast normalization
├── models/                # Model architectures
│   ├── detector/          # Object detection heads
│   ├── fusion/            # Cross-modal fusion modules
│   ├── tracker/           # Tracking algorithms
│   └── risk_model/        # Threat reasoning networks
├── training/              # Training scripts
│   ├── train_detector.py  # Detection model training
│   ├── train_fusion.py    # Fusion module training
│   └── train_risk.py      # Risk model training
├── inference/             # Inference pipelines
│   ├── realtime.py        # Live camera inference
│   └── video.py           # Offline video processing
├── utils/                 # Utility scripts
│   └── check_gpu.py       # GPU/CUDA diagnostics
├── configs/               # Configuration files
│   └── default.yaml       # Default paths & hyperparameters
├── weights/               # Pretrained & trained weights
├── app/                   # FastAPI serving application
├── environment.yml        # Conda environment spec
└── requirements.txt       # Pip dependencies
```

---

## Quick Start

### 1. Create the Conda Environment

```bash
conda env create -f environment.yml
conda activate trinetra
```

Or install via pip:

```bash
pip install -r requirements.txt
```

### 2. Verify GPU Availability

```bash
python utils/check_gpu.py
```

### 3. Configure Paths

Edit `configs/default.yaml` to point to your dataset and weight directories.

---

## Phase Roadmap

| Phase | Description                              | Status       |
|-------|------------------------------------------|--------------|
| 1     | Project scaffold & environment setup     | ✅ Complete   |
| 2     | Data acquisition & dataset preparation   | ✅ Complete   |
| 3     | Preprocessing pipeline (TRC)             | 🔲 Planned   |
| 4     | Cross-modal fusion module                | 🔲 Planned   |
| 5     | Multi-scale object detection             | 🔲 Planned   |
| 6     | Tracking & behaviour analysis            | 🔲 Planned   |
| 7     | Hierarchical threat reasoning            | 🔲 Planned   |
| 8     | Explainable alert generation             | 🔲 Planned   |
| 9     | Real-time inference & deployment         | 🔲 Planned   |

---

## Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA ≥ 11.8 (RTX 3060+ recommended)
- **RAM**: 16 GB minimum, 32 GB recommended
- **Storage**: 50 GB+ for datasets and weights

---

## License

Proprietary — see `Copyright-TriNetra.pdf` for details.
