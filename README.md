# VIETNAMESE AUTOMATIC LICENSE PLATE RECOGNITION (ALPR) SYSTEM

An automated end-to-end system designed to detect and recognize both single-line and two-line Vietnamese vehicle license plates from static images, recorded video files, and real-time camera streams (Webcam / RTSP). The post-processing pipeline strictly adheres to current domestic regulations under Circular 79/2024/TT-BCA and Circular 51/2025/TT-BCA.

---

## 1. Team Members & Task Assignments

The development team comprises 4 members, subdivided into two operational groups based on pipeline boundaries:

| Full Name             | Student ID | Team Role                         | Primary Responsibilities                                                                                                                                                          |
| :-------------------- | :--------- | :-------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Phan Gia Kiet**     | 2411796    | Team 1 Lead                       | Detection dataset curation (Step 3), YOLO plate detection model training & ablation (Step 4), Report compilation.                                                                 |
| **Le Dang Khai**      | 2452509    | Team 1 Member (Git & Integration) | Video ingestion, ByteTrack multi-object tracking, end-to-end pipeline assembly, Streamlit demo application & parking lot management (Step 8).                                     |
| **Pham Lam Tuan Anh** | 2452078    | Team 2 Lead                       | Character dataset preparation, OCR training with YOLO 30-class detector (Method B) and CRNN + CTC (Method C) (Step 5), Slide coordination.                                        |
| **Nguyen Gia Bao**    | 2452127    | Team 2 Member (Evaluation)        | License plate rectification (CLAHE, perspective transform), PaddleOCR baseline (Method A), post-processing & voting logic (Step 6), system-wide quantitative evaluation (Step 7). |

---

## 2. Environment Setup

### System Prerequisites

- **Python**: Version `3.10` or higher
- **Operating System**: Linux / Ubuntu 22.04 LTS, Windows 11 (PowerShell or WSL2), or macOS
- **Hardware**: CPU (Intel/AMD) for standard inference; NVIDIA GPU (CUDA-compatible) recommended for offline training.

### Installation Steps

1. **Clone the repository:**

   ```bash
   git clone <REPOSITORY_URL>
   cd <REPOSITORY_NAME>
   ```

2. **Create and activate a virtual environment:**
   - _On Linux / macOS / WSL2:_
     ```bash
     python -m venv .venv
     source .venv/bin/activate
     ```
   - _On Windows (PowerShell):_
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```
     _(Note: If PowerShell restricts script execution, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` beforehand)._

3. **Install dependencies:**
   Upgrade `pip` and install all required libraries specified in `requirements.txt`:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify environment installation:**
   Run a quick inference check using the default pretrained YOLO weights:
   ```bash
   python -c "from ultralytics import YOLO; model = YOLO('yolo26n.pt'); print('Environment setup successful!')"
   ```

---

## 3. Usage & Execution Guide

### Automated Unit Testing

To validate the string normalization, character correction rules, regex compliance, and provincial code validation:

```bash
pytest tests/
```

### Running the Web Demo Application

Launch the interactive Streamlit user interface (supporting single image upload, video batch processing, live RTSP/Webcam stream, and database history inspection):

```bash
streamlit run app.py
```

### Video Stream Processing with ByteTrack

To execute the video tracking pipeline with multi-frame majority voting:

```bash
python stream_loop.py --source data/samples/test_video.mp4 --tracker bytetrack.yaml
```

---

## 4. Repository Structure

```text
├── .gitignore               # Ignored files (virtual envs, heavy weights, raw datasets)
├── README.md                # Project documentation and quickstart instructions
├── requirements.txt         # Core dependencies and package versions
├── schema.sql               # SQLite database relational schema (sessions, detections)
├── docs/
│   ├── interface.md         # Inter-group interface agreement (H1 milestone)
│   └── literature_review.md # Literature review and benchmark surveys
├── models/
│   ├── best.onnx            # Exported ONNX weights for fast CPU inference
│   └── .gitkeep
├── datasets/
│   ├── data.yaml            # YOLO detection configuration (plate_1line, plate_2line)
│   ├── data_chars.yaml      # YOLO OCR configuration (30 character classes)
│   └── .gitkeep
├── tests/
│   ├── test_postprocess.py  # Unit tests for post-processing and regex logic
│   └── .gitkeep
├── results/                 # Evaluation metric tables, confusion matrices, error logs
└── src/
    ├── detector.py          # YOLO plate detector and ByteTrack tracker interface
    ├── rectify.py           # Padding crop, perspective warp, and CLAHE enhancement
    ├── read_chars.py        # Character reading and line-sorting logic (Method B)
    ├── postprocess.py       # Syntax correction, regex matching, and PlateVoter class
    ├── province_codes.py    # Standard set of 81 Vietnam administrative province codes
    ├── storage.py           # SQLite database interaction module
    └── app.py               # Streamlit web application frontend
```

---

## 5. Branching Conventions

Collaboration follows strict branch separation:

- `main`: Production-ready code; protected branch (requires pull request review).
- `nhom1/integration`: Pipeline assembly, tracking, and demo interface.
- `nhom1/detector`: YOLO plate detection model development.
- `nhom2/ocr-training`: Character recognition training (Method B & Method C).
- `nhom2/postprocess-eval`: Image rectification, regex validation, and benchmark evaluation.
