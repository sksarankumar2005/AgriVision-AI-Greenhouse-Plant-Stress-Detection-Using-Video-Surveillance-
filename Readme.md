# 🌱 AgriVision AI - Greenhouse Plant Stress Detection

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0+-black?logo=flask)
![Ultralytics YOLO](https://img.shields.io/badge/YOLO-Ultralytics-yellow)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)
![License](https://img.shields.io/badge/License-MIT-green)

AgriVision AI is an intelligent computer vision web application designed to detect stress in greenhouse plants. Built with **Flask** and powered by state-of-the-art **YOLO** object detection models, it analyzes both images and videos to identify signs of crop stress, enabling early intervention and yield optimization.

---

## ✨ Key Features

- **Multi-Media Analysis:** Supports both image (JPG, PNG, WEBP) and video (MP4, AVI, MOV, MKV) processing up to 200MB.
- **Adjustable Inference:** Fine-tune confidence thresholds (`conf`) and video processing frame strides (`frame_stride`) on the fly.
- **RESTful API:** Clean API endpoints for UI integration or programmatic inference.
- **Model Pipeline Included:** Includes complete scripts for data preparation, extraction, labeling, and YOLO model training.
- **Dockerized Environment:** Production-ready containerization via Docker and Docker Compose with health checks and persistent volumes.
- **Modern Web Interface:** Built-in responsive HTML UI for easy drag-and-drop media uploads.

---

## 🏗️ Tech Stack

- **Backend:** Python 3.11, Flask, Flask-CORS, Werkzeug
- **AI & Vision:** Ultralytics (YOLO), OpenCV Headless, ImageIO FFmpeg
- **Frontend:** HTML5, CSS3, Vanilla JavaScript
- **Deployment:** Docker, Docker Compose

---

## 📁 Project Structure

```text
AgriVision_AI/
├── app.py                      # Main Flask application entry point
├── predict.py                  # Core inference logic for images and videos
├── health_check.py             # Server health monitoring script
├── Dockerfile                  # Container build instructions
├── docker-compose.yml          # Multi-container orchestration
├── requirements.txt            # Local Python dependencies
├── requirements-docker.txt     # Container-optimized dependencies
├── templates/                  # HTML templates for the web UI
│   └── index.html
├── static/                     # Processed media outputs and frontend assets
├── runs/detect/train/weights/  # Trained YOLO model weights
│   └── best.pt
├── train.py                    # Script to trigger YOLO model training
├── prepare_training.py         # Data preparation utilities
└── extract_and_label.py        # Video frame extraction and dataset creation
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+** (If running locally)
- **Docker & Docker Compose** (If running via containers)
- **Git**

### Option A: Running Locally (Native)

1. **Clone the repository and navigate to the directory:**
   ```bash
   git clone <your-repo-url>
   cd AgriVision_AI
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows
   python -m venv .venv
   .\.venv\Scripts\activate

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the server:**
   ```bash
   # Using Python
   python app.py
   
   # Or using the provided PowerShell script (Windows)
   .\start.ps1
   ```

5. **Access the Web UI:**
   Navigate to `http://127.0.0.1:5000` in your browser.

### Option B: Running with Docker (Recommended)

Docker provides an isolated, reproducible environment.

1. **Build and start the container:**
   ```bash
   docker-compose up -d
   ```
   *(Windows users can optionally execute `.\docker-build.ps1`)*

2. **Verify container health:**
   ```bash
   docker ps
   ```
   *The container includes a health check mechanism to verify API availability.*

3. **Access the Web UI:**
   Navigate to `http://localhost:5000` in your browser.

4. **Stop the application:**
   ```bash
   docker-compose down
   ```

---

## 🔌 API Reference

The application exposes the following endpoints (v2.0.0):

### `GET /`
Returns the main web interface (HTML).

### `GET /health`
Returns the operational status of the server and model configuration.
**Response:**
```json
{
  "status": "ok",
  "app_version": "2.0.0",
  "model": "runs/detect/train/weights/best.pt"
}
```

### `POST /predict`
Performs object detection on uploaded media.
- **Payload (multipart/form-data):**
  - `file`: The image or video file (required).
  - `conf`: Confidence threshold float between 0.05 - 0.95 (optional, default: 0.15).
  - `frame_stride`: Video frame stride int between 1 - 10 (optional, default: 5).
- **Success Response:**
  ```json
  {
    "success": true,
    "media_type": "image",
    "filename": "upload.jpg",
    "summary": "...",
    "detection_count": 3,
    "has_stress": true
  }
  ```

---

## 🧠 Model Training Pipeline

If you wish to retrain or fine-tune the model with your own dataset, the project includes an end-to-end data pipeline:

1. **Extract & Label:** Use `extract_and_label.py` to parse video datasets and extract individual frames for annotation.
2. **Prepare Dataset:** Run `prepare_training.py` to format the dataset directory structure and `.yaml` config required by Ultralytics YOLO.
3. **Train:** Execute `train.py` to initiate the YOLO model training sequence. The best weights will automatically be saved to `runs/detect/train/weights/best.pt`.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---
*Built with ❤️ for Agricultural AI Innovation.*