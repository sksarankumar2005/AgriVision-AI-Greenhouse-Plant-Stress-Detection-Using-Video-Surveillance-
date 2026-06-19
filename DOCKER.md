# AgriVision AI — Docker (minimal image)

## What goes in the image

| Included | Excluded (saves GB) |
|----------|---------------------|
| `app.py`, `predict.py` | `dataset/` |
| `templates/` | `yolo11n.pt`, `yolo26n.pt` |
| `best.pt` (~5 MB) | All `.mp4` videos |
| Flask + Ultralytics | `.venv/`, training scripts |

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- Trained weights: `runs/detect/train/weights/best.pt`

## Quick start (build + run)

```powershell
cd c:\Users\SARANKUMAR\Downloads\AgriVision_AI

# Build, test health endpoint, run
.\docker-build.ps1

# Start with Docker Compose
docker compose up -d
```

Open **http://localhost:5000**

## Push to Docker Hub (secure)

1. Create a repo on [hub.docker.com](https://hub.docker.com) (e.g. `youruser/agrivision-ai`).

2. Login (password/token — never commit credentials):

```powershell
docker login
```

3. Build, test, and push:

```powershell
.\docker-build.ps1 -Registry YOUR_DOCKERHUB_USERNAME -Push
```

Image will be: `YOUR_DOCKERHUB_USERNAME/agrivision-ai:latest`

## Manual commands

```powershell
docker build -t agrivision-ai:latest .
docker run -p 5000:5000 agrivision-ai:latest
docker tag agrivision-ai:latest YOUR_USER/agrivision-ai:latest
docker push YOUR_USER/agrivision-ai:latest
```

## Security notes

- Container runs as non-root user `appuser`
- No secrets in the image; use `docker login` locally only
- Do not copy `.env` files into the image
- Upload size limit: 200 MB (set in `app.py`)

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `best.pt not found` | Run `train.py` or copy weights into `runs/detect/train/weights/` |
| Build slow | First build downloads PyTorch (~1–2 GB); later builds use cache |
| Health check slow | First start loads model ~1–3 min; wait for `healthy` |
| Port in use | Stop local `python app.py` or change port in `docker-compose.yml` |
