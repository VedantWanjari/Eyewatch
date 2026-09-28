# Eyewatch

Lightweight FastAPI application for camera-based inference using the trained YOLO model at `trained_model/best.pt`.

## Run with Docker

Build:

```bash
docker build -t eyewatch-api .
```

Run:

```bash
docker run --rm -p 8000:8000 eyewatch-api
```

Open the browser UI at:

- `http://localhost:8000/`

## Camera UI usage

1. Click **Enable Camera** and allow browser camera access.
2. Click **Capture & Detect** to capture one frame and run inference.
3. Detection results are shown as a list and bounding boxes on the captured frame.

> Camera access usually requires **localhost** or **HTTPS** due to browser security rules.

## API

### `GET /health`

Returns service and model state.

Example response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": "trained_model/best.pt",
  "model_error": null
}
```

### `POST /predict`

Accepts `multipart/form-data` with a single `image` file.

Example response:

```json
{
  "detections": [
    {
      "class_id": 0,
      "class_name": "eye",
      "confidence": 0.93,
      "bbox": [43.2, 25.8, 190.4, 181.0]
    }
  ],
  "count": 1
}
```

## Configuration and runtime notes

- Model path is configurable with `MODEL_PATH` (default: `trained_model/best.pt`).
- Inference is executed on CPU (`device="cpu"`).
- Uploaded camera images are processed in memory only and are not persisted.
- Docker image installs CPU-only PyTorch wheels to keep runtime GPU-independent.
