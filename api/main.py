import io
import os
from typing import Any, Callable

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

ModelLoader = Callable[[str], Any]
MODEL_PATH = os.getenv("MODEL_PATH", "trained_model/best.pt")


def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def create_app(model_loader: ModelLoader = load_model) -> FastAPI:
    app = FastAPI(title="EyeWatch Camera Inference API")
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.on_event("startup")
    def startup_event() -> None:
        app.state.model = None
        app.state.model_error = None
        try:
            app.state.model = model_loader(MODEL_PATH)
        except Exception as exc:  # pragma: no cover - defensive path
            app.state.model_error = str(exc)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(os.path.join(static_dir, "index.html"))

    @app.get("/health")
    def health() -> dict[str, Any]:
        model_loaded = app.state.model is not None
        return {
            "status": "ok" if model_loaded else "degraded",
            "model_loaded": model_loaded,
            "model_path": MODEL_PATH,
            "model_error": app.state.model_error,
        }

    @app.post("/predict")
    async def predict(image: UploadFile = File(...)) -> dict[str, Any]:
        model = app.state.model
        if model is None:
            raise HTTPException(status_code=503, detail="Model is not available")

        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image")

        raw = await image.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Uploaded image is empty")

        try:
            pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid image upload: {exc}") from exc

        results = model.predict(source=pil_image, verbose=False, device="cpu")
        detections: list[dict[str, Any]] = []

        for result in results:
            names = result.names or {}
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls.item())
                if isinstance(names, dict):
                    class_name = names.get(class_id, str(class_id))
                elif isinstance(names, (list, tuple)) and class_id < len(names):
                    class_name = str(names[class_id])
                else:
                    class_name = str(class_id)
                xyxy = [float(value) for value in box.xyxy[0].tolist()]
                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": float(box.conf.item()),
                        "bbox": xyxy,
                    }
                )

        return {"detections": detections, "count": len(detections)}

    return app


app = create_app()
