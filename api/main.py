import os
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, Response
from omegaconf import OmegaConf
from PIL import Image
import requests

from yolo.config.config import NMSConfig
from yolo.model.yolo import create_model
from yolo.tools.data_augmentation import AugmentationComposer
from yolo.tools.drawer import draw_bboxes
from yolo.utils.bounding_box_utils import create_converter
from yolo.utils.model_utils import PostProcess


def _load_class_list() -> List[str]:
    """Load class names from dataset config if available.

    Priority:
    - `YOLO_DATA_YAML` env var if points to a YAML with `names` list (YOLOv5 format).
    - `serve/config/data.yaml` if present.
    - `./data.yaml` in repo root if it has `names`.
    - `yolo/config/dataset/coco.yaml` `class_list` as fallback.
    """
    # Try YOLOv5-style data.yaml
    candidates = []
    env_path = os.getenv("YOLO_DATA_YAML")
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path("serve/config/data.yaml"))
    candidates.append(Path("data.yaml"))

    for path in candidates:
        if path.exists():
            try:
                cfg = OmegaConf.load(str(path))
                names = cfg.get("names", None)
                if names and isinstance(names, (list, tuple)):
                    return list(names)
            except Exception:
                pass

    # Fallback to coco class_list
    coco_cfg_path = Path("yolo/config/dataset/coco.yaml")
    if coco_cfg_path.exists():
        cfg = OmegaConf.load(str(coco_cfg_path))
        class_list = cfg.get("class_list", None)
        if class_list:
            return list(class_list)
    return []


def _load_item_defect_taxonomy() -> Tuple[List[str], List[str]]:
    """Load object and defect type vocabularies if provided.

    Looks for `Defect Types.txt` in repo root with lines like:
    "Object Types: A, B, C" and "Defect Types: X, Y".
    """
    object_types: List[str] = []
    defect_types: List[str] = []
    env_txt = os.getenv("YOLO_DEFECT_TYPES", "")
    # Try env var, then repo root, then serve/config fallback
    candidates = []
    if env_txt:
        candidates.append(Path(env_txt))
    candidates.append(Path("Defect Types.txt"))
    candidates.append(Path("serve/config/Defect Types.txt"))
    txt_path = next((p for p in candidates if p.exists()), None)
    if not txt_path:
        return object_types, defect_types
    try:
        content = txt_path.read_text(encoding="utf-8")
        for line in content.splitlines():
            if ":" not in line:
                continue
            key, values = line.split(":", 1)
            items = [v.strip() for v in values.split(",") if v.strip()]
            if key.lower().startswith("object"):
                object_types = items
            elif key.lower().startswith("defect"):
                defect_types = items
    except Exception:
        pass
    return object_types, defect_types


def _class_to_item_defect(class_name: str, object_vocab: List[str], defect_vocab: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """Best-effort mapping of a single class name to (item_name, defect_name).

    Heuristics:
    - If class name contains a delimiter and pieces match vocabularies, split.
    - Else, if the whole class matches one vocab, assign to that field.
    - Case-insensitive comparison.
    """
    if not class_name:
        return None, None
    name_lower = class_name.lower()
    obj_set = {o.lower() for o in object_vocab}
    def_set = {d.lower() for d in defect_vocab}

    # Direct match
    if name_lower in obj_set:
        return class_name, None
    if name_lower in def_set:
        return None, class_name

    # Try to split on common delimiters
    for delim in ["_", "-", " ", "/"]:
        if delim in class_name:
            parts = [p.strip() for p in class_name.split(delim) if p.strip()]
            item_name = next((p for p in parts if p.lower() in obj_set), None)
            defect_name = next((p for p in parts if p.lower() in def_set), None)
            if item_name or defect_name:
                return item_name, defect_name
    return None, None


def load_model_and_processors():
    """Lazy-loads model and post-processing utilities for inference."""
    # Load a default model config from serve/config to align with house-defect setup
    model_cfg_path = os.getenv("YOLO_MODEL_CFG", "serve/config/model.yaml")
    model_cfg = OmegaConf.load(model_cfg_path)

    # Build model using weights if explicitly provided via env, otherwise
    # auto-detect a local weights file at weights/{model_name}.pt if present.
    weight_env = os.getenv("YOLO_WEIGHT_PATH", "")
    # `weight_path` can be a string path or False to disable pretrained weights
    if weight_env:
        weight_path = weight_env
    else:
        # Prefer a locally-downloaded weights file if available
        local_candidate = Path("weights") / f"{model_cfg.name}.pt"
        weight_path = str(local_candidate) if local_candidate.exists() else False
    model = create_model(model_cfg, weight_path=weight_path, class_num=getattr(model_cfg.anchor, "class_num", 80))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # Converter and post-process (NMS) setup
    image_size = [640, 640]
    converter = create_converter(model_cfg.name, model, model_cfg.anchor, image_size, device)
    nms_cfg = NMSConfig(min_confidence=0.5, min_iou=0.5, max_bbox=300)
    post_process = PostProcess(converter, nms_cfg)

    # Simple preprocessor (pad+resize to 640x640)
    preprocessor = AugmentationComposer([], image_size)

    # Class list and taxonomy for readable labels and mapping
    class_list = _load_class_list()
    object_types, defect_types = _load_item_defect_taxonomy()

    return model, device, post_process, preprocessor, model_cfg, class_list, object_types, defect_types


app = FastAPI(
    title="YOLO FastAPI",
    version="0.1.0",
    description="Run YOLO detections from uploaded images or URLs. Try the built-in web UI at /ui.",
)


@app.on_event("startup")
def _startup():
    (
        app.state.model,
        app.state.device,
        app.state.post_process,
        app.state.preprocessor,
        app.state.model_cfg,
        app.state.class_list,
        app.state.object_types,
        app.state.defect_types,
    ) = load_model_and_processors()

    # Mount static UI if available
    static_dir = Path("serve/static")
    if static_dir.exists():
        # Serves index.html at /ui/ and static assets under /ui
        app.mount("/ui", StaticFiles(directory=str(static_dir), html=True), name="ui")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


def _infer(image: Image.Image):
    model = app.state.model
    device = app.state.device
    post_process = app.state.post_process
    preprocessor = app.state.preprocessor

    # Ensure RGB
    image = image.convert("RGB")
    tensor, _, rev_tensor = preprocessor(image)
    tensor = tensor.unsqueeze(0).to(device)
    rev_tensor = rev_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        predictions = post_process(outputs, rev_tensor=rev_tensor)

    # predictions: List[Tensor] with shape Nx6 [cls, x1, y1, x2, y2, score]
    if not predictions:
        return []
    pred = predictions[0].detach().cpu()
    class_list: List[str] = getattr(app.state, "class_list", [])
    object_vocab: List[str] = getattr(app.state, "object_types", [])
    defect_vocab: List[str] = getattr(app.state, "defect_types", [])

    result = []
    for row in pred:
        cls_idx = int(row[0].item())
        cls_name = class_list[cls_idx] if 0 <= cls_idx < len(class_list) else None
        item_name, defect_name = _class_to_item_defect(cls_name or str(cls_idx), object_vocab, defect_vocab)
        result.append(
            {
                "class": cls_idx,
                "class_name": cls_name,
                "item_name": item_name,
                "defect_name": defect_name,
                "x_min": float(row[1].item()),
                "y_min": float(row[2].item()),
                "x_max": float(row[3].item()),
                "y_max": float(row[4].item()),
                "score": float(row[5].item()) if row.numel() > 5 else None,
            }
        )
    return result


@app.post(
    "/predict",
    summary="Predict from uploaded image",
    responses={
        200: {
            "description": "List of detections (JSON) or annotated PNG when visualize=true",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "class": 3,
                            "class_name": "Window_Cracked",
                            "item_name": "Window",
                            "defect_name": "Cracked",
                            "x_min": 42.3,
                            "y_min": 18.7,
                            "x_max": 210.5,
                            "y_max": 180.2,
                            "score": 0.91
                        }
                    ]
                },
                "image/png": {
                    "schema": {"type": "string", "format": "binary"}
                }
            },
        }
    },
)
async def predict(
    file: UploadFile = File(...),
    visualize: bool = Query(False, description="Return annotated image instead of JSON."),
):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    preds = _infer(image)

    if not visualize:
        return JSONResponse(preds)

    # Draw and return image
    idx2label = getattr(app.state, "class_list", [])
    drawn = draw_bboxes(
        image,
        [[p["class"], p["x_min"], p["y_min"], p["x_max"], p["y_max"], p.get("score", 0.0)] for p in preds],
        idx2label=idx2label if idx2label else None,
    )
    buf = BytesIO()
    drawn.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.post(
    "/predict_by_url",
    summary="Predict from a public or presigned image URL",
    responses={
        200: {
            "description": "List of detections (JSON) or annotated PNG when visualize=true",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "class": 1,
                            "class_name": "Tile_Stained",
                            "item_name": "Tile",
                            "defect_name": "Stained",
                            "x_min": 15.0,
                            "y_min": 22.0,
                            "x_max": 320.0,
                            "y_max": 240.0,
                            "score": 0.87
                        }
                    ]
                },
                "image/png": {
                    "schema": {"type": "string", "format": "binary"}
                }
            },
        }
    },
)
async def predict_by_url(
    payload: dict = Body(
        ...,
        description="JSON with field 'url' pointing to an image (HTTP/S3 presigned URL).",
        examples={
            "basic": {
                "summary": "HTTP image URL",
                "value": {"url": "https://example.com/image.jpg"}
            }
        },
    ),
    visualize: bool = Query(False, description="Return annotated image instead of JSON."),
):
    """Predict from a remote image URL (e.g., S3 presigned GET).

    Request body example:
      { "url": "https://.../image.png?X-Amz-Signature=..." }
    """
    url = payload.get("url")
    if not url or not isinstance(url, str):
        raise HTTPException(status_code=400, detail="Missing or invalid 'url' in JSON body")

    try:
        resp = requests.get(url, timeout=20)
        if resp.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Failed to fetch image: HTTP {resp.status_code}")
        image = Image.open(BytesIO(resp.content))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image URL or data: {e}")

    preds = _infer(image)

    if not visualize:
        return JSONResponse(preds)

    # Draw and return image
    idx2label = getattr(app.state, "class_list", [])
    drawn = draw_bboxes(
        image,
        [[p["class"], p["x_min"], p["y_min"], p["x_max"], p["y_max"], p.get("score", 0.0)] for p in preds],
        idx2label=idx2label if idx2label else None,
    )
    buf = BytesIO()
    drawn.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


# Convenience root endpoint
@app.get("/")
def root():
    return {
        "name": "YOLO FastAPI",
        "version": app.version if hasattr(app, "version") else "0.1.0",
        "endpoints": ["/healthz", "/predict"],
    }
