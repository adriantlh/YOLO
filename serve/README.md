YOLO FastAPI Test Package

This folder provides a minimal, ready-to-run setup to test the FastAPI server that serves YOLO detections with item/defect fields.

Prerequisites
- Python 3.9+
- Install deps: `pip install -r requirements.txt`
- Weights: put your trained `.pt` into `serve/weights/` (or set `YOLO_WEIGHT_PATH=/full/path/to/model.pt`). Without weights, predictions are usually empty.

Quick Start
1) From the repo root, start the server:
   - `bash serve/start.sh` (uses the configs in `serve/config/` by default)

2) Send a test request:
   - JSON response: `bash serve/request_json.sh`
   - Annotated image: `bash serve/request_image.sh` (saves to `serve/out.png`)
   - Web UI: open `http://localhost:8000/ui/` in a browser

3) Predict from a URL (e.g., S3 presigned GET):
   - JSON: `curl -s -H "Accept: application/json" -X POST \
       -H 'Content-Type: application/json' \
       -d '{"url":"https://example-bucket.s3.amazonaws.com/key.png?X-Amz-Signature=..."}' \
       http://localhost:8000/predict_by_url`
   - Annotated image: `curl -s -o out.png -X POST \
       -H 'Content-Type: application/json' \
       'http://localhost:8000/predict_by_url?visualize=true' \
       -d '{"url":"https://.../image.jpg?X-Amz-Signature=..."}'`

Environment Variables (optional)
- `YOLO_MODEL_CFG`: Path to model config YAML (default: `serve/config/model.yaml`).
- `YOLO_WEIGHT_PATH`: Path to your model weights `.pt` file. If unset, `serve/start.sh` auto-picks from `serve/weights/*.pt` if present.
- `YOLO_DATA_YAML`: Path to dataset YAML containing `names` for readable class labels (defaults to `serve/config/data.yaml`).
- `YOLO_DEFECT_TYPES`: Path to Defect/Object taxonomy (defaults to `serve/config/Defect Types.txt`).

Model/Class Configuration
- Ensure the model’s class count matches your dataset. In `serve/config/model.yaml`, set `anchor.class_num` to your dataset class number (e.g., `7`).
- Ensure `serve/config/data.yaml` has matching `nc` and `names` length.

Item/Defect Mapping
- File is at `serve/config/Defect Types.txt` by default with lines:
  - `Object Types: Ceiling, Floor, Tile, ...`
  - `Defect Types: Hollow, Stained, Cracked, ...`
- The server maps the predicted class name into `item_name` and/or `defect_name` using simple heuristics (exact match, or split tokens like `Window_Cracked`).

API Endpoints
- `GET /healthz` → { "status": "ok" }
- `POST /predict` (multipart/form-data)
  - Field: `file` (image)
  - Query: `visualize=true` to return a PNG with drawn boxes.
  - JSON response fields:
    - `class`, `class_name` (label), `item_name`, `defect_name`
    - `x_min`, `y_min`, `x_max`, `y_max` (pixel coords)
    - `score` (confidence)
- `POST /predict_by_url` (application/json)
  - Body: `{ "url": "https://..." }` (HTTP URL or S3 presigned GET)
  - Query: `visualize=true` to return a PNG; otherwise JSON detections

Notes
- Run scripts from the repo root; they `cd` as needed.
- On server start, look for: `Using WEIGHTS=...`. If you see `No weights configured`, place a `.pt` in `serve/weights/` or set `YOLO_WEIGHT_PATH`.
- For Postman: use Body → form-data with key `file` (type File) for `/predict`, or Body → raw (JSON) with `{ "url": "..." }` for `/predict_by_url`.
- Android: upload directly to `/predict` (multipart) or upload to S3 then call `/predict_by_url` with a presigned GET URL.

Docker
- Build (from repo root):
  - `docker build -t yolo-serve .`
- Run (no weights, CPU):
  - `docker run --rm -p 8000:8000 yolo-serve`
- Run with local weights mounted:
  - `docker run --rm -p 8000:8000 \
      -v "$(pwd)/serve/weights:/app/serve/weights:ro" \
      yolo-serve`
  - Or point to an explicit path inside the container:
    - `docker run --rm -p 8000:8000 -e YOLO_WEIGHT_PATH=/app/serve/weights/model.pt \
        -v "$(pwd)/serve/weights:/app/serve/weights:ro" yolo-serve`
- Override configs (optional):
  - `-e YOLO_MODEL_CFG=serve/config/model.yaml`
  - `-e YOLO_DATA_YAML=serve/config/data.yaml`
  - `-e YOLO_DEFECT_TYPES="serve/config/Defect Types.txt"`
- Test from host:
  - `curl http://localhost:8000/healthz`
  - `curl -F "file=@demo/images/inference/image.png" http://localhost:8000/predict`
  - `curl -o out.png -F "file=@demo/images/inference/image.png" "http://localhost:8000/predict?visualize=true"`
  - Web UI: `http://localhost:8000/ui/`
  - Swagger UI: `http://localhost:8000/docs` (includes example payloads)
 - Healthcheck:
   - Containers include a healthcheck that polls `/healthz`. View status with `docker ps` (look for `(healthy)`), or details with `docker inspect yolo-serve`.

Docker (GPU) brief note
- For NVIDIA GPUs, use the NVIDIA Container Runtime and a CUDA base image. You’ll need a Torch build matching your CUDA version.
- Example outline only (not provided here): base `nvidia/cuda:12.x-runtime`, install Python, then `pip install torch==<cuda-matched> torchvision==<...>` and the rest of requirements; run with `--gpus all`.

GPU Dockerfile and Compose
- This repo includes a GPU-ready `Dockerfile.gpu` (CUDA 12.1 runtime) and a compose service `yolo-serve-gpu` under the `gpu` profile.
- Build and run with GPU via compose (interactive):
  - `docker compose --profile gpu run --rm --gpus all yolo-serve-gpu`
- Or run detached (host must default to NVIDIA runtime):
  - `COMPOSE_PROFILES=gpu docker compose up -d`
- Manual Docker (without compose):
  - `docker build -f Dockerfile.gpu -t yolo-serve:gpu .`
  - `docker run --rm --gpus all -p 8000:8000 -v "$(pwd)/serve/weights:/app/serve/weights:ro" yolo-serve:gpu`

Docker Compose
- Build and run (foreground):
  - `docker compose up --build`
- Run detached (background):
  - `docker compose up -d`
- Update and restart after changes:
  - `docker compose up -d --build`
- Stop:
  - `docker compose down`
- Weights mounting: compose mounts `./serve/weights` to `/app/serve/weights` (read-only) by default.
- Override via env:
  - Create a `.env` next to `docker-compose.yml` with lines like:
    - `YOLO_WEIGHT_PATH=/app/serve/weights/model.pt`
    - `YOLO_MODEL_CFG=serve/config/model.yaml`
    - `YOLO_DATA_YAML=serve/config/data.yaml`
    - `YOLO_DEFECT_TYPES=serve/config/Defect Types.txt`
  - A template is provided at `.env.example` — copy to `.env` and edit.
- GPU (quick option):
  - `docker compose run --rm --gpus all yolo-serve`
 - Healthcheck:
   - Compose services report health from `/healthz`. Check via `docker compose ps` or `docker inspect <container>`.

Postman
- Import the collection at `serve/YOLO_FastAPI.postman_collection.json` into Postman.
- Set collection variable `baseUrl` if not `http://localhost:8000`.
- Use the provided requests for JSON or annotated image results.
