"""
Train YOLO v9-c on the dataset under `serve/Data to train` and export weights for serving.

Usage:
  python examples/train_serve_dataset.py \
      --run-name house-defect-v9c \
      --epochs 50 \
      --batch 8 \
      --img 640 640 \
      [--device cuda|cpu]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def ensure_filelists(dataset_root: Path) -> None:
    def write_file_list(phase: str):
        imgdir = dataset_root / phase / "images"
        if not imgdir.is_dir():
            print(f"Skip {phase}: missing {imgdir}")
            return
        rels = []
        for p in sorted(imgdir.glob("*")):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                rels.append(str(p.relative_to(dataset_root)))
        out = dataset_root / f"{phase}.txt"
        out.write_text("\n".join(rels))
        print(f"Wrote {out} ({len(rels)} images)")

    for ph in ("train", "valid", "test"):
        write_file_list(ph)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="house-defect-v9c")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--img", nargs=2, type=int, default=[640, 640])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    root = Path.cwd()
    ds = root / "serve" / "Data to train"
    if not ds.exists():
        print(f"Dataset not found: {ds}")
        return 1

    # Install deps if needed (use system pip)
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    except subprocess.CalledProcessError:
        print("Warning: failed to install requirements. Continuing…")

    # Clean incompatible caches (format differs across Python versions)
    for p in ds.glob("*.pache"):
        try:
            p.unlink()
            print("Removed stale cache:", p)
        except Exception:
            pass
    ensure_filelists(ds)
    (root / "serve" / "weights").mkdir(parents=True, exist_ok=True)

    img_size_arg = f"image_size=[{args.img[0]},{args.img[1]}]"
    cmd = [
        sys.executable,
        "yolo/lazy.py",
        f"name={args.run_name}",
        "task=train",
        "dataset=serve-data",
        "model=v9-c",
        "weight=weights/v9-c.pt",
        f"task.epoch={args.epochs}",
        f"task.data.batch_size={args.batch}",
        "task.data.cpu_num=4",
        img_size_arg,
        f"device={args.device}",
    ]
    print("Running:", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        print("Training failed with return code:", rc)
        return rc

    run_dir = Path("runs/train") / args.run_name / "weights"
    best = run_dir / "best.pt"
    last = run_dir / "last.pt"
    target = Path("serve/weights/model.pt")
    if best.exists():
        target.write_bytes(best.read_bytes())
        print("Copied best ->", target)
    elif last.exists():
        target.write_bytes(last.read_bytes())
        print("Copied last ->", target)
    else:
        print("Did not find exported weights in:", run_dir)
        return 2

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
