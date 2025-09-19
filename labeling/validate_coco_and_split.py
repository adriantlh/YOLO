#!/usr/bin/env python3
"""
Validate a COCO annotations file against a dataset YAML class list,
optionally split into train/val by near/far pairs, and organize images.

Outputs two COCO files:
  <out_dir>/annotations/instances_train.json
  <out_dir>/annotations/instances_val.json

Optionally creates image trees:
  <out_dir>/images/train
  <out_dir>/images/val
with symlinks or copies from a source images directory.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def try_load_class_list(yaml_path: Path) -> Optional[List[str]]:
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(yaml_path.read_text())
        cl = data.get("class_list")
        if isinstance(cl, list) and all(isinstance(x, str) for x in cl):
            return cl
        return None
    except Exception:
        # Fallback: naive parse for a bracketed inline list
        text = yaml_path.read_text()
        m = re.search(r"class_list\s*:\s*\[(.*?)\]", text, flags=re.S)
        if not m:
            return None
        inner = m.group(1)
        items = [x.strip().strip(",").strip().strip("'\"") for x in inner.split("\n") if x.strip()]
        # Merge possible trailing commas across lines
        flat: List[str] = []
        for item in items:
            parts = [p.strip() for p in item.split(",") if p.strip()]
            flat.extend(parts)
        return flat or None


def load_coco(path: Path) -> Dict:
    with path.open("r") as f:
        return json.load(f)


def validate_categories(coco: Dict, allowed: Optional[List[str]]) -> Tuple[List[str], List[str]]:
    names = [c.get("name") for c in coco.get("categories", [])]
    names = [n for n in names if isinstance(n, str)]
    if not allowed:
        return names, []
    unknown = [n for n in names if n not in allowed]
    return names, unknown


def infer_pair_id(stem: str, near_suffix: str, far_suffix: str) -> str:
    if stem.endswith(near_suffix):
        return stem[: -len(near_suffix)]
    if stem.endswith(far_suffix):
        return stem[: -len(far_suffix)]
    return stem


def split_by_pairs(
    coco: Dict,
    val_ratio: float,
    seed: int,
    near_suffix: str,
    far_suffix: str,
) -> Tuple[set, set]:
    images = coco.get("images", [])
    id_to_stem = {img["id"]: Path(img["file_name"]).stem for img in images}
    groups: Dict[str, List[int]] = {}
    for img_id, stem in id_to_stem.items():
        pid = infer_pair_id(stem, near_suffix, far_suffix)
        groups.setdefault(pid, []).append(img_id)

    group_ids = list(groups.keys())
    random.Random(seed).shuffle(group_ids)
    cut = int(len(group_ids) * (1 - val_ratio))
    train_groups = set(group_ids[:cut])
    val_groups = set(group_ids[cut:])

    train_img_ids = {img_id for pid in train_groups for img_id in groups[pid]}
    val_img_ids = {img_id for pid in val_groups for img_id in groups[pid]}
    return train_img_ids, val_img_ids


def filter_coco(coco: Dict, keep_image_ids: set) -> Dict:
    keep_images = [img for img in coco.get("images", []) if img["id"] in keep_image_ids]
    keep_ids = {img["id"] for img in keep_images}
    keep_ann = [ann for ann in coco.get("annotations", []) if ann["image_id"] in keep_ids]
    # Keep categories unchanged to preserve id mapping
    return {"images": keep_images, "annotations": keep_ann, "categories": coco.get("categories", [])}


def index_source_images(src_dir: Path) -> Dict[str, Path]:
    idx: Dict[str, Path] = {}
    for p in src_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            idx.setdefault(p.stem, p)
    return idx


def materialize_images(
    coco: Dict,
    img_id_set: set,
    src_index: Dict[str, Path],
    out_dir: Path,
    copy: bool = False,
) -> List[Tuple[int, Path, Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ops: List[Tuple[int, Path, Path]] = []
    for img in coco.get("images", []):
        if img["id"] not in img_id_set:
            continue
        stem = Path(img["file_name"]).stem
        if stem not in src_index:
            raise FileNotFoundError(f"Cannot locate source image for stem '{stem}' in {src_index and list(src_index.values())[0].parents[1]}")
        src = src_index[stem]
        dst = out_dir / f"{stem}{src.suffix.lower()}"
        if dst.exists():
            ops.append((img["id"], src, dst))
            continue
        if copy:
            shutil.copy2(src, dst)
        else:
            try:
                dst.symlink_to(src.resolve())
            except FileExistsError:
                pass
            except OSError:
                # Fallback to copy if symlink not permitted
                shutil.copy2(src, dst)
        ops.append((img["id"], src, dst))
    return ops


def main():
    ap = argparse.ArgumentParser(description="Validate COCO, split train/val, and organize images.")
    ap.add_argument("--coco", required=True, help="Path to COCO JSON export from Label Studio")
    ap.add_argument(
        "--dataset-yaml",
        default="yolo/config/dataset/home_mvp.yaml",
        help="Dataset YAML with class_list (for validation)",
    )
    ap.add_argument("--out-dir", default="data/home", help="Output dataset root (images/, annotations/)")
    ap.add_argument("--source-images", help="Directory containing original images to link/copy from")
    ap.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio (by pair)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for split")
    ap.add_argument("--near-suffix", default="_near", help="Suffix in filename stem for near images")
    ap.add_argument("--far-suffix", default="_far", help="Suffix in filename stem for far images")
    ap.add_argument("--copy", action="store_true", help="Copy images instead of symlinking")
    args = ap.parse_args()

    coco_path = Path(args.coco)
    out_root = Path(args.out_dir)
    out_anno = out_root / "annotations"
    out_img_train = out_root / "images" / "train"
    out_img_val = out_root / "images" / "val"

    coco = load_coco(coco_path)
    allowed = try_load_class_list(Path(args.dataset_yaml))
    names, unknown = validate_categories(coco, allowed)
    if unknown:
        raise SystemExit(
            "Found categories not present in dataset YAML class_list: " + ", ".join(sorted(set(unknown)))
        )

    train_ids, val_ids = split_by_pairs(coco, args.val_ratio, args.seed, args.near_suffix, args.far_suffix)

    # Save annotations
    out_anno.mkdir(parents=True, exist_ok=True)
    with (out_anno / "instances_train.json").open("w") as f:
        json.dump(filter_coco(coco, train_ids), f)
    with (out_anno / "instances_val.json").open("w") as f:
        json.dump(filter_coco(coco, val_ids), f)

    # Organize images if source provided
    if args.source_images:
        src_dir = Path(args.source_images)
        idx = index_source_images(src_dir)
        materialize_images(coco, train_ids, idx, out_img_train, copy=args.copy)
        materialize_images(coco, val_ids, idx, out_img_val, copy=args.copy)

    print("Done. Summary:")
    print(f"- Categories: {len(names)} (validated)")
    print(f"- Train images: {len(train_ids)}; Val images: {len(val_ids)}")
    print(f"- Annotations written to: {out_anno}")
    if args.source_images:
        print(f"- Images organized under: {out_root / 'images'}")


if __name__ == "__main__":
    main()

