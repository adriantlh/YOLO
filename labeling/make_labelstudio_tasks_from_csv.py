#!/usr/bin/env python3
"""
Generate Label Studio task JSON (for the near/far pair config) from a CSV.

CSV columns:
  pair_id,image_far,image_near

Outputs a JSON array of tasks with keys used by labelstudio_home_mvp_pair.xml:
  [{"pair_id": "scene123", "image_far": "s3://.../scene123_far.jpg", "image_near": "s3://.../scene123_near.jpg"}, ...]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Create Label Studio tasks JSON from CSV (near/far pairs)")
    ap.add_argument("csv", help="Input CSV with columns: pair_id,image_far,image_near")
    ap.add_argument("--out", default="tasks.json", help="Output JSON path")
    args = ap.parse_args()

    rows = []
    with open(args.csv, newline="") as f:
        reader = csv.DictReader(f)
        required = {"pair_id", "image_far", "image_near"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"CSV missing required columns: {', '.join(sorted(missing))}")
        for r in reader:
            rows.append({
                "pair_id": r["pair_id"],
                "image_far": r["image_far"],
                "image_near": r["image_near"],
            })

    out = Path(args.out)
    out.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {out} with {len(rows)} tasks")


if __name__ == "__main__":
    main()

