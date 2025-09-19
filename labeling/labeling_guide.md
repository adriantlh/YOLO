# Home Defect Detection — Labeling Guide (MVP)

This guide explains how to label near/far photo pairs for the MVP dataset and export COCO annotations compatible with this YOLO repo.

## Scope
- Task: Draw bounding boxes around visible defect regions and assign a combined class `object_defect`.
- Images: Two photos per scene — one Far and one Near.
- Dataset: Use the curated 24-class MVP schema in `yolo/config/dataset/home_mvp.yaml`.

## Class Schema (24 labels)
Use these labels exactly (case-sensitive):

- wall_cracked, wall_stained, wall_discoloured, wall_hollow
- ceiling_stained, ceiling_cracked, ceiling_discoloured, ceiling_hollow
- tile_cracked, tile_stained, tile_discoloured
- floor_cracked, floor_stained
- door_scratched, door_dented, door_discoloured
- sink_stained, sink_discoloured
- toilet_bowl_stained, toilet_bowl_cracked
- plaster_hollow, plaster_cracked
- handrail_dented, handrail_scratched

Notes:
- “tcrack” is merged into “cracked”.
- Use `discoloured` (British spelling) consistently.

## What To Label
- Draw a tight box around the defect area (not the whole object).
- If multiple defects are present, create multiple boxes.
- If the defect is only visible in one panel (Near or Far), label only that panel.
- Ignore non-defect regions, reflections, shadows unless they represent a defect class.

Ambiguities:
- Hairline vs. larger crack → both are `*_cracked`.
- Stain vs. discoloration: stain has stronger localized color or fluid pattern; otherwise `*_discoloured`.
- Dents/scratches typically apply to door/handrail/metallic surfaces.

## Near/Far Pairing
- Preferred filenames: `<pair_id>_far.jpg` and `<pair_id>_near.jpg`.
- Keep the same `pair_id` across both panels for consistency.

## Label Studio Setup
1) Install and start Label Studio:
   - `pip install label-studio`
   - `label-studio start`
2) Create New Project → Labeling Setup:
   - For single images: paste `labeling/labelstudio_home_mvp.xml`
   - For paired images: paste `labeling/labelstudio_home_mvp_pair.xml`
3) Import tasks:
   - For pairs, generate from CSV: `python labeling/make_labelstudio_tasks_from_csv.py examples/pairs_example.csv --out tasks.json`
   - Import `tasks.json` (fields required by the paired config: `image_far`, `image_near`, optional `pair_id`).

## Export To COCO
- In Label Studio, export “COCO (Object Detection)”.
- Save the resulting JSON (e.g., `~/exports/home_defects_coco.json`).

## Validate and Split (Train/Val)
- Place your original images in a local folder (or ensure accessible paths): e.g., `/data/images_raw`.
- Run the validator/splitter:

```
python labeling/validate_coco_and_split.py \
  --coco ~/exports/home_defects_coco.json \
  --dataset-yaml yolo/config/dataset/home_mvp.yaml \
  --out-dir data/home \
  --source-images /data/images_raw \
  --val-ratio 0.2
```

This will:
- Validate category names against `home_mvp.yaml`.
- Split by near/far pairs into train/val.
- Write COCO files under `data/home/annotations/`.
- Organize images under `data/home/images/train` and `data/home/images/val` (symlinks by default; use `--copy` to copy files).

## Train
```
python yolo/lazy.py task=train dataset=home_mvp model=v9-s name=home-mvp-v1 device=cuda task.data.batch_size=16
```

## Quality Checklist
- Boxes are tight around the defect, not the entire object.
- Correct class chosen; follow the schema strictly.
- Both panels labeled where the defect is visible.
- Ambiguous cases: leave a note in the `notes` field.
- Spot-check at least 10–20% of tasks for consistency.

## Contact
If a label is missing or unclear, note it and continue; we can iterate the schema after the first validation pass.

