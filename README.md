# VinDr-Mammo Lesion Detection

## Project Structure

```text
configs/                  Dataset and training configuration files
data/raw/                 User-provided annotations and DICOM images
data/processed/           Generated PNG images, YOLO labels, and metadata
src/data/                 Data inspection, subset creation, conversion, and splitting scripts
src/training/             YOLOv8 and RetinaNet training entrypoints
src/evaluation/           YOLOv8, RetinaNet, and comparison evaluation scripts
src/visualization/        Bounding-box visualization script
src/utils/                Shared path, image, bbox, and GPU utilities
runs/                     Generated metrics, checkpoints, and visual outputs
requirements.txt          Python dependencies
```

## Setup

This project has been checked with Python 3.9.13. Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

To check GPU availability:

```bash
python src/utils/check_gpu.py
```

## Data Preparation

Inspect the finding annotation file:

```bash
python src/data/inspect_annotations.py --finding_csv data/raw/finding_annotations.csv --output runs/annotation_summary.txt
```

### Operation 1: Create a Selected-class YOLO Dataset

This operation only creates the dataset. It does not train a model.

To prepare only one category, convert only matching DICOM files, and crop the breast region:

First check that the selected image IDs exist in your DICOM folder:

```bash
python src/data/check_missing_images.py --annotations data/raw/finding_annotations.csv --input_images_path D:\images --target_finding Mass --missing_output data/dataset_mass_cropped/missing_images.csv
```

Then create the dataset:

```bash
python src/data/prepare_selected_yolo_dataset.py --annotations data/raw/finding_annotations.csv --input_images_path D:\images --output_dir data/dataset_mass_cropped_clahe --target_finding Mass --train_percent 80 --val_percent 10 --test_percent 10 --split_by exam --image_size 1024 --apply_clahe --clahe_clip_limit 2.0 --clahe_tile_grid_size 8 --save_debug_samples
```

This writes `data/dataset_mass_cropped_clahe/dataset.yaml`, `classes.txt`, `dataset_stats.json`, split image/label folders, and optional debug samples. Labels contain only class id `0`, with class name `mass`. CLAHE is applied after breast cropping, so bounding boxes are not changed after CLAHE.

To create the separate crop + conservative MLO pectoral attenuation + CLAHE dataset:

```bash
python src/data/prepare_selected_yolo_dataset.py --annotations data/raw/finding_annotations.csv --input_images_path D:\images --output_dir data/dataset_mass_cropped_pectoral_attenuate_clahe --target_finding Mass --train_percent 80 --val_percent 10 --test_percent 10 --split_by exam --image_size 1024 --suppress_pectoral --pectoral_suppress_mode attenuate --pectoral_attenuation_factor 0.5 --pectoral_intensity_percentile 90 --pectoral_min_area_ratio 0.005 --pectoral_max_area_ratio 0.20 --pectoral_max_width_ratio 0.35 --pectoral_max_height_ratio 0.50 --pectoral_max_y_ratio 0.60 --pectoral_order before_clahe --apply_clahe --clahe_clip_limit 2.0 --clahe_tile_grid_size 8 --pectoral_debug --save_debug_samples
```

Suppression is gated by `view_position` and only runs for MLO images. Invalid masks are skipped without changing the image. Use `--pectoral_order before_clahe` for crop -> attenuation -> CLAHE, or write to a different output directory with `--pectoral_order after_clahe` for crop -> CLAHE -> attenuation. Results include `pectoral_log.csv`, pectoral counts in `dataset_stats.json`, and five intermediate images per sampled case under `debug_pectoral/`.

To select more than one category and merge all selected boxes into one YOLO class named `lesion`:

First check the selected images:

```bash
python src/data/check_missing_images.py --annotations data/raw/finding_annotations.csv --input_images_path D:\images --target_findings Mass "Suspicious Calcification" "Architectural Distortion" --missing_output data/dataset_lesion_cropped/missing_images.csv
```

Then create the dataset:

```bash
python src/data/prepare_selected_yolo_dataset.py --annotations data/raw/finding_annotations.csv --input_images_path D:\images --output_dir data/dataset_lesion_cropped_clahe --target_findings Mass "Suspicious Calcification" "Architectural Distortion" --train_percent 80 --val_percent 10 --test_percent 10 --split_by exam --image_size 1024 --apply_clahe --clahe_clip_limit 2.0 --clahe_tile_grid_size 8 --save_debug_samples
```

You can pass 2, 4, 5, or more category names after `--target_findings`. Images without at least one selected category are excluded before DICOM conversion.

`--input_images_path` can point to any folder that contains the DICOM files. The script searches inside it recursively and matches files by `image_id`, so the DICOMs do not have to be moved into `data/raw/images`.

Split values can be percentages (`--train_percent 80 --val_percent 10 --test_percent 10`) or ratios (`--train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1`).

By default, dataset creation uses `--split_by exam`, which means all DICOM images from the same input exam folder stay in exactly one split. Use `--split_by study_id` only if you want to group by the annotation `study_id` instead.

If you already have the cropped PNG YOLO dataset and only want to create the CLAHE version, run:

```bash
python src/data/apply_clahe_to_yolo_dataset.py --input_dataset data/dataset_mass_cropped --output_dataset data/dataset_mass_cropped_clahe --clahe_clip_limit 2.0 --clahe_tile_grid_size 8 --save_debug_samples
```

### Operation 2: Train YOLO

After Operation 1 finishes, train from the generated dataset YAML:

```bash
python src/training/train_yolo.py --data data/dataset_mass_cropped_clahe/dataset.yaml --model yolov8n.pt --imgsz 640 --epochs 100 --batch 8 --project runs/yolo --name mass_cropped_clahe_yolov8n --device 0 --workers 0 --single_cls
```

For the multi-category merged dataset:

```bash
python src/training/train_yolo.py --data data/dataset_lesion_cropped_clahe/dataset.yaml --model yolov8n.pt --imgsz 640 --epochs 100 --batch 8 --project runs/yolo --name lesion_cropped_clahe_yolov8n --device 0 --workers 0 --single_cls
```

Create a subset from locally available DICOM images:

```bash
python src/data/make_subset.py --finding_csv data/raw/finding_annotations.csv --output_csv data/subset/subset_image_ids.csv --images_root data/raw/images --all_available
```

Check for missing selected images:

```bash
python src/data/check_missing_images.py --subset_csv data/subset/subset_image_ids.csv --images_root data/raw/images --missing_output data/subset/missing_images.csv
```

Convert DICOM files to PNG:

```bash
python src/data/dicom_to_png.py --subset_csv data/subset/subset_image_ids.csv --images_root data/raw/images --output_dir data/processed/images/all --metadata_output data/processed/image_metadata.csv --image_size 1024 --apply_clahe
```

Generate single-class YOLO labels:

```bash
python src/data/convert_to_yolo_single_class.py --finding_csv data/raw/finding_annotations.csv --metadata_csv data/processed/image_metadata.csv --labels_output_dir data/processed/labels/all --classes_output data/processed/classes.txt --mapping_output data/processed/class_mapping.json
```

Create train, validation, and test folders:

```bash
python src/data/split_dataset.py --metadata_csv data/processed/image_metadata.csv --subset_csv data/subset/subset_image_ids.csv --images_all data/processed/images/all --labels_all data/processed/labels/all --output_root data/processed --val_ratio 0.2 --test_ratio 0.15 --random_seed 42
```

## Visualization

Visualize generated YOLO labels before training:

```bash
python src/visualization/visualize_bboxes.py --images_dir data/processed/images/train --labels_dir data/processed/labels/train --classes_file data/processed/classes.txt --output_dir runs/visualization_single_class --num_samples 30
```

Review the generated images in `runs/visualization_single_class/` to confirm that bounding boxes align with lesions.

## YOLOv8 Training

Train the single-class YOLOv8 model:

```bash
python src/training/train_yolo.py --data configs/dataset_single_class.yaml --model yolov8n.pt --imgsz 640 --epochs 80 --batch 2 --project runs/yolo --name vindr_single_class_yolov8n_gpu_640 --device 0 --workers 0 --single_cls
```

The script saves metrics to the YOLO run directory. Ultralytics also writes `results.csv` in the same run folder.

Evaluate trained YOLO weights:

```bash
python src/evaluation/evaluate_yolo.py --data configs/dataset_single_class.yaml --weights runs/yolo/vindr_single_class_yolov8n_gpu_640/weights/best.pt --imgsz 640 --device 0 --output runs/yolo/evaluation_metrics.json
```

## RetinaNet Training

The RetinaNet implementation uses the same processed images and YOLO label files as YOLOv8. The dataset loader in `src/data/yolo_detection_dataset.py` converts YOLO labels into torchvision detection targets.

Train RetinaNet:

```bash
python src/training/train_retinanet.py --config configs/train_retinanet.yaml
```

Evaluate the best RetinaNet checkpoint:

```bash
python src/evaluation/evaluate_retinanet.py --checkpoint runs/retinanet/vindr_single_class_retinanet_resnet50/checkpoints/best.pt --images_dir data/processed/images/test --labels_dir data/processed/labels/test --classes_file data/processed/classes.txt --imgsz 640 --device 0 --output runs/retinanet/vindr_single_class_retinanet_resnet50/test_metrics.json
```

## Faster R-CNN Training

Train Faster R-CNN directly from an existing YOLO-format dataset:

```bash
python src/training/train_faster_rcnn.py --data_root data/dataset_mass_cropped_clahe --model fasterrcnn_resnet50_fpn --epochs 100 --batch 2 --lr 0.005 --weight_decay 0.0005 --momentum 0.9 --device cuda --workers 2 --cache ram --patience 15 --min_delta 0.001 --save_period 10 --early_stop_metric map50 --score_thresh 0.25 --predict_every 10 --predict_samples 30 --output_dir runs/faster_rcnn/mass_fasterrcnn_resnet50_clahe --amp
```

If GPU memory is limited, use the lighter model:

```bash
python src/training/train_faster_rcnn.py --data_root data/dataset_mass_cropped_clahe --model fasterrcnn_mobilenet_v3_large_fpn --epochs 100 --batch 1 --lr 0.005 --device cuda --workers 2 --cache none --patience 15 --save_period 10 --early_stop_metric map50 --score_thresh 0.25 --output_dir runs/faster_rcnn/mass_fasterrcnn_mobilenet_clahe --amp
```

Faster R-CNN outputs are saved under `runs/faster_rcnn/.../`, including `weights/best.pt`, `weights/last.pt`, `metrics.csv`, `training_summary.json`, and prediction images.

## Compare Results

Create a comparison table from YOLOv8 and RetinaNet metrics:

```bash
python src/evaluation/compare_results.py --yolo_metrics runs/yolo/vindr_single_class_yolov8n_gpu_640/results.csv --retinanet_metrics runs/retinanet/vindr_single_class_retinanet_resnet50/test_metrics.json --output_csv runs/comparison_results.csv --output_md runs/comparison_results.md
```

## Configuration Files

```text
configs/dataset.yaml                 Multi-class YOLO dataset config
configs/dataset_single_class.yaml    Single-class YOLO dataset config
configs/train_yolo_single_class.yaml Single-class YOLO training settings
configs/train_retinanet.yaml         RetinaNet training settings
```

The main experiment uses `configs/dataset_single_class.yaml`, where all lesion categories are merged into one class named `Lesion`.
