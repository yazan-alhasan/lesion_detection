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

Create and activate a virtual environment, then install the dependencies:

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