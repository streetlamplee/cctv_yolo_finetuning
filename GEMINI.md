# Gemini Project Analysis: YOLOv8 Fine-Tuning and Quantization

## 1. Project Overview

This project is a complete pipeline for fine-tuning a `yolov8n` object detection model on a custom dataset. The process includes training, exporting the model to ONNX format, and performing post-training static INT8 quantization.

## 2. Core Technologies

- **Model Framework**: `ultralytics` (YOLOv8)
- **Inference & Quantization**: `onnx`, `onnxruntime`, `onnxruntime-gpu`
- **Data Handling**: `opencv-python`, `pyyaml`
- **ONNX Optimization**: `onnxslim`

## 3. Directory and File Structure

- **/src**: Contains the primary source code.
  - `main.py`: The main script that orchestrates the entire train-export-quantize pipeline.
  - `predict.py`: A script for running inference using a trained `.pt` model.
  - **/src/quantize**: Sub-package for model quantization.
    - `quantize_pt.py`: A standalone script to convert a `.pt` model to a quantized INT8 ONNX model.
    - `yoloCalibDataset.py`: A custom data reader that provides calibration data for static quantization.
- **/data**: Should contain the `dataset.yaml` file which defines dataset paths and class information.
- **/runs/detect**: Default output directory for YOLOv8 training, containing trained models (`best.pt`) and logs.
- **/raw_data**: Contains scripts for initial data gathering and preparation.
- `requirements.txt`: Lists all Python dependencies.

## 4. Key Scripts and Workflow

### 4.1. Main Pipeline: `src/main.py`

This is the primary entry point for the full workflow. It performs the following steps sequentially:

1.  **Load Model**: Loads the pre-trained `yolov8n.pt` model.
2.  **Fine-Tune**: Trains the model on the custom dataset defined in `data/data.yaml`.
    - Key parameters: `imgsz=224`, `epochs=1`, `batch=16`.
    - The best performing model is saved as `best.pt` inside the `runs/detect/yolov8n_finetune/weights/` directory.
3.  **Export to ONNX**: The `best.pt` model is converted to the standard FP32 ONNX format.
4.  **Quantize to INT8**: The FP32 ONNX model is quantized to INT8 using static quantization.
    - It uses `YOLOv8CalibrationDataReader` from `src/quantize/yoloCalibDataset.py` to feed calibration data to the quantizer.

**To Run:**
```bash
python src/main.py
```
*(Requires `data/data.yaml` to be correctly configured)*

### 4.2. Inference: `src/predict.py`

This script is used to run inference on images using a trained PyTorch (`.pt`) model.

1.  **Finds Model**: It automatically locates the most recently modified `best.pt` file within the `/runs/detect` directory.
2.  **Loads Model**: Initializes a `YOLO` object with the found model path.
3.  **Performs Inference**: Runs the model on a specified input image.
4.  **Saves Results**: Saves the detection results (class ID, normalized xywh) to a `.txt` file with the same name as the image.
5.  **Visualize**: Optionally displays the image with bounding boxes drawn on it.

**To Run (example from script):**
The script is currently hardcoded to process images in the `1002_data/raw_data` folder.

### 4.3. Standalone Quantization: `src/quantize/quantize_pt.py`

This script provides a focused tool for converting a fine-tuned `.pt` model directly to a quantized INT8 ONNX model. This is useful for re-quantizing a model without re-training.

1.  **Configuration**: Requires manual setting of paths for the input `.pt` model and the `dataset.yaml`.
2.  **PT -> FP32 ONNX**: Converts the PyTorch model to FP32 ONNX format.
3.  **FP32 ONNX -> INT8 ONNX**: Performs static quantization using the `YOLOv8CalibrationDataReader`.

**To Run:**
```bash
# 1. Update PT_PATH and YAML_PATH in the script
# 2. Execute the script
python src/quantize/quantize_pt.py
```

## 5. Configuration

- **`data/data.yaml`**: This is the most critical configuration file. It **must** be correctly set up with paths to `train` and `val` image directories, the number of classes (`nc`), and a list of class `names`. The paths in this file are relative to the file itself.
- **Image Size**: The project consistently uses an image size of `224x224` for training, export, and quantization.
