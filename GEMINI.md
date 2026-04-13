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

---

## 6. Prompt for AI Assistant

**Objective:** Your primary goal is to assist in the development, maintenance, and enhancement of this YOLOv8 fine-tuning project. You must act as an expert in machine learning, data processing, and software engineering.

**Core Instructions:**

1.  **Understand First, Act Second:** Before making any changes, thoroughly analyze the relevant files (`main.py`, `predict.py`, `quantize_pt.py`, `data/data.yaml`, etc.) to understand the existing workflow, parameters, and coding conventions.
2.  **Configuration is Key:** Always be mindful of the `data/data.yaml` file. Any task related to training or data requires you to first verify its contents (paths, number of classes, class names). Paths within this file are relative to its location.
3.  **Follow Existing Patterns:** When adding new features or modifying code, strictly adhere to the existing code style, structure, and logic. For example, if adding a new data processing step, model it after the existing pipeline in `main.py`.
4.  **Dependency Management:** Before using a new library, check if it is already listed in `requirements.txt`. If not, you must add it to maintain project reproducibility.
5.  **Path Management:** Be extremely careful with file paths.
    - The project uses both absolute and relative paths. Understand the context for each.
    - The `runs/detect` directory is dynamically created by the `ultralytics` library. Your code must be able to locate the latest training results within this directory (e.g., by finding the most recent `best.pt` file).
6.  **Verify Your Work:** After any modification, you are responsible for verifying that the changes work as expected and have not introduced regressions. This may involve:
    - Running the modified script (`main.py`, `predict.py`, etc.).
    - Adding temporary print statements or logging to trace the execution flow.
    - If you add a new feature, you should also add a corresponding test case in the `src/test` directory.

**Common Tasks & How to Approach Them:**

*   **"Train the model with new data."**
    1.  Ask the user for the location of the new `dataset.yaml` or the new image/label directories.
    2.  Update `data/data.yaml` accordingly.
    3.  Verify the paths and class information are correct.
    4.  Execute the main training pipeline: `python src/main.py`.

*   **"Change a training parameter (e.g., image size, epochs)."**
    1.  Locate the training call (`model.train(...)`) in `src/main.py`.
    2.  Modify the specified parameter (e.g., `imgsz`, `epochs`).
    3.  Remember that changing `imgsz` will require changes in the `export` and `quantize` steps as well. Ensure consistency across the entire pipeline.

*   **"Run inference on a new image."**
    1.  Identify the `predict.py` script as the tool for this.
    2.  Locate the latest trained model (`best.pt`) in the `runs/detect` directory.
    3.  Modify `predict.py` to point to the new input image.
    4.  Execute the script: `python src/predict.py`.

*   **"Debug a problem in the quantization process."**
    1.  Focus your analysis on `src/main.py` (the quantization part) and the `src/quantize/` sub-package.
    2.  Pay close attention to `yoloCalibDataset.py`, as calibration data is a common source of errors. Ensure it's reading images correctly from the `val` set specified in `data/data.yaml`.
    3.  Use `src/quantize/quantize_pt.py` for isolated testing of the quantization step.