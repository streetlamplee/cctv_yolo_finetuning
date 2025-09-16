# Project Overview

This project is set up for fine-tuning a YOLOv8n object detection model. The primary goal is to adapt the pre-trained `yolov8n` model to a custom dataset, with a specific input image size of 224x224 pixels. The dataset is expected to be exported from CVAT in YOLO format.

**Key Technologies:**
*   **Python**: The main programming language.
*   **Ultralytics YOLOv8**: The framework used for model training and inference.

**Architecture:**
The project consists of a `main.py` script that orchestrates the fine-tuning process. It loads the `yolov8n` model, configures training parameters (including image size and dataset path), and initiates the training. The dataset configuration is managed via `dataset.yaml`.

# Building and Running

## 1. Install Dependencies

Ensure you have Python installed. Then, install the required libraries using `pip`:

```bash
pip install -r requirements.txt
```

## 2. Prepare Your Dataset

The project expects the dataset to be in YOLO format, typically exported from tools like CVAT.

1.  **Download Data from CVAT**: Export your annotated dataset from CVAT in "YOLO" format. This usually results in a directory structure like:
    ```
    your_dataset_name/
    ├── images/
    │   ├── train/
    │   └── valid/
    └── labels/
        ├── train/
        └── valid/
    ```
2.  **Update `dataset.yaml`**:
    The `dataset.yaml` file in the project root needs to be updated to point to your dataset's image directories and define your class names and count.
    *   `train`: Path to your training images (e.g., `./your_dataset_name/images/train`).
    *   `val`: Path to your validation images (e.g., `./your_dataset_name/images/valid`).
    *   `nc`: The total number of classes in your dataset.
    *   `names`: A list of your class names, in the correct order corresponding to their IDs.

    **Example `dataset.yaml` structure:**
    ```yaml
    train: /path/to/your/cvat_export/images/train
    val: /path/to/your/cvat_export/images/val

    nc: 3 # Example: 3 classes
    names: ['person', 'car', 'bicycle'] # Example class names
    ```

## 3. Run Training

Once dependencies are installed and `dataset.yaml` is correctly configured, you can start the fine-tuning process by running the `main.py` script:

```bash
python main.py
```

The training will use the `yolov8n` model, an input image size of 224x224, and the dataset defined in `dataset.yaml`. Training results (weights, plots, etc.) will be saved in a `runs/detect/yolov8n_finetune` directory by default.

# Development Conventions

*   **YOLOv8 Framework**: All object detection tasks leverage the Ultralytics YOLOv8 library.
*   **Configuration**: Dataset paths and class information are managed through `dataset.yaml`.
*   **Image Size**: The model is configured to train with an input image size of 224x224.

## Disclaimer

This README.md is written by gemini