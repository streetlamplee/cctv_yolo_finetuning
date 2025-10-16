# YOLOv8 Custom Fine-Tuning Project

## 1. Project Overview

This project is designed for fine-tuning a **YOLOv8n** object detection model on a custom dataset. It adapts the pre-trained `yolov8n` model for a specific input image size of `224x224` pixels. The dataset is expected to be in the YOLO format, as exported from annotation tools like CVAT.

### Key Technologies
*   **Python**: The main programming language.
*   **Ultralytics YOLOv8**: The core framework for model training and inference.

### Architecture
The project is orchestrated by the `src/main.py` script, which handles the entire fine-tuning pipeline:
1.  Loads the pre-trained `yolov8n` model.
2.  Configures training parameters (image size, dataset path, etc.).
3.  Initiates the training process.

Dataset configuration (paths, class names) is managed through the `dataset.yaml` file.

## 2. Setup and Execution

### Step 1: Install Dependencies

Ensure you have Python installed. Then, install the required libraries from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Step 2: Prepare Dataset

The model expects a dataset in YOLO format.

1.  **Export Data**: Export your annotated dataset from a tool like **CVAT** using the "YOLO" format. This will produce a directory structure similar to the following:

    ```
    your_dataset_name/
    ├── images/
    │   ├── train/
    │   └── valid/
    └── labels/
        ├── train/
        └── valid/
    ```

2.  **Configure `dataset.yaml`**: Update the `dataset.yaml` file in the project root to match your dataset's structure and class definitions.

    *   `train`: Path to the training images directory (e.g., `./your_dataset_name/images/train`).
    *   `val`: Path to the validation images directory (e.g., `./your_dataset_name/images/valid`).
    *   `nc`: The total number of classes in your dataset.
    *   `names`: An ordered list of your class names.

    **Example `dataset.yaml`:**
    ```yaml
    train: /path/to/your/cvat_export/images/train
    val: /path/to/your/cvat_export/images/val

    # Class settings
    nc: 3  # Number of classes
    names: ['person', 'car', 'bicycle']  # Class names
    ```

### Step 3: Run Training

Once the dependencies are installed and `dataset.yaml` is configured, start the model fine-tuning by running the `main.py` script:

```bash
python src/main.py
```

Training results, including model weights and performance plots, will be saved in the `runs/detect/yolov8n_finetune` directory by default.

## 3. Development Conventions

*   **Framework**: All object detection tasks are built upon the Ultralytics YOLOv8 library.
*   **Configuration**: Dataset paths and class information are centrally managed in `dataset.yaml`.
*   **Image Size**: The model is configured by default to train with an input image size of `224x224`.
