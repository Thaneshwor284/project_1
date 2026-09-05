# Tomato Leaf Disease Detection System

Automated tomato disease detection system using classical machine learning and deep learning approaches.

## Features
- **Data Preparation**: Automated dataset splitting into train/validation/test sets
- **Image Preprocessing**: Resizing, normalization, and augmentation
- **Feature Extraction**: Color histograms, texture (LBP), and shape descriptors
- **Classical ML**: SVM and KNN classifiers with handcrafted features
- **Deep Learning**: Custom CNN model for end-to-end learning
- **Model Evaluation**: Comprehensive metrics, confusion matrices, and model comparison
- **Inference**: Easy-to-use prediction script for new images

## Dataset
PlantVillage dataset with 10 tomato disease classes:
- Tomato_Bacterial_spot
- Tomato_Early_blight
- Tomato_healthy
- Tomato_Late_blight
- Tomato_Leaf_Mold
- Tomato_Septoria_leaf_spot
- Tomato_Spider_mites_Two_spotted_spider_mite
- Tomato__Target_Spot
- Tomato__Tomato_mosaic_virus
- Tomato__Tomato_YellowLeaf__Curl_Virus

## Setup

### Prerequisites
- Python 3.11+ (tested with Python 3.14.2)
- Windows OS

### Installation

1. **Create virtual environment:**
```bash
python -m venv .venv
```

2. **Activate virtual environment:**
```bash
.venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Dataset Structure
Ensure your dataset is organized as follows:
```
dataset/
└── PlantVillage/
    ├── Tomato_Bacterial_spot/
    ├── Tomato_Early_blight/
    ├── Tomato_healthy/
    ├── Tomato_Late_blight/
    ├── Tomato_Leaf_Mold/
    ├── Tomato_Septoria_leaf_spot/
    ├── Tomato_Spider_mites_Two_spotted_spider_mite/
    ├── Tomato__Target_Spot/
    ├── Tomato__Tomato_mosaic_virus/
    └── Tomato__Tomato_YellowLeaf__Curl_Virus/
```

## Usage

### Complete Pipeline Execution

Run the scripts in order to train and evaluate all models:

```bash
# Step 1: Prepare data splits
python 01_data_preparation.py

# Step 2: Preprocess images
python 02_preprocessing.py

# Step 3: Extract features for traditional ML
python 03_feature_extraction.py

# Step 4: Train SVM and KNN models
python 04_train_svm_knn.py

# Step 5: Train CNN model
python cnn_train.py --dataset-dir dataset\PlantVillage --results-dir results --image-size 128 --epochs 80 --batch-size 32

# Step 6: Evaluate all models
python 06_evaluate.py

# Step 7: Run TFLite inference on a new image
python predict.py <image_path>
```

### Rebuilt Lightweight CNN Training (from scratch)

Use the dedicated end-to-end script below to train/evaluate a lightweight TensorFlow/Keras CNN for the 10-class PlantVillage tomato subset:

```bash
python cnn_train.py --dataset-dir dataset\PlantVillage --results-dir results --image-size 128 --epochs 80 --batch-size 32
```

This script performs:
- Stratified split (70/15/15) directly from the dataset folder structure
- Training-only augmentation (rotation, flip, zoom, brightness)
- Lightweight CNN training with BatchNorm + Dropout
- EarlyStopping, ReduceLROnPlateau, and best-checkpoint saving
- Test evaluation with confusion matrix, classification report, and summary

Artifacts are saved under `results/`:
- `best_model.keras`
- `epoch_metrics.csv`
- `training_curves.png`
- `confusion_matrix.png`
- `classification_report.txt`
- `classification_report.csv`
- `summary.txt`
- `split_summary.json`

### Final CNN and Raspberry Pi Model

The final CNN reaches **91.8% test accuracy** across the 10 tomato disease classes. The converted dynamic-range TFLite model is saved as `results/best_model.tflite` and matches the Keras test accuracy in the conversion check. To classify one image with the TFLite model:

```bash
python predict.py path/to/leaf.jpg
```

The script prints the predicted class and confidence. The conversion and Keras/TFLite comparison are documented in `results/tflite_conversion_summary.txt`.

## Raspberry Pi Spray Automation

### Hardware Needed

- Raspberry Pi with a compatible camera
- 5V opto-isolated relay module
- DC water pump
- Separate power supply for the pump

The pump must **NEVER** be powered from the Pi's own 5V rail or GPIO. Use the pump's separate power supply and switch that supply through the relay. Connect the Pi GPIO only to the relay's control input.

### Running the Spray Controller

From the project root on the Raspberry Pi, run:

```bash
python spray_control.py --model results/best_model.tflite --classes results/split_summary.json --camera-index 0
```

The required arguments are `--model` (the TFLite model), `--classes` (the JSON file containing `class_names`), and `--camera-index` (the camera device index, normally `0`). The controller captures frames, classifies them, and briefly activates the pump through the relay when the prediction is a disease class above the configured confidence threshold.

### Configuration Options

The following constants in `spray_control.py` control the automation:

- `CONFIDENCE_THRESHOLD`: minimum prediction confidence required before spraying
- `HEALTHY_CLASS`: class name that must never trigger spraying
- `SPRAY_DURATION_SEC`: how long the relay stays active for each spray
- `RELAY_PIN`: Raspberry Pi GPIO pin connected to the relay input

### Recommended Hardware Test Order

Test in this order: relay alone -> pump alone -> pump through relay -> prediction alone -> full script on a healthy image -> full script on a diseased image.

### Individual Script Descriptions

#### 1. Data Preparation (`01_data_preparation.py`)
- Scans the dataset directory
- Splits data into train (70%), validation (15%), and test (15%) sets
- Saves split information to `data_splits.json`

#### 2. Image Preprocessing (`02_preprocessing.py`)
- Loads images from data splits
- Resizes images to 128x128 pixels
- Normalizes pixel values
- Saves preprocessed images to `outputs/preprocessed/`

#### 3. Feature Extraction (`03_feature_extraction.py`)
- Extracts color features (HSV statistics)
- Extracts texture features (Local Binary Patterns)
- Extracts shape features (contour analysis)
- Saves features to `models/` as numpy arrays

#### 4. Traditional ML Training (`04_train_svm_knn.py`)
- Trains SVM classifier with RBF kernel
- Trains KNN classifier with k=5
- Evaluates models on validation and test sets
- Saves trained models and results

#### 5. CNN Training (`cnn_train.py`)
- Trains the lightweight CNN on the full tomato dataset
- Uses stratified splits, augmentation, callbacks, and checkpointing
- Saves evaluation reports and plots under `results/`

#### 6. Model Evaluation (`06_evaluate.py`)
- Evaluates all trained models on test set
- Generates confusion matrices
- Creates comprehensive comparison report
- Saves visualizations to `outputs/`

#### 7. TFLite Inference (`predict.py`)
- Loads the Raspberry Pi-ready TFLite model
- Classifies one input image
- Prints the predicted class and confidence

## Output Files

### Model Files (`models/`)
- `svm_model.pkl` - Trained SVM classifier
- `knn_model.pkl` - Trained KNN classifier
- `label_encoder.pkl` - Label encoder for class names
- `cnn_model.pth` - Trained CNN model checkpoint
- `train_features.npy` - Training set features
- `val_features.npy` - Validation set features
- `test_features.npy` - Test set features
- `train_labels.npy` - Training set labels
- `val_labels.npy` - Validation set labels
- `test_labels.npy` - Test set labels

### Output Files (`outputs/`)
- `data_splits.json` - Dataset split information
- `preprocessed/` - Preprocessed images organized by split
- `traditional_ml_results.json` - SVM and KNN performance metrics
- `cnn_results.json` - CNN performance metrics
- `cnn_training_history.png` - Training/validation loss and accuracy plots
- `svm_confusion_matrix.png` - SVM confusion matrix visualization
- `knn_confusion_matrix.png` - KNN confusion matrix visualization
- `model_comparison.json` - Comprehensive model comparison report

## Model Performance

The system provides:
- **SVM**: Good for datasets with clear feature boundaries
- **KNN**: Simple and effective for smaller datasets
- **CNN**: Best for capturing complex patterns in raw image data

Performance metrics include accuracy, precision, recall, and F1-score for each disease class.

## Troubleshooting

### Common Issues

1. **Dataset not found**: Ensure the dataset is in `dataset/PlantVillage/` with the correct folder structure

2. **Memory errors**: Reduce batch size in CNN training or use a smaller subset of data

3. **Import errors**: Make sure all dependencies are installed in the virtual environment

4. **CUDA errors**: The system automatically falls back to CPU if CUDA is not available

## Requirements

See `requirements.txt` for complete list of dependencies:
- numpy
- opencv-python
- matplotlib
- scikit-learn
- scikit-image
- torch
- torchvision
- pillow
- joblib
- pandas
- tqdm
- seaborn

## License

This project is for educational and research purposes. The PlantVillage dataset is available for research use.

## Acknowledgments

- PlantVillage dataset for providing the tomato disease images
- PyTorch and scikit-learn communities for excellent ML frameworks