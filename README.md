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
python 05_train_cnn.py

# Step 6: Evaluate all models
python 06_evaluate.py

# Step 7: Run inference on new images
python 07_inference.py <image_path>
```

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

#### 5. CNN Training (`05_train_cnn.py`)
- Trains a custom 4-layer CNN
- Uses data augmentation and dropout for regularization
- Plots training history
- Saves model checkpoints and results

#### 6. Model Evaluation (`06_evaluate.py`)
- Evaluates all trained models on test set
- Generates confusion matrices
- Creates comprehensive comparison report
- Saves visualizations to `outputs/`

#### 7. Inference (`07_inference.py`)
- Makes predictions on new images
- Supports individual model selection or ensemble prediction
- Provides confidence scores for predictions

Example usage:
```bash
# Use all models for prediction
python 07_inference.py path/to/image.jpg

# Use specific model
python 07_inference.py path/to/image.jpg --model cnn
python 07_inference.py path/to/image.jpg --model svm
python 07_inference.py path/to/image.jpg --model knn
```

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