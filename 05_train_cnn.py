import json
import os
import random

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam

matplotlib.use('Agg')

IMG_SIZE = (224, 224)
MAX_SAMPLES_PER_CLASS = 120


def load_images_for_cnn(data_list, max_per_class=None):
    """Load a memory-safe subset of images for CNN training or evaluation."""
    filtered = []
    counts = {}

    for item in data_list:
        class_name = item.get('class_name', str(item.get('label')))
        counts[class_name] = counts.get(class_name, 0)
        if max_per_class is None or counts[class_name] < max_per_class:
            filtered.append(item)
            counts[class_name] += 1

    random.Random(42).shuffle(filtered)
    images = []
    labels = []

    for item in filtered:
        img = cv2.imread(item['path'])
        if img is None:
            continue
        img = cv2.resize(img, IMG_SIZE)
        img = img.astype('float32') / 255.0
        images.append(img)
        labels.append(item['label'])

    if not images:
        raise ValueError('No valid images were loaded for CNN training.')

    return np.array(images), np.array(labels)


def build_cnn_model(num_classes, input_shape=(224, 224, 3)):
    """Build a memory-efficient CNN model for CPU training."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    return model


def main():
    print('\n' + '=' * 60)
    print('STEP 5: TRAIN CNN CLASSIFIER')
    print('=' * 60)

    with open('data_splits.json', 'r') as f:
        splits = json.load(f)

    class_names = splits['class_names']
    num_classes = len(class_names)
    os.makedirs('outputs', exist_ok=True)

    print(f'\n[OK] Loading balanced CNN dataset (max {MAX_SAMPLES_PER_CLASS} per class)...')
    X_train, y_train = load_images_for_cnn(splits['train'], max_per_class=MAX_SAMPLES_PER_CLASS)
    X_val, y_val = load_images_for_cnn(splits['val'], max_per_class=MAX_SAMPLES_PER_CLASS)
    X_test, y_test = load_images_for_cnn(splits['test'], max_per_class=MAX_SAMPLES_PER_CLASS)

    print(f'  Train: {len(X_train)} samples')
    print(f'  Val:   {len(X_val)} samples')
    print(f'  Test:  {len(X_test)} samples')

    print('\n[OK] Building CNN model...')
    model = build_cnn_model(num_classes)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    print('\n[OK] Training CNN model...')
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=12,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    print('\n[OK] Evaluating on test set...')
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'  Test Loss: {test_loss:.4f}')
    print(f'  Test Accuracy: {test_accuracy:.4f}')

    print('\n[OK] Saving CNN model...')
    model.save('models/cnn_model.h5')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].set_title('Model Loss', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['accuracy'], label='Train Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Accuracy', fontsize=11)
    axes[1].set_title('Model Accuracy', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/cnn_training_history.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    results = {
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss),
        'num_classes': num_classes,
        'class_names': class_names,
        'epochs_trained': len(history.history['loss']),
        'max_samples_per_class': MAX_SAMPLES_PER_CLASS,
    }

    with open('models/cnn_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\n[DONE] CNN model trained and saved!')
    print('=' * 60 + '\n')


if __name__ == '__main__':
    main()
