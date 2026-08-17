import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

class TomatoLeafDataset(Dataset):
    """Custom dataset for tomato leaf images"""
    def __init__(self, data_splits, split_name, transform=None):
        self.data = data_splits[split_name]
        self.split_name = split_name
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['path']
        label = item['label']
        
        # Load preprocessed image
        filename = os.path.basename(image_path)
        preprocessed_path = f"outputs/preprocessed/{self.split_name}/{label}/{filename}"
        
        if not os.path.exists(preprocessed_path):
            # Fallback to original image
            image = Image.open(image_path).convert('RGB')
            image = image.resize((128, 128))
        else:
            image = Image.open(preprocessed_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class SimpleCNN(nn.Module):
    """Simple CNN for tomato leaf disease classification"""
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Conv Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Conv Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def load_data_splits():
    """Load data splits from JSON"""
    with open("data_splits.json", 'r') as f:
        splits = json.load(f)
    
    return splits

def prepare_datasets(splits):
    """Prepare train, validation, and test datasets"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = TomatoLeafDataset(splits, 'train', transform=transform)
    val_dataset = TomatoLeafDataset(splits, 'val', transform=transform)
    test_dataset = TomatoLeafDataset(splits, 'test', transform=transform)
    
    return train_dataset, val_dataset, test_dataset

def encode_labels(train_dataset, val_dataset, test_dataset):
    """Encode string labels to integers"""
    # Get all unique labels
    all_labels = []
    for dataset in [train_dataset, val_dataset, test_dataset]:
        all_labels.extend([item[1] for item in dataset])
    
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    
    # Create new datasets with encoded labels
    class EncodedDataset(Dataset):
        def __init__(self, original_dataset, label_encoder):
            self.original_dataset = original_dataset
            self.label_encoder = label_encoder
        
        def __len__(self):
            return len(self.original_dataset)
        
        def __getitem__(self, idx):
            image, label = self.original_dataset[idx]
            encoded_label = self.label_encoder.transform([label])[0]
            return image, torch.tensor(encoded_label, dtype=torch.long)
    
    train_encoded = EncodedDataset(train_dataset, label_encoder)
    val_encoded = EncodedDataset(val_dataset, label_encoder)
    test_encoded = EncodedDataset(test_dataset, label_encoder)
    
    return train_encoded, val_encoded, test_encoded, label_encoder

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=20):
    """Train the CNN model"""
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    best_val_accuracy = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses, val_accuracies, best_val_accuracy

def evaluate_model(model, test_loader, device, label_encoder):
    """Evaluate model on test set"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert back to string labels
    all_predictions_str = label_encoder.inverse_transform(all_predictions)
    all_labels_str = label_encoder.inverse_transform(all_labels)
    
    accuracy = accuracy_score(all_labels_str, all_predictions_str)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels_str, all_predictions_str))
    
    return accuracy

def plot_training_history(train_losses, val_losses, val_accuracies):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    ax2.plot(val_accuracies, label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('outputs/cnn_training_history.png')
    print("[OK] Training history plot saved")

def main():
    print("=" * 60)
    print("TRAINING CNN MODEL")
    print("=" * 60)
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data splits
    print("Loading data splits...")
    splits = load_data_splits()
    
    # Prepare datasets
    print("Preparing datasets...")
    train_dataset, val_dataset, test_dataset = prepare_datasets(splits)
    
    # Encode labels
    print("Encoding labels...")
    train_dataset, val_dataset, test_dataset, label_encoder = encode_labels(
        train_dataset, val_dataset, test_dataset
    )
    
    num_classes = len(label_encoder.classes_)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {label_encoder.classes_}")
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    print("Initializing CNN model...")
    model = SimpleCNN(num_classes=num_classes).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    print("Starting training...")
    model, train_losses, val_losses, val_accuracies, best_val_accuracy = train_model(
        model, train_loader, val_loader, criterion, optimizer, device, num_epochs=20
    )
    
    # Plot training history
    plot_training_history(train_losses, val_losses, val_accuracies)
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    test_accuracy = evaluate_model(model, test_loader, device, label_encoder)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'label_encoder': label_encoder,
        'num_classes': num_classes,
        'val_accuracy': best_val_accuracy,
        'test_accuracy': test_accuracy
    }, 'models/cnn_model.pth')
    print("[OK] CNN model saved")
    
    # Save results
    results = {
        'val_accuracy': float(best_val_accuracy),
        'test_accuracy': float(test_accuracy),
        'num_classes': num_classes,
        'classes': label_encoder.classes_.tolist()
    }
    
    with open('outputs/cnn_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("[OK] Results saved")
    
    print("\n" + "=" * 60)
    print("CNN TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()