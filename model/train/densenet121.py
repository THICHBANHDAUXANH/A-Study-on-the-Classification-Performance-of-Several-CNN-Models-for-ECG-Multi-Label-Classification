import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
import torchvision.models as models
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
import os
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT/"outputs/models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

X_PATH = REPO_ROOT/"outputs/arrays/ecg_images_array.npy"
Y_PATH = REPO_ROOT/"outputs/arrays/ecg_labels_array.npy"

X = np.load(str(X_PATH))
y = np.load(str(Y_PATH))

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.34, random_state=48)

X_train = np.transpose(X_train, (0, 3, 1, 2))
X_val = np.transpose(X_val, (0, 3, 1, 2))

X_train = X_train.astype(np.float32) / 255.0
X_val = X_val.astype(np.float32) / 255.0

X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.FloatTensor(y_val)

batch_size = 64
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Number of classes: {y.shape[1]}")

class ECGDenseNet121(nn.Module):
    def __init__(self, num_classes):
        super(ECGDenseNet121, self).__init__()
        self.densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

        in_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.densenet(x)

num_classes = y.shape[1]
model = ECGDenseNet121(num_classes).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        predicted = (output > 0.5).float()
        total += target.size(0) * target.size(1)
        correct += (predicted == target).sum().item()
        all_predictions.append(predicted.detach().cpu().numpy())
        all_targets.append(target.detach().cpu().numpy())

        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    stacked_targets = np.vstack(all_targets)
    stacked_predictions = np.vstack(all_predictions)
    epoch_precision = 100.0 * precision_score(
        stacked_targets,
        stacked_predictions,
        average="macro",
        zero_division=0,
    )
    epoch_recall = 100.0 * recall_score(
        stacked_targets,
        stacked_predictions,
        average="macro",
        zero_division=0,
    )
    epoch_f1 = 100.0 * f1_score(
        stacked_targets,
        stacked_predictions,
        average="macro",
        zero_division=0,
    )
    return epoch_loss, epoch_acc, epoch_precision, epoch_recall, epoch_f1

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()

            predicted = (output > 0.5).float()
            total += target.size(0) * target.size(1)
            correct += (predicted == target).sum().item()
            all_predictions.append(predicted.detach().cpu().numpy())
            all_targets.append(target.detach().cpu().numpy())

    val_loss /= len(val_loader)
    val_acc = 100. * correct / total
    stacked_targets = np.vstack(all_targets)
    stacked_predictions = np.vstack(all_predictions)
    val_precision = 100.0 * precision_score(
        stacked_targets,
        stacked_predictions,
        average="macro",
        zero_division=0,
    )
    val_recall = 100.0 * recall_score(
        stacked_targets,
        stacked_predictions,
        average="macro",
        zero_division=0,
    )
    val_f1 = 100.0 * f1_score(
        stacked_targets,
        stacked_predictions,
        average="macro",
        zero_division=0,
    )
    return val_loss, val_acc, val_precision, val_recall, val_f1

num_epochs = 50
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
train_precision_scores = []
val_precision_scores = []
train_recall_scores = []
val_recall_scores = []
train_f1_scores = []
val_f1_scores = []
best_val_f1 = 0.0
best_model_path = MODEL_DIR/"best_densenet121_ecg_model.pth"

for epoch in range(num_epochs):
    print(f'\nEpoch {epoch+1}/{num_epochs}')
    print('-' * 50)

    train_loss, train_acc, train_precision, train_recall, train_f1 = train_epoch(
        model, train_loader, criterion, optimizer, device
    )

    val_loss, val_acc, val_precision, val_recall, val_f1 = validate_epoch(
        model, val_loader, criterion, device
    )

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    train_precision_scores.append(train_precision)
    val_precision_scores.append(val_precision)
    train_recall_scores.append(train_recall)
    val_recall_scores.append(val_recall)
    train_f1_scores.append(train_f1)
    val_f1_scores.append(val_f1)

    print(
        f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
        f'Train Precision: {train_precision:.2f}%, Train Recall: {train_recall:.2f}%, '
        f'Train F1: {train_f1:.2f}%'
    )
    print(
        f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, '
        f'Val Precision: {val_precision:.2f}%, Val Recall: {val_recall:.2f}%, '
        f'Val F1: {val_f1:.2f}%'
    )

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_f1': best_val_f1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_precision': train_precision,
            'val_precision': val_precision,
            'train_recall': train_recall,
            'val_recall': val_recall,
            'train_f1': train_f1,
            'val_f1': val_f1,
        }, str(best_model_path))
        print(f'New best model saved! Validation F1: {best_val_f1:.2f}%')

print(f'\nTraining completed!')
print(f'Best validation F1: {best_val_f1:.2f}%')
print(f'Best model saved as: {best_model_path}')

training_history = {
    'train_losses': train_losses,
    'val_losses': val_losses,
    'train_accuracies': train_accuracies,
    'val_accuracies': val_accuracies,
    'train_precision_scores': train_precision_scores,
    'val_precision_scores': val_precision_scores,
    'train_recall_scores': train_recall_scores,
    'val_recall_scores': val_recall_scores,
    'train_f1_scores': train_f1_scores,
    'val_f1_scores': val_f1_scores,
    'best_val_f1': best_val_f1
}

history_path = MODEL_DIR/"densenet121_training_history.npy"
np.save(str(history_path), training_history)
print(f"Training history saved as '{history_path}'")

print("\nLoading best model for final evaluation...")
checkpoint = torch.load(str(best_model_path))
model.load_state_dict(checkpoint['model_state_dict'])

final_val_loss, final_val_acc, final_val_precision, final_val_recall, final_val_f1 = validate_epoch(
    model, val_loader, criterion, device
)
print(f"Final validation accuracy: {final_val_acc:.2f}%")
print(f"Final validation precision: {final_val_precision:.2f}%")
print(f"Final validation recall: {final_val_recall:.2f}%")
print(f"Final validation F1: {final_val_f1:.2f}%")

def load_trained_model(model_path, num_classes, device):
    """
    Load trained DenseNet121 model
    """
    model = ECGDenseNet121(num_classes).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['best_val_f1']
