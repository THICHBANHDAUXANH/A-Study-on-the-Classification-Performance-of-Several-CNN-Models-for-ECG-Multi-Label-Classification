from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

REPO_ROOT = Path(__file__).resolve().parents[2]

X_PATH = REPO_ROOT/"outputs/arrays/ecg_images_array.npy"
Y_PATH = REPO_ROOT/"outputs/arrays/ecg_labels_array.npy"
CHECKPOINT_PATH = REPO_ROOT/"outputs/models/best_resnet50_ecg_model.pth"

batch_size = 120


class ECGResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ECGResNet50, self).__init__()
        self.resnet = models.resnet50(weights=None)

        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.resnet(x)


def load_validation_data():
    X = np.load(str(X_PATH))
    y = np.load(str(Y_PATH))

    _, X_val, _, y_val = train_test_split(X, y, test_size=0.34, random_state=48)

    X_val = np.transpose(X_val, (0, 3, 1, 2))
    X_val = X_val.astype(np.float32) / 255.0

    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Validation samples: {len(val_dataset)}")
    print(f"Number of classes: {y.shape[1]}")
    return val_loader, y.shape[1]


def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()

            predicted = (output > 0.5).float()
            total += target.size(0) * target.size(1)
            correct += (predicted == target).sum().item()
            all_predictions.append(predicted.detach().cpu().numpy())
            all_targets.append(target.detach().cpu().numpy())

            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(val_loader)}")

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


def load_trained_model(model_path, num_classes, device):
    model = ECGResNet50(num_classes).to(device)
    checkpoint = torch.load(str(model_path), map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint["best_val_f1"]


def main():
    val_loader, num_classes = load_validation_data()
    criterion = nn.BCELoss()

    print("\nLoading best model for final evaluation...")
    model, best_val_f1 = load_trained_model(CHECKPOINT_PATH, num_classes, device)
    print(f"Best validation F1 from checkpoint: {best_val_f1:.2f}%")

    final_val_loss, final_val_acc, final_val_precision, final_val_recall, final_val_f1 = validate_epoch(
        model, val_loader, criterion, device
    )
    print(f"Final validation loss: {final_val_loss:.4f}")
    print(f"Final validation accuracy: {final_val_acc:.2f}%")
    print(f"Final validation precision: {final_val_precision:.2f}%")
    print(f"Final validation recall: {final_val_recall:.2f}%")
    print(f"Final validation F1: {final_val_f1:.2f}%")


if __name__ == "__main__":
    main()
