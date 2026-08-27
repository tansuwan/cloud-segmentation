import torch
import torch.nn as nn

from dataset import CloudDataset, dataloaders, load_patch_list, split_scenes, fit_normalization_stats, DATA_ROOT
from model import UNet

import time
import pandas as pd

from pathlib import Path
import torch

import matplotlib.pyplot as plt


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, prediction, target):
        prediction = prediction.view(-1)
        target = target.view(-1)

        intersection = (prediction * target).sum()

        dice = (2 * intersection + self.smooth) / (prediction.sum() + target.sum() + self.smooth)
        return 1 - dice


class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight

    def forward(self, prediction, target):
        bce_loss = self.bce(prediction, target)
        dice_loss = self.dice(prediction, target)
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss

def calculate_metrics(prediction, target, threshold=0.5, smooth=1e-6):
    pred_binary = (prediction > threshold).float()
    pred_binary = pred_binary.view(-1)
    target = target.view(-1)

    TP = (pred_binary * target).sum()
    FP = (pred_binary * (1 - target)).sum()
    FN = ((1 - pred_binary) * target).sum()

    intersection = TP
    union = TP + FP + FN
    iou = (intersection + smooth) / (union + smooth)

    precision = (TP + smooth) / (TP + FP + smooth)
    recall = (TP + smooth) / (TP + FN + smooth)
    f1 = 2 * (precision * recall) / (precision + recall)

    return iou.item(), f1.item()

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()  
    total_loss = 0
    total_iou = 0
    total_f1 = 0
    
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        prediction = model(images)
        loss = criterion(prediction, masks)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            iou, f1 = calculate_metrics(prediction, masks)

        total_loss += loss.item()
        total_iou += iou
        total_f1 += f1
    
    avg_loss = total_loss / len(dataloader)
    avg_iou = total_iou / len(dataloader)
    avg_f1 = total_f1 / len(dataloader)
    
    return avg_loss, avg_iou, avg_f1


def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()  # โหมด evaluate
    
    total_loss = 0
    total_iou = 0
    total_f1 = 0
    
    with torch.no_grad():  
        for images, masks in dataloader:

            images = images.to(device)
            masks = masks.to(device)
            prediction = model(images)
            loss = criterion(prediction, masks)
            iou, f1 = calculate_metrics(prediction, masks)

            total_loss += loss.item()
            total_iou += iou
            total_f1 += f1
    
    avg_loss = total_loss / len(dataloader)
    avg_iou = total_iou / len(dataloader)
    avg_f1 = total_f1 / len(dataloader)
    
    return avg_loss, avg_iou, avg_f1


class EarlyStoppingCheckpoint:
    def __init__(self, patience=5, checkpoint_path="outputs/best_model.pth"):
        self.patience = patience              
        self.checkpoint_path = checkpoint_path
        self.best_iou = -1                    
        self.counter = 0                      
        self.should_stop = False

        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, val_iou, model):
        if val_iou > self.best_iou:
            self.best_iou = val_iou
            torch.save(model.state_dict(), self.checkpoint_path)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def train_model(model, train_loader, val_loader, criterion, optimizer, device, 
                 num_epochs=30, patience=5, checkpoint_path="outputs/best_model.pth"):
    
    tracker = EarlyStoppingCheckpoint(patience=patience, checkpoint_path=checkpoint_path)
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_iou': [], 'val_iou': [],
        'train_f1': [], 'val_f1': []
    }
    
    for epoch in range(num_epochs):
        train_loss, train_iou, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)

        val_loss, val_iou, val_f1 = validate_one_epoch(model, val_loader, criterion, device)

        print(f"epoch {epoch+1}/{num_epochs} | "
              f"train_loss={train_loss:.4f} train_iou={train_iou:.4f} train_f1={train_f1:.4f} | "
              f"val_loss={val_loss:.4f} val_iou={val_iou:.4f} val_f1={val_f1:.4f}")

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)

        tracker(val_iou, model)
        if tracker.should_stop:
            print(f"early stop ที่ epoch {epoch+1} (val_iou ไม่ดีขึ้นติดกัน {patience} epoch)")
            break

    return history


def plot_training_curves(history, save_path="outputs/training_curves.png"):
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(epochs, history['train_loss'], label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training vs Validation Loss')
    axes[0].legend()
    
    axes[1].plot(epochs, history['train_iou'], label='Train IoU')
    axes[1].plot(epochs, history['val_iou'], label='Val IoU')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('IoU')
    axes[1].set_title('Training vs Validation IoU')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("ใช้ device:", device)
 
    patch_names = load_patch_list("official")
    scenes_dataframe = pd.read_csv(DATA_ROOT / "training_sceneids_38-Cloud.csv")
    scene_ids = scenes_dataframe.iloc[:, 0].tolist()
    train_patches, val_patches, test_patches = split_scenes(scene_ids, patch_names, n_train=12, n_val=3, seed=42)
    norm_stats = fit_normalization_stats(train_patches, DATA_ROOT, sample_size=300)
    train_loader, val_loader, test_loader = dataloaders(
        train_patches, val_patches, test_patches, DATA_ROOT, norm_stats, batch_size=8
    )
 
    model = UNet(in_channels=4, out_channels=1, base_channels=32).to(device)
    criterion = CombinedLoss(bce_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
 
    history = train_model(
        model, train_loader, val_loader, criterion, optimizer, device,
        num_epochs=30, patience=15, checkpoint_path="outputs/best_model.pth"
    )
 
    plot_training_curves(history, save_path="outputs/training_curves.png")
    