import torch
from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd

from dataset import load_patch_list, split_scenes, fit_normalization_stats, dataloaders, DATA_ROOT
from model import UNet


def calculate_full_metrics(prediction, target, threshold=0.5, smooth=1e-6):
    pred_binary = (prediction > threshold).float()
    pred_binary = pred_binary.view(-1)
    target = target.view(-1)

    TP = (pred_binary * target).sum()
    FP = (pred_binary * (1 - target)).sum()
    FN = ((1 - pred_binary) * target).sum()
    TN = ((1 - pred_binary) * (1 - target)).sum()

    pixel_accuracy = (TP + TN + smooth) / (TP + TN + FP + FN + smooth)

    intersection = TP
    union = TP + FP + FN
    iou = (intersection + smooth) / (union + smooth)

    precision = (TP + smooth) / (TP + FP + smooth)
    recall = (TP + smooth) / (TP + FN + smooth)
    f1 = 2 * (precision * recall) / (precision + recall)

    return {
        'pixel_accuracy': pixel_accuracy.item(),
        'iou': iou.item(),
        'f1': f1.item(),
        'precision': precision.item(),
        'recall': recall.item()
    }

def evaluate_model(model, test_loader, device, threshold=0.5):
    model.eval()

    total_metrics = {'pixel_accuracy': 0, 'iou': 0, 'f1': 0, 'precision': 0, 'recall': 0}

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)

            prediction = model(images)

            batch_metrics = calculate_full_metrics(prediction, masks, threshold=threshold)

            for key in total_metrics:
                total_metrics[key] += batch_metrics[key]

    n = len(test_loader)
    avg_metrics = {key: value / n for key, value in total_metrics.items()}

    return avg_metrics

def save_metrics(metrics, save_path="outputs/metrics.json"):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)

def save_prediction_images(model, test_loader, device, num_images=5, save_dir="outputs/predictions", threshold=0.5):
    model.eval()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    count = 0
    with torch.no_grad():
        for images, masks in test_loader:
            images_gpu = images.to(device)
            predictions = model(images_gpu)
            
            for i in range(images.shape[0]):
                if count >= num_images:
                    return  
                
                rgb = images[i, :3, :, :].permute(1, 2, 0).cpu().numpy()

                gt = masks[i, 0, :, :].cpu().numpy()

                pred = (predictions[i, 0, :, :] > threshold).float().cpu().numpy()

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(rgb)
                axes[0].set_title('RGB Image')
                axes[0].axis('off')
                axes[1].imshow(gt, cmap='gray')
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')
                axes[2].imshow(pred, cmap='gray')
                axes[2].set_title('Prediction')
                axes[2].axis('off')

                plt.tight_layout()
                plt.savefig(f"{save_dir}/prediction_{count}.png")
                plt.close()
                
                count += 1

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
    model.load_state_dict(torch.load("outputs/best_model.pth", weights_only=True))

    metrics = evaluate_model(model, test_loader, device)
    print("Test set metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    save_metrics(metrics, save_path="outputs/metrics.json")
    print("บันทึก metrics.json แล้ว")

    save_prediction_images(model, test_loader, device, num_images=5, save_dir="outputs/predictions")
    print("บันทึกภาพ prediction แล้ว")