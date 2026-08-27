import os
import pandas as pd
from pathlib import Path

import random

import rasterio
import numpy as np

import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(os.environ.get("CLOUD_PROJECT_ROOT", "."))
DATA_ROOT = PROJECT_ROOT / "38-Cloud_training"


def load_patch_list(filter_source="official"):
    if filter_source == "official":
        path = DATA_ROOT / "training_patches_38-cloud_nonempty.csv"
        dataframe = pd.read_csv(path)
        patches = dataframe['name'].tolist()
    elif filter_source == "custom":
        path = DATA_ROOT / "training_patches_custom_nonempty.csv"
        dataframe = pd.read_csv(path)
        patches = dataframe[dataframe['is_informative'] == True]['name'].tolist()
    else:
        raise ValueError("Fill 'official' or 'custom'")
    
    print(f"Load {path.name} -> {len(patches)} patches")
    return patches

def split_scenes(scene_ids, patch_names, n_train=12, n_val=3, seed=42):
    # แบ่งตาม scene ไม่ใช่ตาม patch 
    scenes = scene_ids[:]
    random.seed(seed)
    random.shuffle(scenes)
 
    train_scenes = set(scenes[:n_train])
    val_scenes = set(scenes[n_train:n_train + n_val])
    
    train, val, test = [], [], []
    for name in patch_names:
        matched = False
        for scene in train_scenes:
            if scene in name:
                train.append(name)
                matched = True
                break
        if matched: continue
 
        for scene in val_scenes:
            if scene in name:
                val.append(name)
                matched = True
                break
        if not matched:
            test.append(name)
 
    return train, val, test


def fit_normalization_stats(patches, data_root, bands=['red', 'green', 'blue', 'nir'],sample_size=500, seed=42):
    # sample แค่บางส่วนไม่งั้นจะกิน RAM 
    if len(patches) > sample_size:
        random.seed(seed)
        sample = random.sample(patches, sample_size)
    else:
        sample = patches
 
    band_pools = {b: [] for b in bands}
    for patch_suffix in sample:
        for band in bands:
            path = data_root / f"train_{band}" / f"{band}_{patch_suffix}.TIF"
            with rasterio.open(path) as src:
                arr = src.read(1)
                band_pools[band].append(arr[arr !=0])
 
    stats = {}
    for band in bands:
        all_px = np.concatenate(band_pools[band])
        stats[band] = (np.percentile(all_px, 1), np.percentile(all_px, 99))
    return stats


class CloudDataset(Dataset):
    def __init__(self, patch_list, data_root, norm_stats, bands=['red', 'green', 'blue', 'nir'], augment=False):
        self.patch_list = patch_list
        self.data_root = data_root
        self.norm_stats = norm_stats
        self.bands = bands
        self.augment = augment
 
    def __len__(self):
        return len(self.patch_list)
 
    def _augment(self, image, mask):
        if random.random() < 0.5:
            image = image[:, :, ::-1]   
            mask = mask[:, ::-1]
        if random.random() < 0.5:
            image = image[:, ::-1, :]  
            mask = mask[::-1, :]
        if random.random() < 0.5:
            k = random.choice([1, 2, 3])
            image = np.rot90(image, k, axes=(1, 2))
            mask = np.rot90(mask, k, axes=(0, 1))
        return image, mask
 
 
    def __getitem__(self, idx):
        patch_suffix = self.patch_list[idx]
 
        band_arrays = []
        for band in self.bands:
            path = self.data_root / f"train_{band}" / f"{band}_{patch_suffix}.TIF"
            with rasterio.open(path) as src:
                arr = src.read(1).astype(np.float32)
            p1, p99 = self.norm_stats[band]
            arr = (arr - p1) / (p99 - p1)
            arr = np.clip(arr, 0, 1)
            band_arrays.append(arr)
        image = np.stack(band_arrays, axis=0)
 
        gt_path = self.data_root / "train_gt" / f"gt_{patch_suffix}.TIF"
        with rasterio.open(gt_path) as src:
            gt = src.read(1)
        mask = (gt == 255).astype(np.float32)
 
        if self.augment:
            image, mask = self._augment(image, mask)
 
        image_tensor = torch.from_numpy(image.copy()).float()
        mask_tensor = torch.from_numpy(mask.copy()).float().unsqueeze(0)
 
        return image_tensor, mask_tensor 


from torch.utils.data import DataLoader

def dataloaders(train_patches, val_patches, test_patches, data_root, norm_stats, batch_size=8):
    train_data = CloudDataset(train_patches, data_root, norm_stats, augment=True)
    val_data = CloudDataset(val_patches, data_root, norm_stats, augment=False)
    test_data = CloudDataset(test_patches, data_root, norm_stats, augment=False)
 
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)    #True (สลับทุก epoch)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
 
 
if __name__ == "__main__":
    patch_names = load_patch_list("official") # official, custom select
 
    scenes_dataframe = pd.read_csv(DATA_ROOT / "training_sceneids_38-Cloud.csv")
    scene_ids = scenes_dataframe.iloc[:, 0].tolist()
 
    train_patches, val_patches, test_patches = split_scenes(scene_ids, patch_names, n_train=12, n_val=3, seed=42)
 
    print(f"train={len(train_patches)} val={len(val_patches)} test={len(test_patches)} "
          f"(รวม {len(train_patches)+len(val_patches)+len(test_patches)}, ทั้งหมด {len(patch_names)})")
 
    norm_stats = fit_normalization_stats(train_patches, DATA_ROOT, sample_size=300)
    for band, (p1, p99) in norm_stats.items():
        print(f"  {band}: P1={p1:.0f}, P99={p99:.0f}")
 
    train_dataset = CloudDataset(train_patches, DATA_ROOT, norm_stats)
    img, msk = train_dataset[0]
    print(f"dataset size={len(train_dataset)} | image={tuple(img.shape)} mask={tuple(msk.shape)}")
    print(f"image range: {img.min():.3f} - {img.max():.3f}, mask values: {msk.unique().tolist()}")
 
    train_loader, val_loader, test_loader = dataloaders(
        train_patches, val_patches, test_patches, DATA_ROOT, norm_stats, batch_size=8
    )
    print(f"batches -> train:{len(train_loader)} val:{len(val_loader)} test:{len(test_loader)}")
 
    imgs, msks = next(iter(train_loader))
    print(f"batch shape: images={tuple(imgs.shape)} masks={tuple(msks.shape)}")