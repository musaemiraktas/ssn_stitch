from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np


class SupervisedStitchDataset(Dataset):
    def __init__(self, folder, transform=None, mask_dir=None):
        self.folder = folder
        self.transform = transform
        self.mask_dir = mask_dir
        self.paths = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image_path = self.paths[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Mask path
        mask_path = os.path.join(self.mask_dir, os.path.splitext(os.path.basename(image_path))[0] + ".png")
        mask = Image.open(mask_path).convert("L") if os.path.exists(mask_path) else Image.new("L", image.shape[1:])
        mask = transforms.ToTensor()(mask)
        mask = (mask > 0.5).float().squeeze(0)  # Convert to binary mask [H, W]

        # Label is 1 if any mask pixel is 1, else 0
        label = torch.tensor(float(mask.sum() > 0), dtype=torch.float32)

        return {
            "image": image,  # [C, H, W]
            "mask": mask,    # [H, W]
            "label": label,
            "image_path": image_path,
            "mask_path": mask_path,
        }


class SupervisedStitchDataModule:
    def __init__(self, root, image_size, batch_size, num_workers):
        self.root = root
        self.train_dir = os.path.join(root, "train")
        self.test_dir = os.path.join(root, "test")
        self.train_mask_dir = os.path.join(root, "train_masks")
        self.test_mask_dir = os.path.join(root, "test_masks")
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def setup(self):
        self.train_data = SupervisedStitchDataset(self.train_dir, self.transform, self.train_mask_dir)
        self.test_data = SupervisedStitchDataset(self.test_dir, self.transform, self.test_mask_dir)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
