from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import torch


class SimpleImageDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform
        self.paths = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        # Mask ve label ekliyoruz
        return {
            "image": image,  # [C, H, W]
            "mask": torch.zeros(image.shape[1], image.shape[2]),
            "label": torch.tensor(0, dtype=torch.float32),
            "image_path": self.paths[idx],
            "mask_path": "",  # dummy
        }



class SimpleImageDataModule:
    def __init__(self, root, image_size, batch_size, num_workers):
        self.root = root
        self.train_dir = os.path.join(root, "train")
        self.test_dir = os.path.join(root, "test")
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
        self.train_data = SimpleImageDataset(self.train_dir, self.transform)
        self.test_data = SimpleImageDataset(self.test_dir, self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)