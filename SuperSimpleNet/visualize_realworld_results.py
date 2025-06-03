import os
import cv2
import numpy as np
from pathlib import Path
from torchvision.utils import save_image
import torch
from tqdm import tqdm

def visualize_realworld_results(results, save_dir="./visuals"):
    os.makedirs(save_dir, exist_ok=True)

    for image_path, anomaly_map in zip(results["image_path"], results["anomaly_map"]):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Hata: Görüntü okunamadı -> {image_path}")
            continue

        anomaly_map = anomaly_map.squeeze().numpy()
        anomaly_map = (anomaly_map * 255).astype(np.uint8)

        _, binary_mask = cv2.threshold(anomaly_map, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        overlay = cv2.drawContours(image.copy(), contours, -1, (0, 0, 255), 2)

        save_name = Path(image_path).stem + "_overlay.jpg"
        cv2.imwrite(str(Path(save_dir) / save_name), overlay)

    print(f"{len(results['image_path'])} görüntü başarıyla kaydedildi -> {save_dir}")