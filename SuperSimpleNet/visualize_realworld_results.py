import cv2
import torch
import numpy as np
import os
from pathlib import Path
from torchvision.transforms.functional import to_pil_image

def visualize_realworld_results(results: dict, save_dir: str = "./visuals"):
    os.makedirs(save_dir, exist_ok=True)

    for image_path, anomaly_map in zip(results["image_path"], results["anomaly_map"]):
        orig_img = cv2.imread(str(image_path))
        orig_img = cv2.resize(orig_img, (anomaly_map.shape[-1], anomaly_map.shape[-2]))

        anomaly_map_np = anomaly_map.squeeze().numpy()
        anomaly_map_np = (anomaly_map_np - anomaly_map_np.min()) / (anomaly_map_np.max() - anomaly_map_np.min() + 1e-8)
        anomaly_map_np = np.uint8(255 * anomaly_map_np)

        heatmap = cv2.applyColorMap(anomaly_map_np, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(orig_img, 0.6, heatmap, 0.4, 0)

        filename = Path(image_path).stem + "_overlay.png"
        cv2.imwrite(str(Path(save_dir) / filename), overlay)

    print(f"Görseller kaydedildi: {save_dir}")