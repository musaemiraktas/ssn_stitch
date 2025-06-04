import os
import cv2
import numpy as np
from pathlib import Path

def visualize_realworld_results(results: dict,
                                save_dir: Path = Path("./visuals"),
                                threshold: float = 0.4):
    
    os.makedirs(save_dir, exist_ok=True)

    anomaly_maps_tensor = results["anomaly_map"]  # Torch Tensor (N,1,h0,w0)
    all_maps = anomaly_maps_tensor.numpy()
    global_min = all_maps.min()
    global_max = all_maps.max()

    image_paths = results["image_path"]

    for idx, img_path in enumerate(image_paths):
        p = Path(img_path)
        anomaly_map = all_maps[idx, 0, :, :].copy()  # (h0, w0), float32

        norm_map = (anomaly_map - global_min) / (global_max - global_min + 1e-8)

        orig = cv2.imread(str(p))
        if orig is None:
            continue
        H, W, _ = orig.shape

        norm_resized = cv2.resize(norm_map, (W, H), interpolation=cv2.INTER_LINEAR)

        pred_mask = (norm_resized >= threshold).astype(np.uint8) * 255  # (H, W)

        mask_bgr = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)

        contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlaid = orig.copy()
        cv2.drawContours(overlaid, contours, -1, (0, 0, 255), 2)

        combined = np.hstack([mask_bgr, overlaid])

        combined_folder = save_dir / p.parent.name / "combined"
        combined_folder.mkdir(parents=True, exist_ok=True)
        combined_filename = f"{p.stem}_combined.png"
        cv2.imwrite(str(combined_folder / combined_filename), combined)