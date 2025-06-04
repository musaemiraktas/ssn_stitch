import os
import cv2
import numpy as np
from pathlib import Path

def visualize_realworld_results(results: dict,
                                save_dir: Path = Path("./visuals")):
    os.makedirs(save_dir, exist_ok=True)

    anomaly_maps_tensor = results["anomaly_map"]
    all_maps = anomaly_maps_tensor.numpy()
    global_min = all_maps.min()
    global_max = all_maps.max()

    image_paths = results["image_path"]

    for idx, img_path in enumerate(image_paths):
        p = Path(img_path)
        anomaly_map = all_maps[idx, 0, :, :].copy()
        norm_map = (anomaly_map - global_min) / (global_max - global_min + 1e-8)
        print("unique norm values:", np.unique(norm_map))

        orig = cv2.imread(str(p))
        if orig is None:
            continue

        orig = cv2.resize(orig, (norm_map.shape[1], norm_map.shape[0]), interpolation=cv2.INTER_LINEAR)
        overlaid = orig.copy()

        norm_map_uint8 = (norm_map * 255).astype(np.uint8)
        adaptive_mask = cv2.adaptiveThreshold(
            norm_map_uint8,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
            thresholdType=cv2.THRESH_BINARY_INV,
            blockSize=11,
            C=2
        )

        if np.any(adaptive_mask == 255):
            contours, _ = cv2.findContours(adaptive_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlaid, contours, -1, (0, 0, 255), 2)

        mask_bgr = cv2.cvtColor(adaptive_mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([mask_bgr, overlaid])

        combined_folder = save_dir / p.parent.name / "combined_adaptive"
        combined_folder.mkdir(parents=True, exist_ok=True)
        combined_filename = f"{p.stem}_combined_adaptive.png"
        cv2.imwrite(str(combined_folder / combined_filename), combined)
