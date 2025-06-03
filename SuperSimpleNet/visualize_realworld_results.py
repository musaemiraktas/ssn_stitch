import os
import cv2
import numpy as np
from pathlib import Path

def visualize_realworld_results(results: dict,
                                save_dir: Path = Path("./visuals"),
                                threshold: float = 0.3):
    os.makedirs(save_dir, exist_ok=True)

    anomaly_maps = results["anomaly_map"].numpy()  # (N, 1, h0, w0)
    image_paths = results["image_path"]            # ['.../image1.png', '.../image2.png', …]
    scores = results["score"].numpy()              # (N,)

    for idx, img_path in enumerate(image_paths):
        p = Path(img_path)

        anomaly_map_tensor = anomaly_maps[idx, 0, :, :]
        score = scores[idx]

        orig = cv2.imread(str(p))
        if orig is None:
            print(f"Image okunamadı: {p}")
            continue
        H, W, _ = orig.shape

        seg = anomaly_map_tensor.copy()  # float32
        seg_norm = (seg - seg.min()) / (seg.max() - seg.min() + 1e-8)

        seg_resized = cv2.resize(seg_norm, (W, H), interpolation=cv2.INTER_LINEAR)  # float32, (H, W)

        mask_inverted = (seg_resized < threshold).astype(np.uint8) * 255  # uint8, (H, W)

        mask_bgr = cv2.cvtColor(mask_inverted, cv2.COLOR_GRAY2BGR)  # shape: (H, W, 3), 0 veya 255 gerçekte

        contours, _ = cv2.findContours(mask_inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlaid = orig.copy()
        cv2.drawContours(overlaid, contours, -1, (0, 0, 255), 2)  # (B, G, R) = kırmızı çizgi kalınlığı=2

      
        combined = np.hstack([mask_bgr, overlaid])  # shape: (H, 2W, 3)

        combined_folder = save_dir / p.parent.name / "combined"
        combined_folder.mkdir(parents=True, exist_ok=True)

        combined_filename = f"{p.stem}_score_{score:.4f}_combined.png"
        cv2.imwrite(str(combined_folder / combined_filename), combined)

    print(f"Tüm “combined” görseller kaydedildi: {save_dir}")