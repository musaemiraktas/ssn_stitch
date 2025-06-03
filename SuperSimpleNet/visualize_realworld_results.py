import os
import cv2
import numpy as np
from pathlib import Path

def visualize_realworld_results(results: dict, save_dir: Path = Path("./visuals"), threshold: float = 0.3):
    """
    results: {
      "image_path":   list[str],         # str olarak gelen dosya yolları
      "score":        Tensor (N,),
      "anomaly_map":  Tensor (N, 1, h0, w0)
    }
    """
    os.makedirs(save_dir, exist_ok=True)

    anomaly_maps = results["anomaly_map"].numpy()  # (N, 1, h0, w0)
    image_paths = results["image_path"]            # ['.../image1.png', '.../image2.png', …]
    scores = results["score"].numpy()              # (N,)

    for idx, img_path in enumerate(image_paths):
        # >>> Burada img_path bir string, önce Path objesine çeviriyoruz
        p = Path(img_path)

        anomaly_map_tensor = anomaly_maps[idx, 0, :, :]  # shape: (h0, w0)
        score = scores[idx]

        # 1) Orijinal görüntüyü oku (BGR uint8)
        image = cv2.imread(str(p))
        if image is None:
            print(f"Image okunamadı: {p}")
            continue

        # 2) Normalize ve resize anomaly_map
        anomaly_map = anomaly_map_tensor.copy()
        anomaly_map = (anomaly_map - np.min(anomaly_map)) / (np.max(anomaly_map) - np.min(anomaly_map) + 1e-8)
        h, w, _ = image.shape
        anomaly_map_resized = cv2.resize(anomaly_map, (w, h), interpolation=cv2.INTER_LINEAR)

        # 3) Binary maske (threshold 0.3)
        pred_mask = (anomaly_map_resized >= threshold).astype(np.uint8) * 255  # 0 veya 255

        # 4) Konturları bul ve üzerine çiz
        contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlaid = image.copy()
        cv2.drawContours(overlaid, contours, -1, (0, 0, 255), 2)  # kırmızı çizgi

        # 5) Kaydetme klasörünü oluştur
        out_folder = save_dir / p.parent.name
        os.makedirs(out_folder, exist_ok=True)

        # 6) Dosya adı: <stem>_score_<0.xxxx>_overlay.jpg
        filename = f"{p.stem}_score_{score:.4f}_overlay.jpg"
        cv2.imwrite(str(out_folder / filename), overlaid)

    print(f"Tüm patch görselleştirme dosyaları kaydedildi: {save_dir}")
