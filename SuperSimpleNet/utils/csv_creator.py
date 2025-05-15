import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

def generate_label_csv(dataset_root):
    dataset_root = Path(dataset_root)
    splits = ['train', 'test']
    rows = []

    for split in splits:
        img_dir = dataset_root / split / "images"
        mask_dir = dataset_root / split / "masks"

        for img_path in sorted(img_dir.glob("*.jpg")):
            mask_path = mask_dir / f"{img_path.stem}.png"

            if not mask_path.exists():
                print(f"Maske bulunamadı: {mask_path.name}, atlanıyor.")
                continue

            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            label = 1 if mask is not None and np.sum(mask > 0) > 0 else 0

            rows.append({
                "path": str(dataset_root),
                "sample_id": img_path.stem,
                "split": split,
                "image_path": str(img_path),
                "mask_path": str(mask_path),
                "label_index": label
            })

    csv_path = dataset_root / "label_data.csv"
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"\n{csv_path} oluşturuldu. Toplam örnek: {len(df)}")

generate_label_csv("/content/ssn_stitch/SuperSimpleNet/datasets/stitch")
