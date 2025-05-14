from pathlib import Path
import cv2
import numpy as np
import albumentations as A
from pandas import DataFrame
from anomalib.data.utils import Split, LabelName, InputNormalizationMethod
from datamodules.base.datamodule import SSNDataModule
from datamodules.base.dataset import SSNDataset

class StitchSupervisedDataset(SSNDataset):
    def __init__(self, root, supervised, transform, split, flips, normal_flips, debug=False):
        super().__init__(
            root=root,
            supervised=supervised,
            transform=transform,
            split=split,
            flips=flips,
            normal_flips=normal_flips,
            debug=debug
        )

    def make_dataset(self):
        image_dir = self.root / self.split.value / "images"
        mask_dir = self.root / self.split.value / "masks"

        print(f"📂 İşleniyor: {self.split.value.upper()} split")
        print(f"🔍 Görüntü klasörü: {image_dir}")
        print(f"🔍 Maske klasörü: {mask_dir}")

        samples = []
        total_found = 0
        total_skipped = 0

        for img_path in sorted(image_dir.glob("*.jpg")):
            mask_path = mask_dir / f"{img_path.stem}.png"
            if not mask_path.exists():
                print(f"⚠️ Maske bulunamadı: {mask_path.name} → atlandı")
                total_skipped += 1
                continue
            total_found += 1
            label = LabelName.ABNORMAL if self._is_defective(mask_path) else LabelName.NORMAL
            samples.append([
                str(self.root), img_path.stem, self.split.value,
                str(img_path), str(mask_path), label
            ])

        print(f"✅ Eşleşen {total_found} görüntü bulundu, {total_skipped} maske eksik")

        if not samples:
            raise RuntimeError(f"‼️ '{self.split.value}' split için hiç eşleşen görüntü ve maske bulunamadı.")

        df = DataFrame(samples, columns=["path", "sample_id", "split", "image_path", "mask_path", "label_index"])
        df.label_index = df.label_index.astype(int)

        normal_df = df[df.label_index == 0].reset_index(drop=True)
        abnormal_df = df[df.label_index == 1].reset_index(drop=True)

        print(f"🟩 Normal örnekler: {len(normal_df)}")
        print(f"🟥 Anormal örnekler: {len(abnormal_df)}")

        return normal_df, abnormal_df


    def _is_defective(self, mask_path):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        return not np.all(mask == 0)


class StitchSupervised(SSNDataModule):
    def __init__(
        self,
        root,
        image_size=None,
        normalization=InputNormalizationMethod.IMAGENET,
        train_batch_size=8,
        eval_batch_size=8,
        num_workers=0,
        seed=None,
        flips=False,
        debug=False,
    ):
        supervised = True
        super().__init__(root, supervised, image_size, normalization, train_batch_size, eval_batch_size, num_workers, seed, flips)

        self.train_data = StitchSupervisedDataset(
            root=Path(root),
            supervised=True,
            transform=self.transform_train,
            split=Split.TRAIN,
            flips=flips,
            normal_flips=False,
            debug=debug,
        )
        self.test_data = StitchSupervisedDataset(
            root=Path(root),
            supervised=True,
            transform=self.transform_eval,
            split=Split.TEST,
            flips=False,
            normal_flips=False,
            debug=debug,
        )
