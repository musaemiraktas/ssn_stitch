from pathlib import Path
import cv2
import numpy as np
import albumentations as A
from pandas import DataFrame
from anomalib.data.utils import Split, LabelName, InputNormalizationMethod
from datamodules.base.datamodule import SSNDataModule
from datamodules.base.dataset import SSNDataset

from pandas import read_csv
from anomalib.data.utils import Split
from datamodules.base.dataset import SSNDataset
from anomalib.data.utils import LabelName

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
        csv_path = self.root / "label_data.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"label_data.csv bulunamadı: {csv_path}")

        df = read_csv(csv_path)

        df = df[df["split"] == self.split.value].reset_index(drop=True)
        df["label_index"] = df["label_index"].astype(int)

        normal_df = df[df.label_index == LabelName.NORMAL].reset_index(drop=True)
        abnormal_df = df[df.label_index == LabelName.ABNORMAL].reset_index(drop=True)

        print(f"'{self.split.value}' split için toplam {len(df)} örnek yüklendi")
        print(f"Normal: {len(normal_df)} | Anormal: {len(abnormal_df)}")

        return normal_df, abnormal_df



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
