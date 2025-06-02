from pathlib import Path
import albumentations as A
import pandas as pd
from anomalib.data.utils import Split, InputNormalizationMethod
from pandas import DataFrame
from datamodules.base.datamodule import SSNDataModule
from datamodules.base.dataset import SSNDataset
import os
import torch
import cv2


class PatchedDatasetNoMask(SSNDataset):
    def __init__(
        self,
        root: Path,
        transform: A.Compose,
        split: Split,
        normal_flips: bool = False,
        debug: bool = False,
    ) -> None:
        super().__init__(
            transform=transform,
            root=root,
            split=split,
            flips=False,
            normal_flips=normal_flips,
            supervised=False,
            debug=debug,
        )

    def make_dataset(self) -> tuple[DataFrame, DataFrame]:
        image_dir = self.root / self.split.value / "images"
        image_paths = sorted(list(image_dir.glob("*.jpg")))

        samples = []
        for path in image_paths:
            sample_id = path.stem
            label_index = 0  # gerçek hayatta hepsi "kusursuz" varsayılır
            mask_path = ""   # maske kullanılmıyor

            samples.append([
                str(image_dir),
                sample_id,
                self.split.value,
                str(path),
                mask_path,
                label_index,
            ])

        df = pd.DataFrame(samples, columns=[
            "path", "sample_id", "split", "image_path", "mask_path", "label_index"
        ])

        return df, pd.DataFrame()  # no abnormal samples


class PatchedDataModuleNoMask(SSNDataModule):
    def __init__(
        self,
        root: Path | str,
        image_size: tuple[int, int],
        normalization: str | InputNormalizationMethod = InputNormalizationMethod.IMAGENET,
        train_batch_size: int = 8,
        eval_batch_size: int = 8,
        num_workers: int = 0,
        seed: int | None = None,
        normal_flips: bool = False,
        debug: bool = False,
    ) -> None:
        print(f"📂 [PatchedDataModuleNoMask] Resolution set to: {image_size}")

        super().__init__(
            root=root,
            supervised=False,
            image_size=image_size,
            normalization=normalization,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            seed=seed,
            flips=False,
        )

        # Sadece test verisi tanımlanır
        self.test_data = PatchedDatasetNoMask(
            transform=self.transform_eval,
            split=Split.TEST,
            root=root,
            debug=debug,
        )