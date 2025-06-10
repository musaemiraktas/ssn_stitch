from pathlib import Path
import pandas as pd
import albumentations as A
from anomalib.data.utils import Split, InputNormalizationMethod
from datamodules.base.dataset import SSNDataset
from datamodules.base.datamodule import SSNDataModule


class StitchUnsupervisedDataset(SSNDataset):
    """
    Unsupervised dataset for custom stitch/task dataset.

    Expected folder structure:
        root/
          train/
            images/    # only defect-free (good) images
          test/
            images/    # both defect-free and defective images
            masks/     # binary .png mask per test image
    """

    def __init__(
        self,
        root: Path,
        transform: A.Compose,
        split: Split,
        debug: bool = False,
    ) -> None:
        super().__init__(
            transform=transform,
            root=root,
            split=split,
            flips=False,
            normal_flips=False,
            supervised=False,
            debug=debug,
        )
        self.root_split = Path(root) / split.value

    def make_dataset(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        img_dir = self.root_split / "images"
        image_paths = sorted([p for p in img_dir.glob("*") if p.is_file()])
        df = pd.DataFrame({"image_path": [str(p) for p in image_paths]})

        if self.split == Split.TEST:
            mask_dir = self.root_split / "masks"
            mask_map = {p.stem: str(p) for p in mask_dir.glob("*") if p.is_file()}
            df["mask_path"] = df["image_path"].apply(
                lambda x: mask_map.get(Path(x).stem, "")
            )
        else:
            df["mask_path"] = [""] * len(df)
        return df, pd.DataFrame()


class Stitch(SSNDataModule):
    """
    LightningDataModule for unsupervised stitch dataset.
    """

    def __init__(
        self,
        root: Path | str,
        image_size: tuple[int, int] | None = None,
        normalization: str | InputNormalizationMethod = InputNormalizationMethod.IMAGENET,
        train_batch_size: int = 8,
        eval_batch_size: int = 8,
        num_workers: int = 0,
        seed: int | None = None,
        debug: bool = False,
    ) -> None:
        print(f"Resolution set to: {image_size}")
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

        self.train_data = StitchUnsupervisedDataset(
            root=Path(root),
            transform=self.transform_train,
            split=Split.TRAIN,
            debug=debug,
        )
        self.test_data = StitchUnsupervisedDataset(
            root=Path(root),
            transform=self.transform_eval,
            split=Split.TEST,
            debug=debug,
        )
