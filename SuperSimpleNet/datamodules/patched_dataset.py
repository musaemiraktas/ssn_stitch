from pathlib import Path
import albumentations as A
import pandas as pd
from anomalib.data.utils import Split, InputNormalizationMethod
from pandas import DataFrame
from datamodules.base.datamodule import SSNDataModule
from datamodules.base.dataset import SSNDataset
import os
import copy
import torch
from pytorch_lightning import seed_everything

class PatchedDataset(SSNDataset):
    """
    Dataset class for your patch-based unsupervised dataset.

    Args:
        root (Path): path to root of dataset
        transform (A.Compose): transforms used for preprocessing
        split (Split): either train or test split
        debug (bool): debug flag for some debug printing
    """

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
        mask_dir = self.root / self.split.value / "masks"
        image_paths = sorted(list(image_dir.glob("*.jpg")))

        samples = []
        for path in image_paths:
            sample_id = path.stem
            mask_path = mask_dir / f"{sample_id}.png"
            if not mask_path.exists():
                mask_path = ""
            samples.append([
                str(image_dir),
                sample_id,
                self.split.value,
                str(path),
                str(mask_path),
                0,  # label_index = NORMAL (training) or just keep it 0 for unsupervised
            ])

        df = pd.DataFrame(samples, columns=[
            "path", "sample_id", "split", "image_path", "mask_path", "label_index"
        ])

        return df, pd.DataFrame()  # no abnormal samples in unsupervised mode


class PatchedDataModule(SSNDataModule):
    """
    Datamodule for your patch-based unsupervised dataset.

    Args:
        root (Path): path to root of dataset
        image_size (tuple[int, int]): image size in format of (h, w)
        normalization (InputNormalizationMethod): normalization method for images
        train_batch_size (int): batch size used in training
        eval_batch_size (int): batch size used in test / inference
        num_workers (int): number of dataloader workers
        seed (int | None): seed for reproducibility
        debug (bool): debug flag
    """

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

        self.train_data = PatchedDataset(
            transform=self.transform_train,
            split=Split.TRAIN,
            root=root,
            debug=debug,
            normal_flips=normal_flips,
        )

        self.test_data = PatchedDataset(
            transform=self.transform_eval,
            split=Split.TEST,
            root=root,
            debug=debug,
        )

def main_patched_dataset(device, config):
    config = copy.deepcopy(config)
    config["dataset"] = config.get("dataset_name", "patched_dataset")
    config["category"] = config["dataset"]
    config["name"] = f"{config['category']}_{config['setup_name']}"

    from model.supersimplenet import SuperSimpleNet
    from common.results_writer import ResultsWriter
    from train import train_and_eval
    from datamodules.patched_dataset import PatchedDataModule

    seed_everything(config["seed"], workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = SuperSimpleNet(image_size=config["image_size"], config=config)

    datamodule = PatchedDataModule(
        root=Path(config["datasets_folder"]) / config["dataset"],
        image_size=config["image_size"],
        train_batch_size=config["batch"],
        eval_batch_size=config["batch"],
        num_workers=config["num_workers"],
        seed=config["seed"],
    )
    datamodule.setup()

    results = train_and_eval(
        model=model,
        datamodule=datamodule,
        config=config,
        device=device,
    )

    results_writer = ResultsWriter(metrics=["AP-det", "AP-loc", "P-AUROC", "I-AUROC"])
    results_writer.add_result(category=config["category"], last=results)
    results_writer.save(
        Path(config["results_save_path"]) / config["setup_name"] / config["dataset"]
    )
