import torch
from tqdm import tqdm
from pathlib import Path

from model.supersimplenet import SuperSimpleNet
from datamodules.patched_nomask_dataset import PatchedDataModuleNoMask

@torch.no_grad()
def eval_realworld(model, dataloader, device: str):
    model.to(device)
    model.eval()

    results = {
        "image_path": [],
        "score": [],
        "anomaly_map": [],
    }

    for batch in tqdm(dataloader):
        image_batch = batch["image"].to(device)
        anomaly_map, anomaly_score = model.forward(image_batch)

        results["anomaly_map"].append(anomaly_map.cpu())
        results["score"].append(torch.sigmoid(anomaly_score).cpu())
        results["image_path"].extend(batch["image_path"])

    results["anomaly_map"] = torch.cat(results["anomaly_map"])
    results["score"] = torch.cat(results["score"])

    for path, score in zip(results["image_path"], results["score"]):
        print(f"{path} -> score: {score.item():.4f}")

    return results


if __name__ == "__main__":
    datamodule = PatchedDataModuleNoMask(
        root=Path("/content/ssn_stitch/SuperSimpleNet/datasets/patched_dataset"),
        image_size=(512, 512),
        eval_batch_size=8,
        num_workers=2,
        seed=42
    )
    datamodule.setup()


    model_path = Path("/content/drive/MyDrive/AP_Bitirme/results300/superSimpleNet/checkpoints/patched_dataset/patched_dataset/weights.pt")
    model = SuperSimpleNet(image_size=datamodule.image_size, config={})
    model.load_model(model_path)


    eval_realworld(model=model, dataloader=datamodule.test_dataloader(), device="cuda")