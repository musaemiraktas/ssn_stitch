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

    for batch in tqdm(dataloader, desc="Eval RealWorld"):
        image_batch = batch["image"].to(device)
        anomaly_map_batch, anomaly_score_batch = model.forward(image_batch)

        results["anomaly_map"].append(anomaly_map_batch.cpu())
        results["score"].append(torch.sigmoid(anomaly_score_batch).cpu())
        results["image_path"].extend(batch["image_path"])

    results["anomaly_map"] = torch.cat(results["anomaly_map"], dim=0)  # shape: (N, 1, h0, w0)
    results["score"] = torch.cat(results["score"], dim=0)            # shape: (N,)

    return results

def main():
    datamodule = PatchedDataModuleNoMask(
        root=Path("/content/ssn_stitch/SuperSimpleNet/datasets/patched_dataset"),
        image_size=(256, 256),
        eval_batch_size=8,
        num_workers=2,
        seed=42
    )
    datamodule.setup()

    model_path = Path("/content/drive/MyDrive/AP_Bitirme/results300/superSimpleNet/checkpoints/patched_dataset/patched_dataset/weights.pt")
    model = SuperSimpleNet(image_size=datamodule.image_size, config={})
    model.load_model(model_path)

    results = eval_realworld(model=model, dataloader=datamodule.test_dataloader(), device="cuda")

    from visualize_realworld_results import visualize_realworld_results
    visualize_realworld_results(results, save_dir=Path("./visuals"), threshold=0.3)

if __name__ == "__main__":
    main()
