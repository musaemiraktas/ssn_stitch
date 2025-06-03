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

        # GPU -> CPU (tensörleri toplayalım)
        results["anomaly_map"].append(anomaly_map_batch.cpu())
        results["score"].append(torch.sigmoid(anomaly_score_batch).cpu())
        results["image_path"].extend(batch["image_path"])

    # Katmanlı tensörleri tek tensöre çevir
    results["anomaly_map"] = torch.cat(results["anomaly_map"], dim=0)  # shape: (N, 1, h0, w0)
    results["score"] = torch.cat(results["score"], dim=0)            # shape: (N,)

    return results

def main():
    # 1) Datamodule'u hazırlayın
    datamodule = PatchedDataModuleNoMask(
        root=Path("/content/ssn_stitch/SuperSimpleNet/datasets/patched_dataset"),
        image_size=(512, 512),
        eval_batch_size=8,
        num_workers=2,
        seed=42
    )
    datamodule.setup()

    # 2) Modeli yükleyin
    model_path = Path("/content/drive/MyDrive/AP_Bitirme/results300/superSimpleNet/checkpoints/patched_dataset/patched_dataset/weights.pt")
    model = SuperSimpleNet(image_size=datamodule.image_size, config={})
    model.load_model(model_path)

    # 3) Eval'i yapın
    results = eval_realworld(model=model, dataloader=datamodule.test_dataloader(), device="cuda")

    # 4) Görselleştirme (threshold=0.3 veya kendi belirlediğiniz persentil vs.)
    from visualize_realworld_results import visualize_realworld_results
    visualize_realworld_results(results, save_dir=Path("./visuals"), threshold=0.3)

if __name__ == "__main__":
    main()
