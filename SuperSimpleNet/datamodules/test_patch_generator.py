import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from glob import glob
from tqdm import tqdm

def parse_yolo_polygons(txt_path, img_w, img_h):
    polygons = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            coords = parts[1:len(parts)//2]
            points = [(x * img_w, y * img_h) for x, y in zip(coords[::2], coords[1::2])]
            polygons.append(points)
    return polygons

def extract_patch_centers_from_polyline(polyline, n_points):
    cumulative_lengths = [0]
    for i in range(1, len(polyline)):
        dist = np.linalg.norm(np.array(polyline[i]) - np.array(polyline[i-1]))
        cumulative_lengths.append(cumulative_lengths[-1] + dist)

    total_length = cumulative_lengths[-1]
    step = total_length / (n_points + 1)
    target_distances = [(i + 1) * step for i in range(n_points)]

    result_points = []
    j = 0
    for d in target_distances:
        while j < len(cumulative_lengths) - 1 and cumulative_lengths[j+1] < d:
            j += 1
        ratio = (d - cumulative_lengths[j]) / (cumulative_lengths[j+1] - cumulative_lengths[j])
        pt1 = np.array(polyline[j])
        pt2 = np.array(polyline[j+1])
        interp_point = pt1 + ratio * (pt2 - pt1)
        result_points.append(tuple(interp_point))
    return result_points

def extract_patch(image, cx, cy, patch_size):
    half = patch_size // 2
    h, w = image.shape[:2]
    cx, cy = int(round(cx)), int(round(cy))
    x1 = max(cx - half, 0)
    y1 = max(cy - half, 0)
    x2 = min(cx + half, w)
    y2 = min(cy + half, h)
    patch = image[y1:y2, x1:x2]
    if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
        patch = cv2.copyMakeBorder(
            patch,
            top=max(0, half - cy),
            bottom=max(0, (cy + half) - h),
            left=max(0, half - cx),
            right=max(0, (cx + half) - w),
            borderType=cv2.BORDER_REFLECT
        )
    return patch

def process_test_images(img_dir, mask_dir, label_dir, out_img_dir, out_mask_dir, model_path, patch_size=512, n_patches=10):
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    model = YOLO(model_path)
    img_paths = sorted(glob(os.path.join(img_dir, '*.jpg')))

    for img_path in tqdm(img_paths, desc="Processing test images"):
        img_name = Path(img_path).stem
        mask_path = os.path.join(mask_dir, f"{img_name}.png")

        if not os.path.exists(mask_path):
            print(f"Mask not found: {mask_path}, skipping.")
            continue

        result = model(img_path, save_txt=True, conf=0.25, show=False)[0]
        txt_path = os.path.join(label_dir, f"{img_name}.txt")

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        h, w = image.shape[:2]

        polygons = parse_yolo_polygons(txt_path, w, h)
        if not polygons:
            print(f"No polygons in {txt_path}, skipping.")
            continue

        polyline = polygons[0]
        centers = extract_patch_centers_from_polyline(polyline, n_patches)

        for i, (cx, cy) in enumerate(centers):
            patch = extract_patch(image, cx, cy, patch_size)
            mask_patch = extract_patch(mask, cx, cy, patch_size)

            cv2.imwrite(os.path.join(out_img_dir, f"{img_name}_patch_{i}.jpg"), patch)
            cv2.imwrite(os.path.join(out_mask_dir, f"{img_name}_patch_{i}.png"), mask_patch)

def main():
    process_test_images(
        img_dir="datasets/patched_dataset/test/original_images",
        mask_dir="datasets/patched_dataset/test/original_masks",
        label_dir="runs/detect/results/labels",
        out_img_dir="datasets/patched_dataset/test/images",
        out_mask_dir="datasets/patched_dataset/test/masks",
        model_path="best/best.pt",
        patch_size=512,
        n_patches=10
    )

if __name__ == "__main__":
    main()
