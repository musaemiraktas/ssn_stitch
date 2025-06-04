import os
from pathlib import Path
import cv2
import numpy as np
import glob
import tqdm

def calculate_polyline_length(polyline):
    return sum(np.linalg.norm(np.array(polyline[i]) - np.array(polyline[i+1])) for i in range(len(polyline)-1))

def parse_yolo_polygons(txt_path, img_w, img_h):
    polygons = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            coords = parts[1:]
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
        ratio = (d - cumulative_lengths[j]) / (cumulative_lengths[j+1] - cumulative_lengths[j] + 1e-8)
        pt1 = np.array(polyline[j])
        pt2 = np.array(polyline[j+1])
        interp_point = pt1 + ratio * (pt2 - pt1)
        result_points.append(tuple(interp_point))
    return result_points

def extract_rotated_patches(image, polyline, n_patches, patch_size=64):
    centers = extract_patch_centers_from_polyline(polyline, n_patches)
    patches = []
    half = patch_size // 2
    h, w = image.shape[:2]
    for (cx, cy) in centers:
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
                borderType=cv2.BORDER_CONSTANT,
                value=0
            )
        patches.append(patch)
    return patches

def process_all_images_multi_poly(image_dir, yolo_labels_dir, output_dir, total_n_patches=10, patch_size=64):
    os.makedirs(output_dir, exist_ok=True)
    image_paths = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
    for img_path in tqdm.tqdm(image_paths, desc="Extracting smart patches"):
        img_name = Path(img_path).stem
        txt_path = os.path.join(yolo_labels_dir, f"{img_name}.txt")
        if not os.path.exists(txt_path):
            continue
        image = cv2.imread(img_path)
        if image is None:
            continue
        h, w = image.shape[:2]
        polygons = parse_yolo_polygons(txt_path, w, h)
        if len(polygons) == 0:
            continue
        lengths = [calculate_polyline_length(poly) for poly in polygons]
        total_length = sum(lengths)
        patch_counts = [
            max(1, round(total_n_patches * (length / total_length))) for length in lengths
        ]
        diff = sum(patch_counts) - total_n_patches
        if diff != 0:
            idx = np.argmax(patch_counts)
            patch_counts[idx] = max(1, patch_counts[idx] - diff)
        patch_index = 0
        for poly, n_patches in zip(polygons, patch_counts):
            patches = extract_rotated_patches(image, poly, n_patches, patch_size=patch_size)
            for patch in patches:
                out_path = os.path.join(output_dir, f"{img_name}_patch_{patch_index}.jpg")
                cv2.imwrite(out_path, patch)
                patch_index += 1

def main():
    base_yolo_runs_dir = "/content/runs/local_test"
    original_image_dir = "/content/drive/MyDrive/AP_Bitirme/bitirme/images"
    total_n_patches = 10
    patch_size = 256
    for yolo_subfolder in sorted(Path(base_yolo_runs_dir).iterdir()):
        if not yolo_subfolder.is_dir():
            continue
        labels_folder = yolo_subfolder / "labels"
        if not labels_folder.exists():
            continue
        txt_files = list(labels_folder.glob("*.txt"))
        if len(txt_files) == 0:
            print(f"{yolo_subfolder.name} içinde dikiş bölgesi tespiti yapılamadı")
            continue
        patches_output_dir = yolo_subfolder / "patches"
        process_all_images_multi_poly(
            image_dir=original_image_dir,
            yolo_labels_dir=str(labels_folder),
            output_dir=str(patches_output_dir),
            total_n_patches=total_n_patches,
            patch_size=patch_size
        )

if __name__ == "__main__":
    main()