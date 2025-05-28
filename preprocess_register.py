### Applying Contrast Enhancement & Edge Enhancement ###


import os
import argparse
import numpy as np
from PIL import Image, ImageEnhance
import cv2


def enhance_contrast(img_pil, factor=1.8):
    enhancer = ImageEnhance.Contrast(img_pil)
    return enhancer.enhance(factor)


def apply_edge_enhancement(img_pil):
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)

    laplacian_colored = cv2.merge([laplacian]*3)
    enhanced = cv2.addWeighted(img_cv, 0.8, laplacian_colored, 0.5, 0)
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    return Image.fromarray(enhanced_rgb)


def process_directory(input_dir, output_dir, contrast_factor=1.8):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        input_path = os.path.join(input_dir, fname)
        output_path = os.path.join(output_dir, fname)

        try:
            img = Image.open(input_path).convert("RGB")
            img = enhance_contrast(img, factor=contrast_factor)
            img = apply_edge_enhancement(img)
            img.save(output_path)
            print(f"Processed: {fname}")
        except Exception as e:
            print(f"Failed: {fname} → {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Contrast + Edge Filtering for Register Images")
    parser.add_argument('--input_dir', type=str,
                        default='data/footprints/train/register',
                        help='Path to input directory containing original register images')
    parser.add_argument('--output_dir', type=str,
                        default='data/footprints/train_filtered/register_filtered',
                        help='Path to output directory for processed images')
    parser.add_argument('--contrast_factor', type=float,
                        default=1.8,
                        help='Contrast enhancement factor (default=1.8)')

    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir, contrast_factor=args.contrast_factor)
