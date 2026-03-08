import os
import argparse
import numpy as np
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify negative labels: prints per-image white pixel count and folder average."
    )
    parser.add_argument("--in_dir", type=str, required=True,
                        help="directory of negative label images")
    args = parser.parse_args()

    files = [f for f in os.listdir(args.in_dir) if os.path.isfile(os.path.join(args.in_dir, f))]

    if not files:
        print("No files found in directory.")
        exit(1)

    counts = []
    for fname in sorted(files):
        img = np.array(Image.open(os.path.join(args.in_dir, fname)))
        white_pixels = int((img > 0).sum())
        counts.append(white_pixels)
        print(f"{fname}: {white_pixels} white pixels")

    print(f"\nTotal images : {len(counts)}")
    print(f"Average white pixels: {np.mean(counts):.4f}")
    print(f"Max white pixels    : {max(counts)}")
