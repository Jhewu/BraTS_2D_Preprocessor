import os
import shutil
import argparse
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

def CreateDir(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def is_positive(label_path, pixel_thres):
    """Return True if the label has more than pixel_thres white pixels."""
    img = np.array(Image.open(label_path))
    return (img > 0).sum() > pixel_thres

def classify_labels(label_dir, pixel_thres):
    """
    Walk label_dir (preserving train/test subdirs) and return two dicts:
        positive[split] = [stem, ...]
        negative[split] = [stem, ...]
    where stem is the filename without extension.
    """
    positive = {}
    negative = {}
    for split in os.listdir(label_dir):
        split_path = os.path.join(label_dir, split)
        if not os.path.isdir(split_path):
            continue
        positive[split] = []
        negative[split] = []
        for fname in os.listdir(split_path):
            fpath = os.path.join(split_path, fname)
            if not os.path.isfile(fpath):
                continue
            stem = os.path.splitext(fname)[0]
            if is_positive(fpath, pixel_thres):
                positive[split].append(stem)
            else:
                negative[split].append(stem)
    return positive, negative

def copy_files(stems, split, src_images_dir, src_labels_dir,
               dst_images_dir, dst_labels_dir):
    """Copy image and label files for the given stems into dst directories."""
    src_img_split = os.path.join(src_images_dir, split)
    src_lbl_split = os.path.join(src_labels_dir, split)
    dst_img_split = os.path.join(dst_images_dir, split)
    dst_lbl_split = os.path.join(dst_labels_dir, split)
    CreateDir(dst_img_split)
    CreateDir(dst_lbl_split)

    for stem in stems:
        # Find matching image file (may differ in extension from label)
        img_copied = False
        for fname in os.listdir(src_img_split):
            if os.path.splitext(fname)[0] == stem:
                shutil.copy2(
                    os.path.join(src_img_split, fname),
                    os.path.join(dst_img_split, fname)
                )
                img_copied = True
                break
        if not img_copied:
            print(f"  [WARN] No matching image found for stem '{stem}' in {src_img_split}")

        # Find matching label file
        lbl_copied = False
        for fname in os.listdir(src_lbl_split):
            if os.path.splitext(fname)[0] == stem:
                shutil.copy2(
                    os.path.join(src_lbl_split, fname),
                    os.path.join(dst_lbl_split, fname)
                )
                lbl_copied = True
                break
        if not lbl_copied:
            print(f"  [WARN] No matching label found for stem '{stem}' in {src_lbl_split}")

def SplitPosNeg():
    images_dir = os.path.join(IN_DIR, "images")
    labels_dir = os.path.join(IN_DIR, "labels")

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"images directory not found: {images_dir}")
    if not os.path.isdir(labels_dir):
        raise FileNotFoundError(f"labels directory not found: {labels_dir}")

    print(f"Classifying labels in: {labels_dir}")
    positive, negative = classify_labels(labels_dir, PIXEL_THRES)

    for split in positive:
        print(f"  [{split}] positive: {len(positive[split])}, negative: {len(negative[split])}")

    pos_images = os.path.join(OUT_DIR, "positive", "images")
    pos_labels = os.path.join(OUT_DIR, "positive", "labels")
    neg_images = os.path.join(OUT_DIR, "negative", "images")
    neg_labels = os.path.join(OUT_DIR, "negative", "labels")

    tasks = []
    for split in positive:
        tasks.append((positive[split], split, images_dir, labels_dir, pos_images, pos_labels))
        tasks.append((negative[split], split, images_dir, labels_dir, neg_images, neg_labels))

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        for args in tasks:
            executor.submit(copy_files, *args)

    print("\nDone. Output written to:", OUT_DIR)

if __name__ == "__main__":
    des = """
    Split a stacked_segmentation dataset into two subsets:
      positive/ — slices whose label contains more than --pixel_thres white pixels
      negative/ — all remaining slices

    Run once per stacked_segmentation_* directory.
    """

    parser = argparse.ArgumentParser(
        description=des.lstrip(" "),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--in_dir", type=str, required=True,
                        help="root dir of a single stacked_segmentation dataset\t[required]")
    parser.add_argument("--out_dir", type=str, default="pos_neg_split",
                        help="destination directory for positive/ and negative/ datasets\t[pos_neg_split]")
    parser.add_argument("--pixel_thres", type=int, default=10,
                        help="pixel count threshold — above this → positive\t[10]")
    parser.add_argument("--workers", type=int, default=8,
                        help="number of copy threads\t[8]")
    args = parser.parse_args()

    IN_DIR = args.in_dir
    OUT_DIR = args.out_dir
    PIXEL_THRES = args.pixel_thres
    WORKERS = args.workers

    SplitPosNeg()
    print("Please verify the output directory.\n")
