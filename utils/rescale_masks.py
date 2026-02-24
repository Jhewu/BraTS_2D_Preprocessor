import os
import cv2
import argparse
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def CreateDir(path: str) -> None:
    """
    Creates a directory if it does not already exist.   
    Args:
        path (str): The path of the directory to create.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    
def RescaleMasks(input_dir: str, output_dir: str) -> None: 
    """
    Rescale segmentation masks from [0, 255] to [0, 1, 255] for YOLO masks_to_polygons.py
    """
    CreateDir(output_dir)
    for file in os.listdir(input_dir):
        if file.endswith(".png"):
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, file)
            unscaled_label = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            scaled_label = (unscaled_label / 255).astype(np.uint8)
            cv2.imwrite(output_path, scaled_label)
            print(f"Rescaled {input_path} and saved to {output_path}")

def ProcessDirectory() -> None: 
    """
    Process the directories and create a Threadpool process to rescale the segmentation
    masks from [0, 255] to [0, 1, 255] for YOLO masks_to_polygons.py
    """
    for mod in MODALITY:
        label_dir = f"{ROOT_DIR}/{mod}_{IN_DIR}/labels"
        dest_label_dir = f"{ROOT_DIR}/{mod}_{OUT_DIR}/labels"

        # The list of splits (e.g., test, train and val)
        gt_dir_list = os.listdir(label_dir)

        # -----------------------------------------------------------
        # LEAVE THIS FOR TESTING
        # for i in range( len(gt_dir_list) ):
        #     input_dir = os.path.join(label_dir, gt_dir_list[i])        
        #     output_dir = os.path.join(dest_label_dir, gt_dir_list[i])
        #     RescaleMasks(input_dir, output_dir)
        # -----------------------------------------------------------
        # THREADPOOL PARALLEL
        for i in range( len(gt_dir_list) ):
            with ThreadPoolExecutor(max_workers=WORKERS) as executor:
                input_dir = os.path.join(label_dir, gt_dir_list[i])        
                output_dir = os.path.join(dest_label_dir, gt_dir_list[i])
                executor.submit(RescaleMasks, input_dir, output_dir)
        # -----------------------------------------------------------

if __name__ == "__main__": 
    # ------------------------------------------------------------------
    des="""
    Rescale segmentation masks from [0, 255] to [0, 1, 255] for YOLO masks_to_polygons.py
    """
    # ------------------------------------------------------------------

    MODALITY = ["t1c" , "t1n", "t2f" ,"t2w"] 

    parser = argparse.ArgumentParser(description=des.lstrip(" "), formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--root_dir", type=str, help='root directory of the dataset\t[None]')
    parser.add_argument("--in_dir", type=str,help='input directory of images\t[None]')
    parser.add_argument('--out_dir',type=str,help='output directory prefix\t[None]')
    parser.add_argument('--modality', type=str, choices=MODALITY, nargs='+', help=f'BraTS dataset modalities to use\t[t1c, t1n, t2f, t2w]')
    parser.add_argument('--workers', type=int, help='number of threads/workers to use\t[10]')
    args = parser.parse_args()

    if args.root_dir is not None:
        ROOT_DIR = args.root_dir
    else: ROOT_DIR = "."
    if args.in_dir is not None:
        IN_DIR = args.in_dir
    else: IN_DIR = "segmentation"
    if args.out_dir is not None:
        OUT_DIR = args.out_dir
    else: OUT_DIR = "segmentation_rescaled"
    if args.workers is not None:
        WORKERS = args.workers
    else: WORKERS = 10
    if args.modality is not None:
        MODALITY = [mod for mod in args.modality]

    ProcessDirectory()
    print("\nFinish rescaling binary mask to mask_to_polygons.py compatible format, check your directory for labels")
