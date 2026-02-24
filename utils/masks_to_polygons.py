import os
import argparse
from concurrent.futures import ThreadPoolExecutor
from ultralytics.data.converter import convert_segment_masks_to_yolo_seg

def CreateDir(path: str) -> None:
    """Creates a directory if it does not already exist.   
        Args:
            path (str): The path of the directory to create.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def MaskToPolygons(dir_to_examine: str, output_dir:str) -> None:
    """
    Converts rescaled segmentation masks in the range [0,1,255] into the YOLO segmentation polygons format. 

    Args:
        dir_to_examine (str): The directory containing the rescaled segmentation masks to convert.
        output_dir (str): The directory where the converted YOLO segmentation polygon labels will be saved
    """
    CreateDir(output_dir)
    convert_segment_masks_to_yolo_seg(masks_dir=dir_to_examine, output_dir=output_dir, classes=1)

def DirectoryProcessor() -> None:
    """
    Process the directories and creates a Threadpool process to convert masks to polygons
    """
    for mod in MODALITY:
        label_dir = f"{ROOT_DIR}/{mod}_{IN_DIR}/labels"
        dest_label_dir = f"{ROOT_DIR}/{mod}_{OUT_DIR}/labels"

        root = os.getcwd() 
        gt_dir = os.path.join(root, label_dir)

        # The list of splits (e.g., test, train and val)
        gt_dir_list = os.listdir(gt_dir)

        # -----------------------------------------------------------
        # LEAVE THIS FOR TESTING
        for i in range( len(gt_dir_list) ):
            input_dir = os.path.join(gt_dir, gt_dir_list[i])        
            output_dir = os.path.join(dest_label_dir, gt_dir_list[i])
            MaskToPolygons(input_dir, output_dir)

        # -----------------------------------------------------------
        # THREADPOOL PARALLEL
        # for i in range( len(gt_dir_list) ):
        #     with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        #         input_dir = os.path.join(gt_dir, gt_dir_list[i])        
        #         output_dir = os.path.join(dest_label_dir, gt_dir_list[i])
        #         executor.submit(MaskToPolygons, input_dir, output_dir)

        ### ----------------------------------------------------------- ###

if __name__ == "__main__": 
    # ------------------------------------------------------------------
    des="""
    Converts rescaled segmentation masks in the range [0,1,255] into the 
    YOLO segmentation polygons format
    """
    # ------------------------------------------------------------------

    MODALITY = ["t1c" , "t1n", "t2f" ,"t2w"] 

    parser = argparse.ArgumentParser(description=des.lstrip(" "), formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--root_dir", type=str, help='root directory of the dataset\t[None]')
    parser.add_argument("--in_dir", type=str, help='input directory of images\t[None]')
    parser.add_argument('--out_dir',type=str,help='output directory prefix\t[None]')
    parser.add_argument('--modality', type=str, choices=MODALITY, nargs='+', help=f'BraTS dataset modalities to use\t[t1c, t1n, t2f, t2w]')
    parser.add_argument('--workers', type=int, help='number of threads/workers to use\t[10]')
    args = parser.parse_args()

    if args.root_dir is not None:
        ROOT_DIR = args.root_dir
    else: ROOT_DIR = "."
    if args.in_dir is not None:
        IN_DIR = args.in_dir
    else: IN_DIR = "segmentation_rescaled"
    if args.out_dir is not None:
        OUT_DIR = args.out_dir
    else: OUT_DIR = "yoloseg"
    if args.workers is not None:
        WORKERS = args.workers
    else: WORKERS = 10
    if args.modality is not None:
        MODALITY = [mod for mod in args.modality]

    DirectoryProcessor()
    print("\nFinish converting binary mask to polygon, check your directory for labels")
