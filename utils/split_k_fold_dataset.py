import os
import argparse
from math import ceil
from random import shuffle, seed
import shutil
import threading

def createDir(folder_name):
   if not os.path.exists(folder_name):
       os.makedirs(folder_name) 
       
def copyFile(dataset_dir_list, dataset_dir, dataset_dest):
    # dir is the patient directory
    for dir in dataset_dir_list:
        dir_to_copy = os.path.join(dataset_dir, dir)
        dir_to_copy_to = os.path.join(dataset_dest, dir)
        if os.path.exists(dir_to_copy):
            shutil.copytree(dir_to_copy, dir_to_copy_to)
        else:
            print(f"Source directory does not exist: {dir_to_copy}")

# ------------------------------------------------------------------------------
# Main Runtime
# ------------------------------------------------------------------------------
def SplitKFoldDataset(): 
    root_dir = os.getcwd()
    dataset_dir = os.path.join(root_dir, IN_DIR)
    dataset_dir_list = os.listdir(dataset_dir)

    # Report information
    dataset_length = len(dataset_dir_list)
    print(f"There is a total of: {dataset_length} patients in the directory\n")

    # Update the seed and shuffle the list
    seed(SEED)
    shuffle(dataset_dir_list)

    # Create the k-folds lists
    fold_size = dataset_length//K
    
    for fold in range(K): 
        # Create destination directories
        dest_dir = os.path.join(root_dir, f"{OUT_DIR}_{fold}/")
        dest_train = os.path.join(dest_dir, "train") ; createDir(dest_train)
        dest_test = os.path.join(dest_dir, "test") ; createDir(dest_test)

        # Establish the start and end indices
        start = fold * fold_size
        end = start + fold_size

        # Create the test/train lists of directories
        test_list = dataset_dir_list[start:end] # <- Select the range
        train_list = dataset_dir_list[:start] + dataset_dir_list[end:] # <- Wrap around
 
        # Define the threads for copying directories
        threads = []
    
        train_thread = threading.Thread(target=copyFile, args=(train_list, dataset_dir, dest_train))
        threads.append(train_thread)
        train_thread.start()
    
        test_thread = threading.Thread(target=copyFile, args=(test_list, dataset_dir, dest_test))
        threads.append(test_thread)
        test_thread.start()   

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

    print("\nAll directories copied successfully.")

if __name__ == "__main__": 
    # -------------------------------------------------------------

    des="""
    This script creates a k-fold cross-validation dataset for a BraTS 
    dataset, resulting in k directories where each fold it's reserved
    for testing at least once. 
    """

    # -------------------------------------------------------------

    parser = argparse.ArgumentParser(description=des.lstrip(" "), formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--in_dir", type=str, help='Input directory of images\t[ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2]')
    parser.add_argument('--out_dir',type=str, help='Output directory prefix\t[dataset_split]')
    parser.add_argument('--seed',   type=int, help='Shuffling seed\t[42]')
    parser.add_argument('--k',      type=int, help='K parameter for k-fold cross validation\t[4]')
    args = parser.parse_args()

    if args.in_dir is None: 
        raise IOError
        
    IN_DIR = args.in_dir
    OUT_DIR = args.out_dir or "dataset_split"
    SEED = args.seed or 42
    K = args.k or 3

    SplitKFoldDataset()
    print("\nFinish splitting the k-fold cross-validation dataset, please check your directory...\n")
