#!/bin/bash

## -------------------------------------------------
## 	This script it's for processing BraTS dataset for
## both YOLO object detection, YOLO object segmentation
## and UNET Training with K-fold cross validation
## All four modality is stacked onto each PNG channel

## 	After running this bash script, it will create
## temporary directories and automatically remove them

## 	There will be K instances of (1) stacked_detection_k, and (2) 
## stacked_segmentation_k

## 	Feel free to COMMENT or UNCOMMENT if you only want
## to process certain datasets. Keep in mind, the YOLO
## detection and segmentation dataset depends on the UNet
## segmentation dataset, therefore UNet must run first
## -------------------------------------------------

## --- Variables to Set (YOU NEED TO SET THIS) --- ##
K=3
## --- Variables to Set (YOU NEED TO SET THIS) --- ##

# Create K fold split (NECESSARY)
python3 utils/split_k_fold_dataset.py --in_dir ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2 --out_dir dataset_split --k ${K}

for ((i=0; i<K; i++))
do
	## --- For UNet Segmentation (ALWAYS ON) --- ##
	python3 utils/brats_2d_slicer.py --in_dir "dataset_split_${i}"
	python3 utils/crop_clean_binarize.py --out_dir "output_${i}"

	python3 utils/stack_images.py --in_dir "output_${i}" --out_dir "stacked_segmentation_${i}" --dataset segmentation
	python3 utils/copy_labels.py --in_dir "output_${i}" --out_dir "stacked_segmentation_${i}" --dataset segmentation

	echo -e "\nFinished with Segmentation!!!\n"
	## --- For UNet Segmentation (ALWAYS ON) --- ##
	
	## --- For YOLO Object Detection (COMMENT OR UNCOMMENT) --- ##
	python3 utils/masks_to_boxes.py --in_dir "output_${i}" --out_dir "output_${i}"
	python3 utils/copy_training_img.py --in_dir "output_${i}" --out_dir "output_${i}" --dataset_to_copy_from segmentation --dataset_to_copy_to detection

	python3 utils/stack_images.py --in_dir "output_${i}" --out_dir "stacked_detection_${i}" --dataset detection
	python3 utils/copy_labels.py --in_dir "output_${i}" --out_dir "stacked_detection_${i}" --dataset detection

	echo -e "\nFinished with Detection!!!\n"
	## --- For YOLO Object Detection (COMMENT OR UNCOMMENT) --- ##

	## --- For YOLO Segmentation (COMMENT OR UNCOMMENT) --- ##
	python utils/rescale_masks.py --root_dir "output_${i}" --in_dir segmentation --out_dir segmentation_rescaled
	python3 utils/masks_to_polygons.py --root_dir "output_${i}" --in_dir segmentation_rescaled --out_dir yoloseg
	python3 utils/copy_training_img.py --in_dir "output_${i}" --out_dir "output_${i}" --dataset_to_copy_from segmentation --dataset_to_copy_to yoloseg

	python3 utils/stack_images.py --in_dir "output_${i}" --out_dir "stacked_yoloseg_${i}" --dataset yoloseg
	python3 utils/copy_labels.py --in_dir "output_${i}" --out_dir "stacked_yoloseg_${i}" --dataset yoloseg

	echo -e "\nFinished with YOLO Segmentation!!!\n"
	## --- For YOLO Segmentation (COMMENT OR UNCOMMENT) --- ##

	# Remove Temporary Objects (NECESSARY)
	rm -r ./t1c
	rm -r ./t1n
	rm -r ./t2f
	rm -r ./t2w
 
done

### --------- Remove reusable components (Optional) --------- ###
# for ((i=0; i<K; i++)) 
# do 
# 	rm -r "output_${i}"
# 	rm -r "./dataset_split_${i}"
# done
### --------- Remove reusable components (Optional) --------- ###
