#!/bin/bash

directories=$(find $1 ! -path $1 -type d)
base_label="cropped"
pixels=7

for d in $directories
do
	files=$(find "$d" -type f)
	label=$(basename "$d")

	newdir="$1${base_label}_${label}/"
	mkdir $newdir

	for img in $files
	do
		echo $img
		python3 crop_image_grid.py $img $pixels $newdir
	done
done
