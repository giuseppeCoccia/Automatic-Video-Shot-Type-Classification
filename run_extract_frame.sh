#!/bin/bash

if [ "$#" -ne 3 ]; then
	echo "Usage: ./run_extract_frames.sh path_to_video(no extention) path_to_output_dir -mode"
fi

video_name=$1
path_dir=$2
mode=$3
python3 parse_video_info.py $video_name $mode | ./extract_frames.sh "${video_name}_faces.MP4" $path_dir
