#!/bin/bash

video_name=$1
python3 parse_video_info.py $video_name | ./extract_frames.sh "${video_name}_faces.MP4" '../Data/Videos/extracted_frames'
