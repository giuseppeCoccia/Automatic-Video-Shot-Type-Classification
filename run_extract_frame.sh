#!/bin/sh

video_name=$1
parse_video_info.py '../Data/Videos/'+$video_name | extract_frames.sh $video_name+'.mp4'
