#!/bin/bash
ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=width,height,nb_read_frames -of default=nokey=1:noprint_wrappers=1 $1
