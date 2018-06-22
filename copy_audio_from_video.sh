#!/bin/sh

# $1 -> video
# $2 -> audio
# $3 -> output name
ffmpeg -i $1 -i $2 -c copy -map 0:0 -map 1:1 -shortest $3
