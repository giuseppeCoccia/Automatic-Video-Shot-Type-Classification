import cv2
import sys
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(description="Script for flipping image")
parser.add_argument('img_path', type=str, help='path to img')
parser.add_argument('w', type=int, help='weight of the scaled image')
parser.add_argument('h', type=int, help='heigh of the scaled image')
parser.add_argument('-outdir', '--outdir', nargs='?', type=str, help='path to output dir. if none, it is saved in the same directory of the input image')

args = parser.parse_args()
img_path = args.img_path
w = args.w
h = args.h
output_dir = args.outdir

# read the image
img = cv2.imread(img_path)
img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

# save img
img_name = os.path.basename(img_path)
name, ext = os.path.splitext(img_name)

if output_dir is None:
    output_dir = os.path.dirname(img_path)

cv2.imwrite(output_dir+"/"+name+"_brightned"+ext, img)
