import cv2
import sys
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(description="Script for flipping image")
parser.add_argument('img_path', type=str, help='path to img')
parser.add_argument('value', type=int, help='path to img')
parser.add_argument('-outdir', '--outdir', nargs='?', type=str, help='path to output dir. if none, it is saved in the same directory of the input image')

args = parser.parse_args()
img_path = args.img_path
value = args.value
output_dir = args.outdir

# read the image
img = cv2.imread(img_path)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #convert it to hsv

h, s, v = cv2.split(hsv)
v = np.where((255-v)<value,255,v+value)
final_hsv = cv2.merge((h, s, v))

img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

# save img
img_name = os.path.basename(img_path)
name, ext = os.path.splitext(img_name)

if output_dir is None:
    output_dir = os.path.dirname(img_path)

cv2.imwrite(output_dir+"/"+name+"_brightned"+ext, img)
