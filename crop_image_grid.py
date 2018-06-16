import cv2
import sys
import argparse
import os

parser = argparse.ArgumentParser(description="Script for cropping image")
parser.add_argument('img_path', type=str, help='path to img')
parser.add_argument('pixels', type=int, help='n pixels to be used for the grid n*n')
parser.add_argument('-outdir', '--outdir', nargs='?', type=str, help='path to output dir. if none, it is saved in the same directory of the input image')

# read the image
args = parser.parse_args()
img_path = args.img_path
output_dir = args.outdir
pixels = args.pixels

img_name = os.path.basename(img_path)
name, ext = os.path.splitext(img_name)

if output_dir is None:
    output_dir = os.path.dirname(img_path)

img = cv2.imread(img_path)

len_y = len(img)
len_x = len(img[0])

for i in range(pixels):
    for j in range(pixels):
        new_img = img[i:len_y, j:len_x]
        cv2.imwrite(output_dir+'/'+name+'_cropped_'+str(i)+str(j)+ext, new_img)
