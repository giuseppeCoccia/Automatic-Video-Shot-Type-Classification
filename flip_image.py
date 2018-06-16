import cv2
import sys
import argparse
import os

parser = argparse.ArgumentParser(description="Script for flipping image")
parser.add_argument('img_path', type=str, help='path to img')
parser.add_argument('-outdir', '--outdir', nargs='?', type=str, help='path to output dir. if none, it is saved in the same directory of the input image')

args = parser.parse_args()
img_path = args.img_path
output_dir = args.outdir

# read the image
img = cv2.imread(img_path)

# copy image to display all 4 variations
horizontal_img = img.copy()
vertical_img = img.copy()
both_img = img.copy()

# flip img horizontally, vertically,
# and both axes with flip()
horizontal_img = cv2.flip( img, 0 )
vertical_img = cv2.flip( img, 1 )
both_img = cv2.flip( img, -1 )

# save img
img_name = os.path.basename(img_path)
name, ext = os.path.splitext(img_name)

if output_dir is None:
    output_dir = os.path.dirname(img_path)

cv2.imwrite(output_dir+"/"+name+"_horizontal"+ext, horizontal_img)
cv2.imwrite(output_dir+"/"+name+"_vertical"+ext, vertical_img)
cv2.imwrite(output_dir+"/"+name+"_both"+ext, both_img)
