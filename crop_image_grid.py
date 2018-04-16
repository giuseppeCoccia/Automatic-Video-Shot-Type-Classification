import cv2
import sys


# read the image
img_name = sys.argv[1]
output_dir = sys.argv[2]
img = cv2.imread(img_name)


len_y = len(img)
len_x = len(img[0])

pixels = int(sys.argv[2])

for i in range(pixels):
    for j in range(pixels):
        new_img = img[i:len_y, j:len_x]
        cv2.imwrite(output_dir+'/cropped_'+str(i)+str(j)+'_'+img_name, crop_img)
