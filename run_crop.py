import sys
import os

directory = sys.argv[1]
pixels = 7

for path, subdirs, files in os.walk(directory):
    for name in files:
        if ".jpg" in name:
            img_path = os.path.join(path, name)
            cropped_path = directory+"cropped_"+path.split("/")[-1]
            if not os.path.exists(cropped_path):
                os.makedirs(cropped_path)
            os.system('python3 crop_image_grid.py "'+img_path+'" '+str(pixels)+' "'+cropped_path+'/"')

