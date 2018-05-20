import cv2
import os

# image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path, size=224):
    img = cv2.imread(path)
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy + short_edge, xx:xx + short_edge]
    resized_img = cv2.resize(crop_img, (size, size))
    return resized_img


# given path to dir, return all the images (and labels) from images of that dir
# if there are subdirs, it goes into them (each subdir is a different label)
def read_images(path_):
        listimgs = list()
        listlabels = list()
	if(os.path.isfile(path_)):
		listimgs.append(path_)
		dirpath = os.path.dirname(path_)	
		if path[-1] == '/':
			listlabels.append(path.split('/')[-2])
		else:
			listlabels.append(path.split('/')[-1])
		return listimgs, listlabels

        for path, subdirs, files in os.walk(path_):
                for name in files:
                        if ".jpg" in name:
                                listimgs.append(os.path.join(path, name))
                                if path[-1] == '/':
                                        listlabels.append(path.split('/')[-2])
                                else:
                                        listlabels.append(path.split('/')[-1])
        return listimgs, listlabels
