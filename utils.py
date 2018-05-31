import cv2
import os
import csv



def checkpoint_fn(layers):
    return 'ResNet-L%d.ckpt' % layers

# used to load the pretrained model
def meta_fn(layers):
    return 'ResNet-L%d.meta' % layers

def save_features(features, filename="resnet_training_features.json"):
    # save file with avg_pool output
    with open(filename, "w") as f:
        for feature in features:
            f.write(json.dumps(features) + "\n") # Print features in file "img_features.json"
    print('File save completed')

def load_features(filename="resnet_training_features.json"):
    features = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            features.append(json.loads(line))
    return features


# image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path, size=224, resize=True, grayscale=False):
    img = cv2.imread(path)
    img = cv2.normalize(img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F, dst=None)
    #short_edge = min(img.shape[:2])
    #yy = int((img.shape[0] - short_edge) / 2)
    #xx = int((img.shape[1] - short_edge) / 2)
    #crop_img = img[yy:yy + short_edge, xx:xx + short_edge]
    #resized_img = cv2.resize(crop_img, (size, size))
    if resize == True:
    	img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    if(grayscale):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


# given path to dir, return all the images (and labels) from images of that dir
# if there are subdirs, it goes into them (each subdir is a different label)
def read_images(path_):
	listimgs = list()
	listlabels = list()
	if(os.path.isfile(path_)):
		listimgs.append(path_)
		dirpath = os.path.dirname(path_)	
		if path_[-1] == '/':
			listlabels.append(path_.split('/')[-2])
		else:
			listlabels.append(path_.split('/')[-1])
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


def accuracy(true_labels, predicted_labels):
        correct = 0
        for t, p in zip(true_labels, predicted_labels):
                if t == p: correct += 1
        return (correct/len(true_labels))


def export_csv(loss, train, validation, filename='output.csv'):
	with open(filename, 'w+') as f:
		writer = csv.writer(f, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		writer.writerow(['Loss', 'Train_Accuracy', 'Validation_Accuracy'])
		for l, t, v in zip(loss, train, validation):
			writer.writerow([l, t, v])


