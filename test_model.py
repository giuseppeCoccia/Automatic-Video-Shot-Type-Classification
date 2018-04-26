import tensorflow as tf
import os
import numpy as np
import cv2
import sys



##### UTILS #####

# returns image of shape [224, 224, 3]
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
def read_images(dir):
	listimgs = list()
	listlabels = list()
	for path, subdirs, files in os.walk(dir):
		for name in files:
			if ".jpg" in name:
				listimgs.append(os.path.join(path, name))
				if path[-1] == '/':
					listlabels.append(path.split('/')[-2])
				else:
					listlabels.append(path.split('/')[-1])
	return listimgs, listlabels



### START EXECUTION

##### LOAD IMAGES ######
# read images
listimgs, listlabels = [], []
for path in sys.argv:
	imgs, labels = read_images(path)
	listimgs += imgs
	listlabels += labels
print('Completed loading images names')
print('Loaded', len(listimgs), 'images and', len(listlabels), 'labels')

u,indices = np.unique(np.array(listlabels), return_inverse=True)
print('Categories: ', u)


# load images
loaded_imgs = []
for image in listimgs:
	img = load_image(image)
	batch = img.reshape((224, 224, 3))
	loaded_imgs.append(batch)
print('Completed loading images')



##### MODEL #####
sess = tf.Session()

# restore model
new_saver = tf.train.import_meta_graph("new_model.meta")
new_saver.restore(sess, "new_model.ckpt")
print("Loaded Model")

graph = tf.get_default_graph()
features_tensor = graph.get_tensor_by_name("avg_pool:0")
images = graph.get_tensor_by_name("images:0")

batch_size, num_units_in = features_tensor.get_shape().as_list()
bottleneck_input = tf.placeholder(tf.float32, shape=[batch_size,num_units_in], name='BottleneckInputPlaceholder') # define the input tensor

final_tensor = graph.get_tensor_by_name("final_result:0")
features = sess.run(features_tensor, feed_dict = {images: loaded_imgs[0]})

print("Features loaded")
prob = sess.run(final_tensor, feed_dict = {features_tensor: features})
print(prob[0])
