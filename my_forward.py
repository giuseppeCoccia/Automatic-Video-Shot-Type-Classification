#from convert import print_prob, load_image, checkpoint_fn, meta_fn
import tensorflow as tf
import json
import os
import numpy as np



# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path, size=224):
    img = skimage.io.imread(path)
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy + short_edge, xx:xx + short_edge]
    resized_img = skimage.transform.resize(crop_img, (size, size))
    return resized_img


def checkpoint_fn(layers):
    return 'ResNet-L%d.ckpt' % layers


def meta_fn(layers):
    return 'ResNet-L%d.meta' % layers





layers = 152

sess = tf.Session()

new_saver = tf.train.import_meta_graph(meta_fn(layers))
new_saver.restore(sess, checkpoint_fn(layers))

dir = "../Data/Images_Plans"
listimgs = list()
for path, subdirs, files in os.walk(dir):
	for name in files:
		if ".jpg" in name:
			listimgs.append(os.path.join(path, name))
print('Completed loading images names')
			
loaded_imgs = []
for image in listimgs:
	img = load_image(image)
	batch = img.reshape((1, 224, 224, 3))
	loaded_imgs.append(batch)
print('Completed loading images')
	
graph = tf.get_default_graph()
features_tensor = graph.get_tensor_by_name("avg_pool:0")
filename = "img_features.json" # Name of the file where you will print your image features
images = graph.get_tensor_by_name("images:0") # Not 100% sure of the name of the tensor
feed_dict = {images: np.array(loaded_imgs)}
features = sess.run(features_tensor, feed_dict=feed_dict) # Run the ResNet on loaded images
print('Completed running ResNet')

with open(filename, "w") as f:
    for i in range(len(loaded_imgs)):
        feats_i = features[i].tolist()
        res = [listimgs[i], feats_i]
        f.write(json.dumps(res) + "\n") # Print features in file "img_features.json"
