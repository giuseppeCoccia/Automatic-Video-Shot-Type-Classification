from convert import print_prob, load_image, checkpoint_fn, meta_fn
import tensorflow as tf
import json
import os
import numpy as np

layers = 152

sess = tf.Session()

new_saver = tf.train.import_meta_graph(meta_fn(layers))
new_saver.restore(sess, checkpoint_fn(layers))

dir = "/home/giuseppe/Desktop/flower_photos"
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