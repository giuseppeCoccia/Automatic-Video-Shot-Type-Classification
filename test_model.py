import tensorflow as tf
import numpy as np
import sys
from utils import *


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
new_saver.restore(sess, tf.train.latest_checkpoint('./'))
print("Restored Model")

graph = tf.get_default_graph()
features_tensor = graph.get_tensor_by_name("avg_pool:0")
images = graph.get_tensor_by_name("images:0")
features = sess.run(features_tensor, feed_dict = {images: loaded_imgs})

bottleneck_input = graph.get_tensor_by_name("BottleneckInputPlaceholder:0")
final_tensor = graph.get_tensor_by_name("final_result:0")
prob = sess.run(final_tensor, feed_dict = {images: batch, bottleneck_input: features})
print(prob[0])
