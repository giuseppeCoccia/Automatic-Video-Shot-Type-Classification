import tensorflow as tf
import numpy as np
import sys
from utils import *
import argparse

### START EXECUTION
parser = argparse.ArgumentParser(description="Script for testing the retrained architecture")
parser.add_argument('test', nargs='+', help='path to test directory')
parser.add_argument('-c', '--categories', nargs='+', default=['Gros plan', 'Plan moyen', 'Plan rapproche'], help='categories on which has been trained the model, in the order of the probabilities array')

args = parser.parse_args()
test_paths = args.test
categories = args.categories

##### LOAD IMAGES ######
# read images
listimgs, listlabels = [], []
for path in test_paths:
        imgs, labels = read_images(path)
       	listimgs += imgs
        listlabels += labels
loaded_imgs = [load_image(img).reshape((224, 224, 3)) for img in listimgs]
print('Loaded', len(listimgs), 'images and', len(listlabels), 'labels')

##### MODEL #####
sess = tf.Session()

# restore model
new_saver = tf.train.import_meta_graph("new_model.meta")
new_saver.restore(sess, tf.train.latest_checkpoint('./'))
print("Restored Model")

# get tensors
graph = tf.get_default_graph()
features_tensor = graph.get_tensor_by_name("avg_pool:0")
images = graph.get_tensor_by_name("images:0")
features = sess.run(features_tensor, feed_dict = {images: loaded_imgs})
#features = [np.tanh(array) for array in features] # apply tanh to squeeze features
features = [np.log1p(array) for array in features] # apply tanh to squeeze features

bottleneck_input = graph.get_tensor_by_name("BottleneckInputPlaceholder:0")
final_tensor = graph.get_tensor_by_name("final_result:0")

# get probabilities
prob = sess.run(final_tensor, feed_dict = {images: loaded_imgs, bottleneck_input: features})
print(prob)
print([categories[np.argmax(probability)] for probability in prob])
print("Accuracy:", accuracy(listlabels, [categories[np.argmax(probability)] for probability in prob]))
