import tensorflow as tf
import numpy as np
import sys
from utils import *
import argparse

import tensorflow_hub as hub

### START EXECUTION
parser = argparse.ArgumentParser(description="Script for testing the retrained architecture")
parser.add_argument('test', nargs='+', help='path to test directory')
parser.add_argument('-hub', '--tfhub_module', type=str, default='https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/1', help='TensorFlow Hub module to use for getting informations about the architecture (height and width of the input)')

args = parser.parse_args()
test_paths = args.test
tfhub = args.tfhub_module

module_spec = hub.load_module_spec(tfhub)
height, width = hub.get_expected_image_size(module_spec)

##### LOAD IMAGES ######
# read images
listimgs, listlabels = [], []
for path in test_paths:
        imgs, labels = read_images(path)
       	listimgs += imgs
        listlabels += labels
loaded_imgs = [load_image(img, size=height).reshape((height, width, 3)) for img in listimgs]
print('Loaded', len(listimgs), 'images and', len(listlabels), 'labels')

u,indices = np.unique(np.array(listlabels), return_inverse=True)
print('Categories: ', u)
num_categories = len(u)

##### MODEL #####
sess = tf.Session()

# restore model
new_saver = tf.train.import_meta_graph("resnet_model.meta")
new_saver.restore(sess, tf.train.latest_checkpoint('./'))
print("Restored Model")

# get tensor
graph = tf.get_default_graph()
features_tensor = graph.get_tensor_by_name("module_apply_default/hub_output/feature_vector/SpatialSqueeze:0")
images = graph.get_tensor_by_name("ImageInput:0")
features = sess.run(features_tensor, feed_dict = {images: loaded_imgs})
bottleneck_input = graph.get_tensor_by_name("BottleneckInput:0")
final_tensor = graph.get_tensor_by_name("final_result:0")
# get probabilities
prob = sess.run(final_tensor, feed_dict = {images: loaded_imgs, bottleneck_input: features})

print(prob)
print([u[np.argmax(probability)] for probability in prob])
print("Accuracy:", accuracy(listlabels, [u[np.argmax(probability)] for probability in prob]))
export_predictions(listimgs, [u[np.argmax(probability)] for probability in prob])
