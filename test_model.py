import tensorflow as tf
import numpy as np
import sys
from utils import *
import argparse

import tensorflow_hub as hub

### START EXECUTION
parser = argparse.ArgumentParser(description="Script for testing the retrained architecture")
parser.add_argument('test', nargs='+', help='path to test directory')
parser.add_argument('-c', '--categories', nargs='+', default=['Gros plan', 'Plan moyen', 'Plan rapproche', 'Unknown'], help='categories to test on')
parser.add_argument('-hub', '--tfhub_module', type=str, default='https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/1', help='TensorFlow Hub module to use for getting informations about the architecture (height and width of the input)')
parser.add_argument('-pred', '--prediction_output', nargs='?', type=str, help='name of the output csv file for the predictions on the test set, file is not saved otherwise')
parser.add_argument('-model', '--model', type=str, help='restore given model')

args = parser.parse_args()
test_paths = args.test
tfhub = args.tfhub_module
u = args.categories
pred_out =args.prediction_output
model_to_restore = args.model

module_spec = hub.load_module_spec(tfhub)
height, width = hub.get_expected_image_size(module_spec)
channels = hub.get_num_image_channels(module_spec)

##### LOAD IMAGES ######
# read images
# read paths and labels for each image
listimgs, listlabels = parse_input(test_paths)
loaded_imgs = [load_image(img, size=height).reshape((height, width, channels)) for img in listimgs]
print('[TEST] Loaded', len(loaded_imgs), 'images and', len(listlabels), 'labels')
listlabels = [x if x in u else 'Unknown' for x in listlabels]


##### MODEL #####
sess = tf.Session()

# restore model
new_saver = tf.train.import_meta_graph(model_to_restore+".meta")
new_saver.restore(sess, model_to_restore+".ckpt")
print("Restored Model")

# get tensor
graph = tf.get_default_graph()
features_tensor = graph.get_tensor_by_name("module_apply_default/hub_output/feature_vector/SpatialSqueeze:0")
images = graph.get_tensor_by_name("ImageInput:0")
features = sess.run(features_tensor, feed_dict = {images: loaded_imgs})
bottleneck_input = graph.get_tensor_by_name("BottleneckInput:0")
final_tensor = graph.get_tensor_by_name("final_result:0")
# get probabilities
prob = []
for i in range(0, len(loaded_imgs), 200): # go with batches of 200 for loading features
    prob.extend(sess.run(final_tensor, feed_dict={images: loaded_imgs[i: i+200], bottleneck_input: features[i:i+200]}))

print(np.array(prob))
print([u[np.argmax(probability)] for probability in prob])
print("Accuracy:", accuracy(listlabels, [u[np.argmax(probability)] for probability in prob]))
if pred_out is not None:
    export_predictions(listimgs, [u[np.argmax(probability)] for probability in prob], pred_out)
