import tensorflow as tf
import json
import os
import numpy as np
import cv2
from utils import *
import argparse
from sklearn.utils import shuffle

epochs = 20
learning_rate = 0.0001
FC_WEIGHT_STDDEV = 0.01


##### UTILS #####
# cross entropy loss, as it is a classification problem it is better
def loss(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    
    #regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    #loss_ = tf.add_n([cross_entropy_mean] + regularization_losses)
    loss_ = cross_entropy_mean
    #tf.summary.scalar('loss', loss_)
    
    return loss_


### START EXECUTION
parser = argparse.ArgumentParser(description="Script for retraining last layer of the resnet architecture")
parser.add_argument('-a', '--arch', nargs='?', type=int, default=50, choices=[50, 152], help='chose pretrained model')
parser.add_argument('-t', '--train', nargs='+', help='paths to training directories', required=True)
parser.add_argument('-v', '--validation', nargs='+', help='paths to validation directory', required=True)
parser.add_argument('-test', nargs='+', help='path to test directory')

args = parser.parse_args()
train_paths = args.train
validation_paths = args.validation
test_paths = args.test


##### LOAD IMAGES ######

### training images
# read images
listimgs, listlabels = [], []
for path in train_paths:
	imgs, labels = read_images(path)
	listimgs += imgs[:42]
	listlabels += labels[:42]

# load images
loaded_imgs = [load_image(img).reshape((224, 224, 3)) for img in listimgs]
print('[TRAINING] Loaded', len(loaded_imgs), 'images and', len(listlabels), 'labels')


### validation images
listimgs_v, listlabels_v = [], []
for path in validation_paths:
        imgs, labels = read_images(path)
        listimgs_v += imgs
        listlabels_v += labels
loaded_imgs_v = [load_image(img).reshape((224, 224, 3)) for img in listimgs_v]
print('[VALIDATION] Loaded', len(loaded_imgs_v), 'images and', len(listlabels_v), 'labels')







##### MODEL #####
# load from existing model and retrain last layer
layers = args.arch # model to be loaded


sess = tf.Session()

# restore model
new_saver = tf.train.import_meta_graph(meta_fn(layers))
new_saver.restore(sess, checkpoint_fn(layers))
print("Completed restoring pretrained model")

# load last-but-one (layer) tensor after feeding images
graph = tf.get_default_graph()
features_tensor = graph.get_tensor_by_name("avg_pool:0")
images = graph.get_tensor_by_name("images:0")

#print('Completed running ResNet') # features now containes avg_pool output




#### RETRAINING LAST LAYER #####

## placeholders and variables

# map string labels to unique integers
u,indices = np.unique(np.array(listlabels), return_inverse=True)
print('Categories: ', u)
num_categories = len(u)

# get avg pool dimensions
batch_size, num_units_in = features_tensor.get_shape().as_list()

# define placeholder that will contain the inputs of the new layer
bottleneck_input = tf.placeholder(tf.float32, shape=[batch_size,num_units_in], name='BottleneckInputPlaceholder') # define the input tensor
# define placeholder for the categories
labelsVar = tf.placeholder(tf.int32, shape=[batch_size], name='labelsVar')


# weights and biases
weights_initializer = tf.truncated_normal_initializer(stddev=FC_WEIGHT_STDDEV)
weights = tf.get_variable('weights', shape=[num_units_in, num_categories], initializer=weights_initializer)
biases = tf.get_variable('biases', shape=[num_categories], initializer=tf.zeros_initializer)

# operations
logits = tf.matmul(bottleneck_input, weights)
logits = tf.nn.bias_add(logits, biases)
final_tensor = tf.nn.softmax(logits, name="final_result")

loss_ = loss(logits, labelsVar)
ops = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = ops.minimize(loss_)

# run training session
init=tf.global_variables_initializer()
sess.run(init)

for epoch in range(epochs):
	# shuffle dataset
	X_train, y_train = shuffle(loaded_imgs, indices)

	# get features and optimize
	features = sess.run(features_tensor, feed_dict={images: X_train}) # Run the ResNet on loaded images
	# save file with avg_pool output
	#save_features(features)
	# apply tanh transformation
	features = [np.tanh(array) for array in features]
	# run session
	_, loss = sess.run([train_op, loss_], feed_dict={bottleneck_input: features, labelsVar: y_train})

	# print accuracy
	features = sess.run(features_tensor, feed_dict = {images: loaded_imgs_v})
	features = [np.tanh(array) for array in features]
	prob = sess.run(final_tensor, feed_dict = {bottleneck_input: features})
	acc = accuracy(listlabels_v, [u[np.argmax(probability)] for probability in prob])
	print(epoch+1, ": Loss", loss, "- Training Accuracy", acc)
print("Completed training")


# saving new model
tf.train.export_meta_graph(filename='new_model.meta')
saver = tf.train.Saver()
save_path = saver.save(sess, "new_model.ckpt")
print("Model saved")











#### TEST ####
if test_paths is not None:
	print("Starting test")	

	# read images
	listimgs_t, listlabels_t = [], []
	for path in test_paths:
		imgs, labels = read_images(path)
		listimgs_t += imgs
		listlabels_t += labels
	loaded_imgs_t = [load_image(img).reshape((224, 224, 3)) for img in listimgs_t]
	print('[TEST] Loaded', len(loaded_imgs_t), 'images and', len(listlabels_t), 'labels')	
	
	# test
	features = sess.run(features_tensor, feed_dict = {images: loaded_imgs_t})
	features = [np.tanh(array) for array in features]
	prob = sess.run(final_tensor, feed_dict = {bottleneck_input: features})
	print("PROB:", prob)
	print([u[np.argmax(probability)] for probability in prob])
	print("Accuracy:", accuracy(listlabels_t, [u[np.argmax(probability)] for probability in prob]))
