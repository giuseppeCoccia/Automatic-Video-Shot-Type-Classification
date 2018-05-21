import tensorflow as tf
import time
import os
import numpy as np
import cv2
from utils import *
import argparse
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten

# Functions for weigths and bias initilization
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0., shape=shape)
	return tf.Variable(initial)

def LeNet5(image):
	# your inmplementation goes here

	# Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
	conv1_W = weight_variable(shape=(5, 5, 1, 6))
	conv1_b = bias_variable(shape = [6])
	conv1_output = tf.nn.conv2d(image, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
	# Activation.
	conv1_output = tf.nn.relu(conv1_output)
	# Pooling. Input = 28x28x6. Output = 14x14x6.
	conv1_output = tf.nn.max_pool(conv1_output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

	# Layer 2: Convolutional. Input = 14x14x6. Output = 10x10x16.
	conv2_W = weight_variable(shape=(5, 5, 6, 16))
	conv2_b = bias_variable(shape = [16])
	conv2_output   = tf.nn.conv2d(conv1_output, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
	# Activation.
	conv2_output = tf.nn.relu(conv2_output)
	# Pooling. Input = 10x10x16. Output = 5x5x16.
	conv2_output = tf.nn.max_pool(conv2_output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

	# Flatten. Input = 5x5x16. Output = 400.
	fc0_output = flatten(conv2_output)  #tf.reshape(conv2_output, [-1])

	# Layer 3: Fully Connected. Input = 400. Output = 120.
	fc1_W = weight_variable(shape=(400, 120))
	fc1_b = bias_variable(shape = [120])
	fc1_output = tf.matmul(fc0_output, fc1_W) + fc1_b
	# Activation.
	fc1_output = tf.nn.relu(fc1_output)

	# Layer 4: Fully Connected. Input = 120. Output = 84.
	fc2_W = weight_variable(shape=(120, 84))
	fc2_b = bias_variable(shape = [84])
	fc2_output = tf.matmul(fc1_output, fc2_W) + fc2_b
	# Activation.
	fc2_output = tf.nn.relu(fc2_output)

	# Layer 5: Fully Connected. Input = 84. Output = 10.
	fc3_W = weight_variable(shape=(84, 3))
	fc3_b = bias_variable(shape = [3])
	fc3_output = tf.matmul(fc2_output, fc3_W) + fc3_b

	return fc3_output

# encoding
def encode(labels):
	u, indices = np.unique(np.array(labels), return_inverse=True)
	onehot_y = np.zeros((indices.size, indices.max()+1))
	onehot_y[np.arange(indices.size), indices] = 1
	return u, onehot_y

def evaluate(logits, labels, batch_size=32):
    # logits will be the outputs of your model, labels will be one-hot vectors corresponding to the actual labels
    # logits and labels are numpy arrays
    # this function should return the accuracy of your model
    num_examples = len(logits)
    total_accuracy = 0
    sess = tf.get_default_session()
    
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = logits[offset:offset+batch_size], labels[offset:offset+batch_size]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
        
    return total_accuracy / num_examples

def train(init, sess, n_epochs, batch_size, optimizer, cost, merged_summary_op):
	# optimizer and cost are the same kinds of objects as in Section 1
	# Train your model
	global X_train, y_train, X_validation, y_validation
	sess.run(init)
	# op to write logs to Tensorboard
	#summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
	# Training cycle
	for epoch in range(n_epochs):
		avg_cost = 0.
		total_batch = int(len(X_train)/batch_size) if len(X_train) % batch_size == 0 else int(len(X_train)/batch_size)+1
		# Loop over all batches
		X_train, y_train = shuffle(X_train, y_train)
		for offset in range(0, len(X_train), batch_size):
			batch_xs, batch_ys = X_train[offset:offset+batch_size], y_train[offset:offset+batch_size]
			# Run optimization op (backprop), cost op (to get loss value)
			# and summary nodes
			_, c, summary = sess.run([optimizer, cost, merged_summary_op],
									 feed_dict={x: batch_xs, y: batch_ys})
									 # Write logs at every iteration
									 #summary_writer.add_summary(summary, epoch * total_batch + i)
									 # Compute average loss
			avg_cost += c / total_batch
		# Display logs per epoch step
		if (epoch+1) % display_step == 0:
			print("Epoch: ", '%02d' % (epoch+1), "  =====> Loss=", "{:.9f}".format(avg_cost),
				  "  =====> Training Accuracy=", evaluate(X_train, y_train),
				  "  =====> Validation Accuracy=", evaluate(X_validation, y_validation))

print("Optimization Finished!")
#summary_writer.flush()


### START EXECUTION
parser = argparse.ArgumentParser(description="Script for retraining last layer of the resnet architecture")
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
loaded_imgs = [load_image(img, size=32, grayscale=True).reshape((32, 32, 1)) for img in listimgs]
print('[TRAINING] Loaded', len(loaded_imgs), 'images', loaded_imgs[0].shape, 'and', len(listlabels), 'labels')
u, onehot_y = encode(listlabels)
print('Categories: ', u)
X_train = loaded_imgs
y_train = onehot_y


### validation images
listimgs_v, listlabels_v = [], []
for path in validation_paths:
	imgs, labels = read_images(path)
	listimgs_v += imgs
	listlabels_v += labels
loaded_imgs_v = [load_image(img, size=32, grayscale=True).reshape((32, 32, 1)) for img in listimgs_v]
print('[VALIDATION] Loaded', len(loaded_imgs_v), 'images and', len(listlabels_v), 'labels')
X_validation = loaded_imgs_v
y_validation = np.zeros((len(listlabels_v), len(u)))
y_validation[np.arange(len(listlabels_v)), [np.argwhere(u == label) for label in listlabels_v]] = 1


### test images
listimgs_t, listlabels_t = [], []
for path in test_paths:
	imgs, labels = read_images(path)
	listimgs_t += imgs
	listlabels_t += labels
loaded_imgs_t = [load_image(img, size=32, grayscale=True).reshape((32, 32, 1)) for img in listimgs_t]
print('[TEST] Loaded', len(loaded_imgs_t), 'images and', len(listlabels_t), 'labels')
X_test = loaded_imgs_t
y_test = np.zeros((len(listlabels_t), len(u)))
y_test[np.arange(len(listlabels_t)), [np.argwhere(u == label) for label in listlabels_t]] = 1


##### MODEL #####
tf.reset_default_graph() # reset the default graph before defining a new model

# Parameters
learning_rate = 0.001
training_epochs = 40
batch_size = 128
display_step = 1

# Model, loss function and accuracy

# tf Graph Input:  mnist data image of shape 28*28=784
x = tf.placeholder(tf.float32, (None, 32, 32, 1), name='InputData')
# 0-9 digits recognition,  10 classes
y = tf.placeholder(tf.int32, [None, len(u)], name='LabelData')

# Construct model and encapsulating all ops into scopes, making Tensorboard's Graph visualization more convenient
with tf.name_scope('Model'):
	# Model
	pred = LeNet5(x)
with tf.name_scope('Loss'):
	# Minimize error using cross entropy
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)
	cost = tf.reduce_mean(cross_entropy)
with tf.name_scope('Adam'):
	# Gradient Descent
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
with tf.name_scope('Accuracy'):
	# Accuracy
	acc = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	acc = tf.reduce_mean(tf.cast(acc, tf.float32))


correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

# Initializing the variables
init = tf.global_variables_initializer()
# Create a summary to monitor cost tensor
tf.summary.scalar("Loss_LeNet-5_Adam", cost)
# Create a summary to monitor accuracy tensor
tf.summary.scalar("Accuracy_LeNet-5_Adam", acc)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

with tf.Session() as sess:
	t0 = time.time()
	train(init, sess, training_epochs, batch_size, optimizer, cost, merged_summary_op)
	t1 = time.time()

	print("Training time:", t1-t0)

	# saving model
	#saver.save(sess, './LeNet_Adam')
	#print("Model saved")

	# Test model
	# Print the accuracy on testing data
	print("Accuracy:", acc.eval({x: X_test, y: y_test}))
