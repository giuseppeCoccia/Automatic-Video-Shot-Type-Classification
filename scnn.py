import tensorflow as tf
import time
import numpy as np
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

def SCNN(image):
	# adding pooling layer to reduce input size
	#image = tf.nn.max_pool(image, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

	# Layer 1: Convolutional. Input = 32x32x3. Output = 30x30x32.
	conv1_W = weight_variable(shape=(3, 3, 3, 32))
	conv1_b = bias_variable(shape = [32])
	conv1_output = tf.nn.conv2d(image, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
	# Activation.
	conv1_output = tf.nn.relu(conv1_output)

	# Layer 2: Convolutional. Input = 30x30x32. Output = 28x28x32.
	conv2_W = weight_variable(shape=(3, 3, 32, 32))
	conv2_b = bias_variable(shape = [32])
	conv2_output   = tf.nn.conv2d(conv1_output, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
	# Activation.
	conv2_output = tf.nn.relu(conv2_output)

	# Flatten
	fc0_output = flatten(conv2_output)

	# Dense
	dns1_output = tf.layers.dense(fc0_output, 64, activation=tf.nn.relu)

	# Dropout
	dns1_output = tf.nn.dropout(dns1_output, keep_prob=0.75)

	# Dense
	dns2_output = tf.layers.dense(fc0_output, 3)
	return dns2_output

# encoding
def encode(labels, indices=None, u=None):
	if(indices is None):
		u, indices = np.unique(np.array(labels), return_inverse=True)
	onehot_y = np.zeros((len(labels), len(u)))
	onehot_y[np.arange(len(labels)), indices] = 1
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

def train(init, sess, n_epochs, batch_size, optimizer, cost):
	# optimizer and cost are the same kinds of objects as in Section 1
	# Train your model
	global X_train, y_train, X_validation, y_validation
	sess.run(init)
	
	losses, train_accs, val_accs = [], [], []
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
			_, c  = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})
			# Compute average loss
			avg_cost += c / total_batch

		train_acc = evaluate(X_train, y_train)
		val_acc = evaluate(X_validation, y_validation)

		losses.append(avg_cost)
		train_accs.append(train_acc)
		val_accs.append(val_acc)
		# Display logs per epoch step
		if (epoch+1) % display_step == 0:
			print("Epoch: ", '%02d' % (epoch+1), "  =====> Loss=", "{:.9f}".format(avg_cost),
				  "  =====> Training Accuracy=", train_acc,
				  "  =====> Validation Accuracy=", val_acc)
	return losses, train_accs, val_accs

print("Optimization Finished!")



### START EXECUTION
parser = argparse.ArgumentParser(description="Script for retraining last layer of the resnet architecture")
parser.add_argument('-t', '--train', nargs='+', help='paths to training directories', required=True)
parser.add_argument('-v', '--validation', nargs='+', help='paths to validation directory', required=True)
parser.add_argument('-test', nargs='+', help='path to test directory')
parser.add_argument('-lr', '--learning_rate', nargs='?', type=float, default=0.001, help='learning rate to be used')
parser.add_argument('-csv', '--csv_output', nargs='?', type=str, help='name of the output csv file for the loss and accuracy, file is not saved otherwise')
parser.add_argument('-e', '--epochs', nargs='?', type=int, default=20, help='number of epochs')

args = parser.parse_args()
train_paths = args.train
validation_paths = args.validation
test_paths = args.test

# Parameters
learning_rate = args.learning_rate 
training_epochs = args.epochs
batch_size = 128
display_step = 1
csv_out = args.csv_output




##### LOAD IMAGES ######

### training images
# read images
listimgs, listlabels = [], []
for path in train_paths:
	imgs, labels = read_images(path)
	listimgs += imgs
	listlabels += labels

# load images
loaded_imgs = [load_image(img, size=32).reshape((32, 32, 3)) for img in listimgs]
print('[TRAINING] Loaded', len(loaded_imgs), 'images', loaded_imgs[0].shape, 'and', len(listlabels), 'labels')
iu, y_train = np.unique(np.array(listlabels), return_inverse=True)
u, y_train = encode(listlabels)
X_train = loaded_imgs
#y_train = [[x] for x in y_train]
print('Categories: ', u)
cv2.imwrite('img.png',loaded_imgs[0])


### validation images
listimgs_v, listlabels_v = [], []
for path in validation_paths:
	imgs, labels = read_images(path)
	listimgs_v += imgs
	listlabels_v += labels
loaded_imgs_v = [load_image(img, size=32).reshape((32, 32, 3)) for img in listimgs_v]
print('[VALIDATION] Loaded', len(loaded_imgs_v), 'images and', len(listlabels_v), 'labels')
X_validation = loaded_imgs_v
#y_validation = [np.argwhere(u == label)[0] for label in listlabels_v]
_, y_validation = encode(listlabels_v, [np.argwhere(u == label) for label in listlabels_v], u)

if test_paths is not None:
	### test images
	listimgs_t, listlabels_t = [], []
	for path in test_paths:
		imgs, labels = read_images(path)
		listimgs_t += imgs
		listlabels_t += labels
	loaded_imgs_t = [load_image(img, size=32).reshape((32, 32, 3)) for img in listimgs_t]
	print('[TEST] Loaded', len(loaded_imgs_t), 'images and', len(listlabels_t), 'labels')
	X_test = loaded_imgs_t
	#y_test = [np.argwhere(u == label)[0] for label in listlabels_t]
	_, y_test = encode(listlabels_t, [np.argwhere(u == label) for label in listlabels_t], u)







##### MODEL #####
tf.reset_default_graph() # reset the default graph before defining a new model

# Model, loss function and accuracy

# tf Graph Input:  mnist data image of shape 28*28=784
x = tf.placeholder(tf.float32, (None, 32, 32, 3), name='InputData')
# 0-9 digits recognition,  10 classes
y = tf.placeholder(tf.float32, [None, 3], name='LabelData')

# Construct model and encapsulating all ops into scopes, making Tensorboard's Graph visualization more convenient
with tf.name_scope('Model'):
	# Model
	pred = SCNN(x)
with tf.name_scope('Loss'):
	# Minimize error using cross entropy
	cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=pred)
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

with tf.Session() as sess:
	t0 = time.time()
	losses, train_accs, val_accs = train(init, sess, training_epochs, batch_size, optimizer, cost)
	t1 = time.time()

	print("Training time:", t1-t0)

	# saving model
	#saver.save(sess, './LeNet_Adam')
	#print("Model saved")

	if test_paths is not None:
		# Test model
		print("Accuracy:", evaluate(X_test, y_test))

	if csv_out is not None:
                export_csv(losses, train_accs, val_accs, filename=csv_out)
