import tensorflow as tf
import os
import numpy as np
import cv2
from utils import *
import argparse
from sklearn.utils import shuffle


##### UTILS #####
# cross entropy loss, as it is a classification problem it is better
def loss(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    
    #regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    #loss_ = tf.add_n([cross_entropy_mean] + regularization_losses)
    loss_ = cross_entropy_mean
    
    return loss_


### START EXECUTION
parser = argparse.ArgumentParser(description="Script for retraining last layer of the resnet architecture")
parser.add_argument('-lr', '--learning_rate', nargs='?', type=float, default=0.001, help='learning rate to be used')
parser.add_argument('-csv', '--csv_output', nargs='?', type=str, help='name of the output csv file for the loss and accuracy, file is not saved otherwise')
parser.add_argument('-e', '--epochs', nargs='?', type=int, default=20, help='number of epochs')
parser.add_argument('-transform', '--transformation', nargs='?', type=str, default='tanh', choices=['tanh', 'log1p'], help='transformation to be used on the features')
parser.add_argument('-a', '--arch', nargs='?', type=int, default=50, choices=[50, 152], help='chose pretrained model')
parser.add_argument('-t', '--train', nargs='+', help='paths to training directories', required=True)
parser.add_argument('-v', '--validation', nargs='+', help='paths to validation directory', required=True)
parser.add_argument('-test', nargs='+', help='paths to test directory')

args = parser.parse_args()
train_paths = args.train
validation_paths = args.validation
test_paths = args.test

# train params
epochs = args.epochs
transform = args.transformation
batch_size = 50
learning_rate = args.learning_rate
csv_out = args.csv_output
FC_WEIGHT_STDDEV = 0.01



##### LOAD IMAGES ######

### training images
# read images
listimgs, listlabels = [], []
for path in train_paths:
	imgs, labels = read_images(path)
	listimgs += imgs
	listlabels += labels

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




#### RETRAINING LAST LAYER #####

## placeholders and variables

# map string labels to unique integers
u,indices = np.unique(np.array(listlabels), return_inverse=True)
print('Categories: ', u)
num_categories = len(u)

# get avg pool dimensions
b_size, num_units_in = features_tensor.get_shape().as_list()

# define placeholder that will contain the inputs of the new layer
bottleneck_input = tf.placeholder(tf.float32, shape=[b_size,num_units_in], name='BottleneckInputPlaceholder') # define the input tensor
# define placeholder for the categories
labelsVar = tf.placeholder(tf.int32, shape=[b_size], name='labelsVar')


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

losses, train_accs, val_accs = [], [], []
for epoch in range(epochs):
	# shuffle dataset
	X_train, y_train = shuffle(loaded_imgs, indices)
	avg_cost = 0
	avg_acc = 0
	total_batch = int(len(X_train)/batch_size) if (len(X_train) % batch_size) == 0 else int(len(X_train)/batch_size)+1
	for offset in range(0, len(X_train), batch_size):
		batch_xs, batch_ys = X_train[offset:offset+batch_size], y_train[offset:offset+batch_size]

		# get features and optimize
		features = sess.run(features_tensor, feed_dict={images: batch_xs}) # Run the ResNet on loaded images
		# save file with avg_pool output
		#save_features(features)
		# apply features transformation
		if transform == 'tanh':
			features = [np.tanh(array) for array in features]
		elif transform == 'log1p':
			features = [np.log1p(array) for array in features]
		# run session
		_, loss = sess.run([train_op, loss_], feed_dict={bottleneck_input: features, labelsVar: batch_ys})
		avg_cost += loss / total_batch

		# get training accuracy
		prob = sess.run(final_tensor, feed_dict = {bottleneck_input: features})
		avg_acc += accuracy(batch_ys, [np.argmax(probability) for probability in prob]) / total_batch
		
	
	features = sess.run(features_tensor, feed_dict = {images: loaded_imgs_v})
	if transform == 'tanh':
		features = [np.tanh(array) for array in features]
	elif transform == 'log1p':
		features = [np.log1p(array) for array in features]
	prob = sess.run(final_tensor, feed_dict = {bottleneck_input: features})
	acc_v = accuracy(listlabels_v, [u[np.argmax(probability)] for probability in prob])
	print(epoch+1, ": Training Loss", avg_cost, "-Training Accuracy", avg_acc, "- Validation Accuracy", acc_v)
	losses.append(avg_cost)
	train_accs.append(avg_acc)
	val_accs.append(acc_v)
print("Completed training")


# saving new model
tf.train.export_meta_graph(filename='new_model.meta')
saver = tf.train.Saver()
save_path = saver.save(sess, "new_model.ckpt")
print("Model saved")

if csv_out is not None:
	export_csv(losses, train_accs, val_accs, filename=csv_out)
	print(csv_out, "saved")


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
	if transform == 'tanh':
		features = [np.tanh(array) for array in features]
	elif transform == 'log1p':
		features = [np.log1p(array) for array in features]
	prob = sess.run(final_tensor, feed_dict = {bottleneck_input: features})
	print("PROB:", prob)
	print([u[np.argmax(probability)] for probability in prob])
	print("Accuracy:", accuracy(listlabels_t, [u[np.argmax(probability)] for probability in prob]))
