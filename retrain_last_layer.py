import tensorflow as tf
import json
import os
import numpy as np
import cv2
import sys
from utils import *

FC_WEIGHT_STDDEV = 0.01


##### UTILS #####
# used to load the pretrained model
def checkpoint_fn(layers):
    return 'ResNet-L%d.ckpt' % layers

# used to load the pretrained model
def meta_fn(layers):
    return 'ResNet-L%d.meta' % layers

# cross entropy loss, as it is a classification problem it is better
def loss(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss_ = tf.add_n([cross_entropy_mean] + regularization_losses)
    #loss_ = tf.add_n([cross_entropy_mean])
    tf.summary.scalar('loss', loss_)
    
    return loss_


### START EXECUTION

##### LOAD IMAGES ######
# read images
listimgs, listlabels = [], []
for path in sys.argv:
	imgs, labels = read_images(path)
	# RICORDATI DI TOGLIERE :42 !!!!!!
	listimgs += imgs[:42]
	listlabels += labels[:42]
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
# load from existing model and retrain last layer
layers = 50 # model to be loaded


sess = tf.Session()

# restore model
new_saver = tf.train.import_meta_graph(meta_fn(layers))
new_saver.restore(sess, checkpoint_fn(layers))
print("Completed restoring pretrained model")
# load last-but-one (layer) tensor after feeding images
graph = tf.get_default_graph()
features_tensor = graph.get_tensor_by_name("avg_pool:0")
images = graph.get_tensor_by_name("images:0")
feed_dict = {images: loaded_imgs}
features = sess.run(features_tensor, feed_dict=feed_dict) # Run the ResNet on loaded images
print('Completed running ResNet') # features now containes avg_pool output

print('Read', len(features), 'features')


# save file with avg_pool output
filename = "img_features.json"
with open(filename, "w") as f:
    for i in range(len(features)):
        feats_i = features[i].tolist()
        res = [listimgs[i], feats_i]
        f.write(json.dumps(res) + "\n") # Print features in file "img_features.json"
print('File save completed')





#### RETRAINING LAST LAYER ##### 

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

logits = tf.matmul(bottleneck_input, weights)
logits = tf.nn.bias_add(logits, biases)
final_tensor = tf.nn.softmax(logits, name="final_result")

loss_ = loss(logits, labelsVar)
ops = tf.train.AdamOptimizer(learning_rate=0.001)
train_op = ops.minimize(loss_)

# run training session
init=tf.global_variables_initializer()
sess.run(init)

features = [np.tanh(array) for array in features]
sess.run(train_op, feed_dict={bottleneck_input: features, labelsVar: indices})
print("Completed training")


# saving new model
tf.train.export_meta_graph(filename='new_model.meta')
saver = tf.train.Saver()
save_path = saver.save(sess, "new_model.ckpt")












#### TEST ####
def accuracy(true_labels, predicted_labels):
	correct = 0
	for i in range(len(true_labels)):
		if true_labels[i] == predicted_labels[i]:
			correct += 1
	return (correct/len(true_labels))*100

# read images
base_dir = '../Data/Videos/'

listimgs, listlabels = read_images(base_dir+"extracted_frames_plan_americain")
listimgs = listimgs[:30]
print("LEN LISTIMGS TEST:", len(listimgs))

# load images
loaded_imgs = []
for image in listimgs:
        img = load_image(image)
        batch = img.reshape((224, 224, 3))
        loaded_imgs.append(batch)


#img = load_image(listimgs)
#batch = img.reshape((1, 224, 224, 3))

features = sess.run(features_tensor, feed_dict = {images: loaded_imgs})
features = [np.tanh(array) for array in features]
print("FEATURES:", features)
prob = sess.run(final_tensor, feed_dict = {bottleneck_input: features})
print("First element probability:", prob[0], "->", u[np.argmax(prob[0])])
print("PROB:", prob)

print([u[np.argmax(probability)] for probability in prob])
print("Accuracy:", accuracy(["extracted_frames_gros_plan" for x in range(len(loaded_imgs))], [u[np.argmax(probability)] for probability in prob]), "%")
