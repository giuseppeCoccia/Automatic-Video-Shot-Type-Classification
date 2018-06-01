import tensorflow as tf
import os
import numpy as np
import cv2
from utils import *
import argparse
from sklearn.utils import shuffle

import tensorflow_hub as hub


##### UTILS #####
# cross entropy loss, as it is a classification problem it is better
def loss(logits, labels):
    #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    #cross_entropy_mean = tf.reduce_mean(cross_entropy, name="loss")
    cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(labels, logits)
    #regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    #loss_ = tf.add_n([cross_entropy_mean] + regularization_losses)
    loss_ = cross_entropy_mean
    
    return loss_


def create_module_graph(module_spec):
        height, width = hub.get_expected_image_size(module_spec)
        with tf.Graph().as_default() as graph:
            resized_input_tensor = tf.placeholder(tf.float32, [None, height, width, 3])
            m = hub.Module(module_spec)
            bottleneck_tensor = m(resized_input_tensor)
            #wants_quantization = any(node.op in FAKE_QUANT_OPS for node in graph.as_graph_def().node)
            return graph, bottleneck_tensor, resized_input_tensor


def resnet_model(layers, num_categories, dropout, FC_WEIGHT_STDDEV=0.01):
    # restore model
    #saver = tf.train.import_meta_graph(meta_fn(layers))
    #saver.restore(sess, checkpoint_fn(layers))
    #print("Completed restoring pretrained model")
    
    module_spec = hub.load_module_spec("https://tfhub.dev/google/imagenet/resnet_v1_50/feature_vector/1")
    graph, features_tensor, images = create_module_graph(module_spec)
    print("Completed restoring pretrained model")

    with graph.as_default():
        # load last-but-one (layer) tensor after feeding images
        #graph = tf.get_default_graph()
        #features_tensor = graph.get_tensor_by_name("avg_pool:0")
        if(dropout == True):
            features_tensor = tf.nn.dropout(features_tensor, keep_prob=0.75)
        #images = graph.get_tensor_by_name("images:0")
        # get avg pool dimensions
        b_size, num_units_in = features_tensor.get_shape().as_list()

        # define placeholder that will contain the inputs of the new layer
        bottleneck_input = tf.placeholder(tf.float32, shape=[b_size,num_units_in], name='BottleneckInput') # define the input tensor
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
        train_op = ops.minimize(loss_, name="train_op")

        return graph, images, features_tensor, bottleneck_input, labelsVar, final_tensor, loss_, train_op


def train(sess, listimgs, loaded_imgs, listlabels_v, listimgs_v, loaded_imgs_v, indices, u, images, features_tensor, bottleneck_input, labelsVar, final_tensor, loss_, train_op, epochs, batch_size, transform, save_models=False, load_train_features=False, load_validation_features=False):
    losses, train_accs, val_accs = [], [], []

    if not load_train_features:
        # get features and optimize
        features = sess.run(features_tensor, feed_dict={images: loaded_imgs}) # Run the ResNet on loaded images
	# save file with avg_pool output
        save_features(features, listimgs, filename="resnet_train_features.json")
    else:
        features = load_features(filename="resnet_train_features.json")
    # apply features transformation
    #if transform == 'tanh':
    #    features = [np.tanh(array) for array in features]
    #elif transform == 'log1p':
    #    features = [np.log1p(array) for array in features]
    if not load_validation_features:
        features_v = sess.run(features_tensor, feed_dict = {images: loaded_imgs_v})
        save_features(features_v, listimgs_v, filename="resnet_validation_features.json")
    else:
        features_v = load_features(filename="resnet_validation_features.json")
    #if transform == 'tanh':
    #    features = [np.tanh(array) for array in features]
    #elif transform == 'log1p':
    #    features = [np.log1p(array) for array in features]
         
    for epoch in range(epochs):
        # shuffle dataset
        X_train_indices, y_train = shuffle(np.arange(len(loaded_imgs)), indices)
        avg_cost = 0
        avg_acc = 0
        total_batch = int(len(loaded_imgs)/batch_size) if (len(loaded_imgs) % batch_size) == 0 else int(len(loaded_imgs)/batch_size)+1
        for offset in range(0, len(loaded_imgs), batch_size):
            batch_xs_indices, batch_ys = X_train_indices[offset:offset+batch_size], y_train[offset:offset+batch_size]
            
            # run session
            _, loss = sess.run([train_op, loss_], feed_dict={bottleneck_input: features[batch_xs_indices], labelsVar: batch_ys})
            avg_cost += loss / total_batch
            
            # get training accuracy
            prob = sess.run(final_tensor, feed_dict = {bottleneck_input: features[batch_xs_indices]})
            avg_acc += accuracy(batch_ys, [np.argmax(probability) for probability in prob]) / total_batch

        prob = sess.run(final_tensor, feed_dict = {bottleneck_input: features_v})
        acc_v = accuracy(listlabels_v, [u[np.argmax(probability)] for probability in prob])
        print(epoch+1, ": Training Loss", avg_cost, "-Training Accuracy", avg_acc, "- Validation Accuracy", acc_v)
        losses.append(avg_cost)
        train_accs.append(avg_acc)
        val_accs.append(acc_v)

        # save model
        if save_models:
        	saver = tf.train.Saver()
        	saver.save(sess, "resnet_model"+str(epoch+1)+".ckpt")

    return losses, train_accs, val_accs


### START EXECUTION
parser = argparse.ArgumentParser(description="Script for retraining last layer of the resnet architecture")
parser.add_argument('-lr', '--learning_rate', nargs='?', type=float, default=0.001, help='learning rate to be used')
parser.add_argument('-csv', '--csv_output', nargs='?', type=str, help='name of the output csv file for the loss and accuracy, file is not saved otherwise')
parser.add_argument('-m', '--model_epoch', nargs='?', type=int, help='epoch number of the model to be restored')
parser.add_argument('-bs', '--batch_size', nargs='?', type=int, default=32, help='batch size for training batches')
parser.add_argument('-e', '--epochs', nargs='?', type=int, default=20, help='number of epochs')
parser.add_argument('-transform', '--transformation', nargs='?', type=str, default='tanh', choices=['tanh', 'log1p'], help='transformation to be used on the features')
parser.add_argument('-a', '--arch', nargs='?', type=int, default=50, choices=[50, 152], help='chose pretrained model')
parser.add_argument('-t', '--train', nargs='+', help='paths to training directories', required=True)
parser.add_argument('-v', '--validation', nargs='+', help='paths to validation directory', required=True)
parser.add_argument('-test', nargs='+', help='paths to test directory')
parser.add_argument('-d', '--dropout', nargs='?', type=bool, default=False, help='Use or not dropout of features going to last fully connected layer')
parser.add_argument('-s', '--save', type=bool, default=False, help='if True, save models at each epoch')
parser.add_argument('-if', '--import_features', type=bool, default=False, help='if True, read features files')

args = parser.parse_args()
train_paths = args.train
validation_paths = args.validation
test_paths = args.test
model_to_restore = args.model_epoch

# train params
epochs = args.epochs
transform = args.transformation
batch_size = args.batch_size
learning_rate = args.learning_rate
csv_out = args.csv_output
dropout = args.dropout
save_models = args.save
import_features = args.import_features


##### LOAD IMAGES ######

layers = args.arch # model to be loaded
graph, images, features_tensor, bottleneck_input, labelsVar, final_tensor, loss_, train_op = resnet_model(layers, 3, dropout)

def load_image_tensor(sess, img):
	input_height, input_width = 224, 224
	input_depth = 3 
	jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
	decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
	# Convert from full range of uint8 to range [0,1] of float32.
	decoded_image_as_float = tf.image.convert_image_dtype(decoded_image, tf.float32)
	decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
	resize_shape = tf.stack([input_height, input_width])
	resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
	resized_image = tf.image.resize_bilinear(decoded_image_4d, resize_shape_as_int)
	image_data = tf.gfile.FastGFile(img, 'rb').read()
	resized_input_values = sess.run(resized_image,
                                  {jpeg_data: image_data})
	return resized_input_values

### training images
# read images

listimgs, listlabels = [], []
for path in train_paths:
	imgs, labels = read_images(path)
	listimgs += imgs
	listlabels += labels

# load images
loaded_imgs = [load_image(img).reshape((224, 224, 3)) for img in listimgs]
#loaded_imgs = [load_image_tensor(sess, img)[0] for img in listimgs]
print('[TRAINING] Loaded', len(loaded_imgs), 'images and', len(listlabels), 'labels')

### validation images
listimgs_v, listlabels_v = [], []
for path in validation_paths:
        imgs, labels = read_images(path)
        listimgs_v += imgs
        listlabels_v += labels
loaded_imgs_v = [load_image(img).reshape((224, 224, 3)) for img in listimgs_v]
#loaded_imgs_v = [load_image_tensor(sess, img)[0] for img in listimgs_v]
print('[VALIDATION] Loaded', len(loaded_imgs_v), 'images and', len(listlabels_v), 'labels')


# map string labels to unique integers
u,indices = np.unique(np.array(listlabels), return_inverse=True)
print('Categories: ', u)
num_categories = len(u)






##### MODEL #####
with tf.Session(graph=graph) as sess:
    init=tf.global_variables_initializer()
    sess.run(init)

    if model_to_restore is None:
        # load from existing model and retrain last layer

        # export meta graph
        tf.train.export_meta_graph(filename='resnet_model.meta')
        losses, train_accs, val_accs = train(sess,
                                             listimgs, loaded_imgs,
                                             listlabels_v,
                                             listimgs_v, loaded_imgs_v,
                                             indices,
                                             u,
                                             images,
                                             features_tensor,
                                             bottleneck_input,
                                             labelsVar,
                                             final_tensor,
                                             loss_,
                                             train_op,
                                             epochs,
                                             batch_size,
                                             transform,
                         save_models=save_models,
                         load_train_features=import_features,
                         load_validation_features=import_features)
        print("Completed training")
    else:
        # restore model
        new_saver = tf.train.import_meta_graph("resnet_model.meta")
        new_saver.restore(sess, "resnet_model"+str(model_to_restore)+".ckpt")
        print("Restored Model")

        graph = tf.get_default_graph()
        features_tensor = graph.get_tensor_by_name("avg_pool:0")	
        images = graph.get_tensor_by_name("images:0")
        bottleneck_input = graph.get_tensor_by_name("BottleneckInput:0")
        labelsVar = graph.get_tensor_by_name("labelsVar:0")
        final_tensor = graph.get_tensor_by_name("final_result:0")
        loss_ = graph.get_tensor_by_name("loss:0")
        train_op = graph.get_operation_by_name("train_op")

        losses, train_accs, val_accs = train(sess,
                                             listimgs, loaded_imgs,
                                             listlabels_v,
                                             listimgs_v, loaded_imgs_v,
                                             indices,
                                             u,
                                             images,
                                             features_tensor,
                                             bottleneck_input,
                                             labelsVar,
                                             final_tensor,
                                             loss_,
                                             train_op,
                                             epochs,
                                             batch_size,
                                             transform,
                         save_models=save_models,
                         load_train_features=import_features,
                         load_validation_features=import_features)


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
