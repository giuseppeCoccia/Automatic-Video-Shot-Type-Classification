import tensorflow as tf
import json
import os
import numpy as np
import cv2


FC_WEIGHT_STDDEV = 0.01



##### UTILS #####

# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path, size=224):
    #img = skimage.io.imread(path)
    img = cv2.imread(path)
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy + short_edge, xx:xx + short_edge]
    #resized_img = skimage.transform.resize(crop_img, (size, size))
    resized_img = cv2.resize(crop_img, (size, size))
    return resized_img

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
    tf.summary.scalar('loss', loss_)
    
    return loss_


def read_images(dir):
     listimgs = list()
     listlabels = list()
     for path, subdirs, files in os.walk(dir):
          for name in files:
               if ".jpg" in name:
                    listimgs.append(os.path.join(path, name))
                    listlabels.append(path.split('/')[-1])
     return listimgs, listlabels





### START EXECUTION

##### LOAD IMAGES ######

# read images
base_dir = "../Data/Images_Plans/"
#listimgs, listlabels = read_images(base_dir)
listimgs, listlabels = read_images(base_dir+"Gros plan")
a, b = read_images(base_dir+"Plan moyen")
listimgs += a
listlabels += b
a, b = read_images(base_dir+"Plan rapproche")
listimgs += a
listlabels += b
print('Completed loading images names')




# load images
loaded_imgs = []
for image in listimgs:
	img = load_image(image)
	batch = img.reshape((224, 224, 3))
	loaded_imgs.append(batch)
print('Completed loading images')
print('Loaded', len(listimgs), 'images and', len(listlabels), 'labels')





##### MODEL #####
layers = 152


sess = tf.Session()

new_saver = tf.train.import_meta_graph(meta_fn(layers))
new_saver.restore(sess, checkpoint_fn(layers))
	
graph = tf.get_default_graph()
features_tensor = graph.get_tensor_by_name("avg_pool:0")
images = graph.get_tensor_by_name("images:0")
feed_dict = {images: loaded_imgs}
features = sess.run(features_tensor, feed_dict=feed_dict) # Run the ResNet on loaded images
print('Completed running ResNet')

print('Read', len(features), 'features')


# save file
filename = "img_features.json"
with open(filename, "w") as f:
    for i in range(len(features)):
        feats_i = features[i].tolist()
        res = [listimgs[i], feats_i]
        f.write(json.dumps(res) + "\n") # Print features in file "img_features.json"
print('File save completed')



# retrain model

# map string labels to unique integers
u,indices = np.unique(np.array(listlabels), return_inverse=True)
print('Categories: ', u)
num_categories = len(u)
batch_size = len(features)

# get avg pool dimensions
batch_size, num_units_in = features_tensor.get_shape().as_list()

# define placeholder that will contain the inputs of the new layer
bottleneck_input = tf.placeholder(tf.float32, shape=[batch_size,num_units_in], name='BottleneckInputPlaceholder') # define the input tensor
# define placeholder for the categories
labelsVar = tf.placeholder(tf.int32, shape=(batch_size), name='labelsVar')


# weights and biases
weights_initializer = tf.truncated_normal_initializer(stddev=FC_WEIGHT_STDDEV)
weights = tf.get_variable('weights', shape=[num_units_in, num_categories], initializer=weights_initializer)
biases = tf.get_variable('biases', shape=[num_categories], initializer=tf.zeros_initializer)

logits = tf.matmul(bottleneck_input, weights)
logits = tf.nn.bias_add(logits, biases)

final_tensor = tf.nn.softmax(logits, name="final_result")

loss_ = loss(logits, labelsVar)
ops = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = ops.minimize(loss_)

# run training session
sess = tf.Session()
init=tf.global_variables_initializer()
sess.run(init)

sess.run(train_op, feed_dict={bottleneck_input: features, labelsVar: indices})
print("Completed training")



# saving new model
#tf.train.export_meta_graph(filename='tmp_model.meta')
#saver = tf.train.Saver()
#save_path = saver.save(sess, "tmp_model.ckpt")

# test the model
# read images
listimgs, listlabels = read_images(base_dir+"Tres gros plan")
img = load_image(listimgs[0])
batch = img.reshape((1, 224, 224, 3))

features = sess.run(features_tensor, feed_dict = {images: batch})
prob = sess.run(final_tensor, feed_dict = {bottleneck_input: features})
print(prob[0], "->", u[np.argwhere(prob[0])])
