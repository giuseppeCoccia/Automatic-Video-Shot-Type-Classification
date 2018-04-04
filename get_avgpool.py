from convert import print_prob, load_image, checkpoint_fn, meta_fn
import tensorflow as tf

layers = 152

img = load_image("data/cat.jpg")

sess = tf.Session()

new_saver = tf.train.import_meta_graph(meta_fn(layers))
new_saver.restore(sess, checkpoint_fn(layers))

graph = tf.get_default_graph()

for op in graph.get_operations():
    print op.name

print "graph restored"

input_tensor = graph.get_tensor_by_name("images:0")
features_tensor = graph.get_tensor_by_name("avg_pool:0")

batch = img.reshape((1, 224, 224, 3))

feed_dict = {input_tensor: batch}

prob = sess.run(input_tensor, feed_dict=feed_dict)
print(features_tensor[0])
