import tensorflow as tf
import cv2

def print_prob(prob):
    #print prob
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    top1 = synset[pred[0]]
    print("Top1: ", top1)
    # Get top5 label
    top5 = [synset[pred[i]] for i in range(5)]
    print("Top5: ", top5)
    return top1

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

layers = 50

img = load_image("../Data/Images_Plans/Titres/AFE86001229_L'Europe des travailleurs_0''.jpg")

sess = tf.Session()

new_saver = tf.train.import_meta_graph('tmp_model.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('./'))

graph = tf.get_default_graph()
prob_tensor = graph.get_tensor_by_name("prob:0")
images = graph.get_tensor_by_name("images:0")

#init = tf.initialize_all_variables()
#sess.run(init)
print("graph restored")

batch = img.reshape((1, 224, 224, 3))

feed_dict = {images: batch}

prob = sess.run(prob_tensor, feed_dict=feed_dict)

print(prob[0])
#print_prob(prob[0])
