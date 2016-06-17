import tensorflow as tf
import utils
from scipy.misc import imread, imresize
import msgpack_numpy
from time import time


cat = msgpack_numpy.load(open('dog.p','rb'))

with open("vgg16.tfmodel", mode='rb') as f:
  fileContent = f.read()

graph_def = tf.GraphDef()
graph_def.ParseFromString(fileContent)

images = tf.placeholder("float", [None, 224, 224, 3])

tf.import_graph_def(graph_def, input_map={"images": images})
print "graph loaded from disk"

graph = tf.get_default_graph()


with tf.Session() as sess:
  init = tf.initialize_all_variables()
  sess.run(init)
  print "variables initialized"

  batch = cat.reshape((1, 224, 224, 3))
  assert batch.shape == (1, 224, 224, 3)

  feed_dict = {images: batch}

  prob_tensor = graph.get_tensor_by_name("import/prob:0")
  start = time()
  prob = sess.run(prob_tensor, feed_dict=feed_dict)
  end = time()
  print 'predict time :', (end - start)
utils.print_prob(prob[0])


