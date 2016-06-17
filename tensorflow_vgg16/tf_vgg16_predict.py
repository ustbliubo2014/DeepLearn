# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: tf_vgg16_predict.py
@time: 2016/6/17 11:51
@contact: ustb_liubo@qq.com
@annotation: tf_vgg16_predict
"""

import sys
reload(sys)
sys.setdefaultencoding("utf-8")
from scipy.misc import imread, imresize
import tensorflow as tf
from time import time
import numpy as np
import pdb

pic = imresize(imread('cat.jpg'), (224, 224, 3))
print pic.shape

synset = [l.strip() for l in open('synset.txt').readlines()]

def print_prob(prob):
    print "prob shape", prob.shape
    pred = np.argsort(prob)[::-1]

    top1 = synset[pred[0]]
    print "Top1: ", top1

    top5 = [synset[pred[i]] for i in range(5)]
    print "Top5: ", top5
    return top1


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

    batch = pic.reshape((1, 224, 224, 3))
    assert batch.shape == (1, 224, 224, 3)

    feed_dict = {images: batch}

    prob_tensor = graph.get_tensor_by_name("import/prob:0")
    start = time()
    prob = sess.run(prob_tensor, feed_dict=feed_dict)
    end = time()
    print 'predict time :', (end - start)

