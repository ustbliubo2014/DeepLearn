from scipy.misc import imread, imresize
import numpy as np
import msgpack_numpy

synset = [l.strip() for l in open('synset.txt').readlines()]


# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path):
    arr = imread(path)
    print 'arr.shape :', arr.shape
    return imresize(arr, (224,224,3))


# returns the top1 string
def print_prob(prob):
    print "prob shape", prob.shape
    pred = np.argsort(prob)[::-1]

    top1 = synset[pred[0]]
    print "Top1: ", top1

    top5 = [synset[pred[i]] for i in range(5)]
    print "Top5: ", top5
    return top1


if __name__ == '__main__':
    path = '2a4e941296f7452c5614dfce5718d80e.jpg'
    # resized_img = load_image(path)
    # msgpack_numpy.dump(resized_img, open('dog.p','wb'))
    arr = imread(path)
    print 'arr.shape :', arr.shape
    print imresize(arr, (224, 224, 3)) / 255.0
