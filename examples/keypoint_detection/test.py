import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
sys.path.append('../../python/')
import caffe

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
im = cv2.imread(sys.argv[1])
in_ = np.array(im, dtype=np.float32)
in_ = in_[:,:,::-1]
in_ -= np.array((102.9801, 115.9465, 122.7717))
in_ = in_.transpose((2,0,1))

# load net
net = caffe.Net('deploy.prototxt', 'kp_detection_lr_6_iter_10000.caffemodel', caffe.TEST)#kp_detection_lr_6_iter_10000.caffemodel', caffe.TEST)
# shape for input (data blob is N x C x H x W), set data
net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_
# run net and take argmax for prediction
net.forward()
f = plt.figure()
for i in xrange(10):
    out = net.blobs['score_10'].data[0][i]
    ax = f.add_subplot(int('52'+str(i)))
    ax.imshow(out)
    ax.imshow(im, alpha=0.5)
plt.show()
