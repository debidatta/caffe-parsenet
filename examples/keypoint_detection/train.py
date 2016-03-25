from __future__ import division
import sys

caffe_root = '/home/debidatd/parsenet/'
sys.path.insert(0, caffe_root + 'python')

import caffe
import numpy as np

# init
caffe.set_mode_gpu()
caffe.set_device(0)

# caffe.set_mode_cpu()

solver = caffe.SGDSolver(sys.argv[1])
solver.net.copy_from(sys.argv[2])

niter = 50000 
test_iter = 100
test_interval = 500
train_loss = np.zeros(niter)
test_loss = np.zeros(niter)
f = open('train_log.txt', 'w')
g = open('test_log.txt', 'w')

for i in range(niter):
    solver.step(1)
    train_loss[i] = solver.net.blobs['loss'].data
    f.write('{} {}\n'.format(i, train_loss[i]))
    if (i+1)%test_interval  == 0:
        for j in xrange(test_iter):
            solver.test_nets[0].forward()
            test_loss[i] += solver.test_nets[0].blobs['loss'].data
        test_loss[i] = test_loss[i]/test_iter
        g.write('{} {}\n'.format(i, test_loss[i]))
        g.flush()
    f.flush()  
    
f.close()
g.close()
