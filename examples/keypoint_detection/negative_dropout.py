import caffe
import numpy as np
 
class NegativeDropoutLayer(caffe.Layer):
    
    def set_mask(self, bottom):
	bottom_data = bottom[0].data
	label = bottom[1].data

	count = bottom[0].count
	num = bottom_data.shape[0];
	out_channel = bottom_data.shape[1]
	out_height = bottom_data.shape[3]
	out_width = bottom_data.shape[2]
	mapsize = out_height * out_width

	drop_neg_ratio = 1.0
	hard_ratio = 0.5
	rand_ratio = 0.5
	self.mask_data = np.zeros_like(bottom_data, dtype=np.float32)
        self.mask_data[label>0] = 1
        for i in xrange(num):
            for j in xrange(out_channel):
                pos_num = np.count_nonzero(label[i][j])
                neg_num = int(drop_neg_ratio * pos_num) + 10
                neg_indices = np.where(label[i][j] == 0)
                chosen_neg_indices = np.random.permutation(len(neg_indices[0]))
                for n in xrange(neg_num):
                    k = neg_indices[0][chosen_neg_indices[n]]
                    l = neg_indices[1][chosen_neg_indices[n]]
                    self.mask_data[i][j][k][l] = 1
    
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs: one score and one label.")
 
    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        bottom_shape = bottom[0].data.shape
        top[0].reshape(bottom_shape[0], bottom_shape[1], bottom_shape[2], bottom_shape[3])
 
    def forward(self, bottom, top):
        self.set_mask(bottom)
        top[0].data[self.mask_data==1] = bottom[0].data[self.mask_data==1]  
        top[0].data[self.mask_data==0] = bottom[1].data[self.mask_data==0]

    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...] = np.zeros_like(bottom[0].data, dtype=np.float32)
        bottom[0].diff[self.mask_data==1] = top[0].diff[self.mask_data==1]
