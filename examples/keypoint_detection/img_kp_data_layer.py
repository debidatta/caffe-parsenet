import os
import sys
sys.path.append("/home/debidatd/parsenet/python")
import caffe
import numpy as np
import yaml
import cv2
import matplotlib.pyplot as plt
import matplotlib.mlab as ml
import matplotlib.cm as cm
import os
from PIL import Image

def get_heatmap_from_kp(cx, cy, w, h):
    x = np.linspace(1, w, w)
    y = np.linspace(1, h, h)
    X, Y = np.meshgrid(x, y)
    # 1.5 pixels from human keypoint annotation
    Z = ml.bivariate_normal(X, Y, 1.5, 1.5, cx, cy)
    if np.max(Z) != 0:
        Z = Z/np.max(Z)
    #im = Image.fromarray(np.uint8(Z))
    return Z

class ImageKPDataLayer(caffe.Layer):
    """Data layer used for preparing input images and keypoint heatmaps."""
    #def transform_image(self, im, transform_params=None):
    #    s = 1
    #    if transform_params == None:
    #        if np.random.randint(100)%2 == 0:
    #            s = np.random.uniform(0.5, 1.2)
    #    else:
    #        s = transform_params
    #    transform_params = s
    #    im = cv2.resize(im, (0,0), fx=s, fy=s)
    #    return im, transform_params 
        
    def augment_data(self, im, kps):
        crop_mode = np.random.random_integers(0,5)
        if crop_mode == 0:
            im = im[0:480, 0:640, :]
            kps = kps[0:480, 0:640, :]
        elif crop_mode == 1:
            im = im[60:, 320:, :]
            kps = kps[60:, 320:, :]
        elif crop_mode == 2:
            im = im[0:480, 320:, :]
            kps = kps[0:480, 320:, :]
        elif crop_mode == 3:
            im = im[60:, 0:640, :]
            kps = kps[60:, 0:640, :]
        elif crop_mode == 4:
            im = im[30:510, 160:800, :]
            kps = kps[30:510, 160:800, :]
        return im, kps

        if np.random.random_integers(1,5) <= 2: #scale variation
            s = np.random.uniform(0.8, 1.2)
            im = cv2.resize(im, (0,0), fx=s, fy=s)
            kps = cv2.resize(kps, (0,0), fx=s, fy=s)
        if np.random.random_integers(1,5) <= 2:  #random crops
            h, w, _ = img.shape
            if np.random.randint(100)%2 == 0:
                c_x_start = 0
                c_x_end = np.random.randint(w/2, w-1)
            else:
                c_x_start = np.random.randint(0, h/2) 
                c_x_end = h-1
            if np.random.randint(100)%2 == 0:
                c_y_start = 0
                c_y_end = np.random.randint(h/2, h-1)
            else:
                c_y_start = np.random.randint(0, h/2)   
                c_y_end = h-1
            im = im[c_y_start:c_y_end, c_x_start:c_x_end, :]
            kps = kps[c_y_start:c_y_end, c_x_start:c_x_end, :]
        if np.random.random_integers(1,5) <= 2:  #random flip
            im = im[:,::-1, :]
            kps = kps[:,::-1, :] 
        return im, kps

    def kps_list_to_blob(self, kps):
        """Convert a list of keypoint heatmaps into a network input.
        Assumes images are already prepared (means subtracted, BGR order, ...).
        """
        max_shape = np.array([kp.shape for kp in kps]).max(axis=0)
        num_images = len(kps)
        blob = np.zeros((num_images, max_shape[0], max_shape[1], max_shape[2]),
                        dtype=np.float32)
        for i in xrange(num_images):
            blob[i, 0:kp.shape[0], 0:kp.shape[1], :] = kps[i]
        # Move channels (axis 3) to axis 1
        # Axis order will become: (batch elem, channel, height, width)
        channel_swap = (0, 3, 1, 2)
        blob = blob.transpose(channel_swap)
        return blob

    def im_list_to_blob(self, ims):
        """Convert a list of images into a network input.
        Assumes images are already prepared (means subtracted, BGR order, ...).
        """
        max_shape = np.array([im.shape for im in ims]).max(axis=0)
        num_images = len(ims)
        blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                        dtype=np.float32)
        for i in xrange(num_images):
            im = ims[i]
            blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
        # Move channels (axis 3) to axis 1
        # Axis order will become: (batch elem, channel, height, width)
        channel_swap = (0, 3, 1, 2)
        blob = blob.transpose(channel_swap)
        return blob

    def prep_im_for_blob(self, im, pixel_means=np.array([[[102.9801, 115.9465, 122.7717]]]), target_size=600):
        """Mean subtract and scale an image for use in a blob."""
        im = im.astype(np.float32, copy=False)
        im -= pixel_means
        #im_shape = im.shape
        #im_size_min = np.min(im_shape[0:2])
        #im_size_max = np.max(im_shape[0:2])
        #im_scale = float(target_size) / float(im_size_min)
        #im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
        #            interpolation=cv2.INTER_LINEAR)

        return im
    
    def prep_kps_for_blob(self, kps_file, kps_list):#, transform_params):
        with open(kps_file) as f:
            kps_all = [x.strip().split() for x in f.readlines()]
        
        for i,kp_id in enumerate(kps_list):
            if kps_all[kp_id][1] == -1:
                kp_heatmap =  get_heatmap_from_kp(-1000, -1000, 960, 540)
            else:
                kp_heatmap =  get_heatmap_from_kp(float(kps_all[kp_id][1]), float(kps_all[kp_id][2]), 960, 540)
            #if self._phase == 'TRAIN':
            #    kp_heatmap, transform_params = self.transform_image(kp_heatmap, transform_params)
            #if np.max(kp_heatmap) != 0:
            #    kp_heatmap = kp_heatmap.astype(np.float32, copy=False)/np.max(kp_heatmap)
            if i==0:
                shape = kp_heatmap.shape
                kps = np.zeros((shape[0], shape[1], len(kps_list)), dtype=np.float32)
            kps[:,:,i] = kp_heatmap
        return kps    
            
    def get_minibatch(self, minibatch_db):
        num_images = len(minibatch_db)
        processed_ims = []
        processed_kps = []
        for i in xrange(num_images):
            im = cv2.imread(minibatch_db[i]['im'])
            kp = minibatch_db[i]['kp']
            #transform_params = None
            #if self._phase == 'TRAIN':
            #    im, transform_params = self.transform_image(im)
            #if roidb[i]['flipped']:
            #    im = im[:, ::-1, :]
            #target_size = cfg.TRAIN.SCALES[scale_inds[i]]
            im = self.prep_im_for_blob(im)
            kps = self.prep_kps_for_blob(kp, self._kps)#, transform_params)
            if self._phase == 'TRAIN':
                im, kps = self.augment_data(im, kps)  
            #im_scales.append(im_scale)
            processed_ims.append(im)
            processed_kps.append(kps)
        # Create a blob to hold the input images
        imblob = self.im_list_to_blob(processed_ims)
        kpblob = self.kps_list_to_blob(processed_kps) 
       
        blobs = {'data': imblob,
                 'kps': kpblob}

        return blobs
 
    def _shuffle_imdb_inds(self):
        """Randomly permute the training roidb."""
        self._perm = np.random.permutation(np.arange(len(self._imdb)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        if self._cur + 1 >= len(self._imdb):
            self._shuffle_imdb_inds()

        db_inds = self._perm[self._cur:self._cur + 1] #cfg.TRAIN.IMS_PER_BATCH]
        self._cur += 1 #cfg.TRAIN.IMS_PER_BATCH
        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.
        """
        db_inds = self._get_next_minibatch_inds()
        minibatch_db = [self._imdb[i] for i in db_inds]
        return self.get_minibatch(minibatch_db)
     	
    def get_imdb(self, im_list_file, kp_list_file):
        with open(im_list_file) as f:
            im_kp_files = [x.strip().split() for x in f.readlines()]
        with open(kp_list_file) as f:
            kps = [int(x.strip()) for x in f.readlines()]
        self._kps = kps
        imdb = []
        for im_kp in im_kp_files:
            imdb_entry = {}
            imdb_entry['im'] = os.path.join(im_kp[0])
            imdb_entry['kp'] = os.path.join(im_kp[1])
            imdb.append(imdb_entry)
        return imdb, len(kps)    
             
    def setup(self, bottom, top):
        """Setup the ImageKPDataLayer."""
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)
        #self._num_keypoints = 10#layer_params['num_keypoints']
        #self._im_list_file = 'a.txt'#layer_params['im_list']
        #print "as"
        self._keypoint_list = layer_params['kp_list']
        self._im_list_file = layer_params['im_list']
        self._phase = layer_params['phase'] 
        self._name_to_top_map = {
            'data': 0,
            'kps': 1}
        
        self._imdb, self._num_keypoints = self.get_imdb(self._im_list_file, self._keypoint_list)
        self._shuffle_imdb_inds()
        # data blob: holds a batch of N images, each with 3 channels
        # The height and width (100 x 100) are dummy values
        top[0].reshape(1, 3, 640, 480)

        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        top[1].reshape(1, self._num_keypoints, 640, 480)

        # labels blob: R categorical labels in [0, ..., K] for K foreground
        # classes plus background
        # top[2].reshape(1)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()

        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

