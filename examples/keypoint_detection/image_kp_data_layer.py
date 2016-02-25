import os
import sys
sys.path.append("/home/debidatd/parsenet/python")
import caffe
import numpy as np
import yaml
import cv2

im_datapath = '/home/debidatd/parsenet/examples/keypoint_detection/cereal_box/'
kp_datapath = '/media/dey/debidatd/rgbd_uw/rgbd-keypoints/cereal_box/cereal_box_1/'

class ImageKPDataLayer(caffe.Layer):
    """Data layer used for preparing input images and keypoint heatmaps."""
    def transform_image(self, im, transform_params=None):
        s = 1
        if transform_params == None:
            if np.random.randint(100)%2 == 0:
                s = np.random.uniform(0.5, 1.2)
        else:
            s = transform_params
        transform_params = s
        im = cv2.resize(im, (0,0), fx=s, fy=s)
        return im, transform_params 
        
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
    
    def prep_kps_for_blob(self, kps_list, transform_params):
        for i,kp_file in enumerate(kps_list):
            kp_heatmap = cv2.imread(kp_file, cv2.IMREAD_UNCHANGED)
            if self._phase == 'TRAIN':
                kp_heatmap, transform_params = self.transform_image(kp_heatmap, transform_params)
            if np.max(kp_heatmap) != 0:
                kp_heatmap = kp_heatmap.astype(np.float32, copy=False)/np.max(kp_heatmap)
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
            transform_params = None
            if self._phase == 'TRAIN':
                im, transform_params = self.transform_image(im)
            #if roidb[i]['flipped']:
            #    im = im[:, ::-1, :]
            #target_size = cfg.TRAIN.SCALES[scale_inds[i]]
            im = self.prep_im_for_blob(im)
            kps = self.prep_kps_for_blob(minibatch_db[i]['kps'], transform_params)
                
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
     	
    def get_imdb(self, im_list_file, num_kps):
        with open(im_list_file) as f:
            im_files = [x.strip() for x in f.readlines()]
        imdb = []
        for im in im_files:
            imdb_entry = {}
            imdb_entry['im'] = os.path.join(im_datapath, im)
            kp_heatmap_files = []
            
            for kp in xrange(1, num_kps+1):
                kp_heatmap_files.append(os.path.join(kp_datapath,im.split('/')[-1][:-4],str(kp)+'.png'))
            imdb_entry['kps'] = kp_heatmap_files
            imdb.append(imdb_entry)
        return imdb    
             
    def setup(self, bottom, top):
        """Setup the ImageKPDataLayer."""

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)
        #self._num_keypoints = 10#layer_params['num_keypoints']
        #self._im_list_file = 'a.txt'#layer_params['im_list']
        #print "as"
        self._num_keypoints = layer_params['num_keypoints']
        self._im_list_file = layer_params['im_list']
        self._phase = layer_params['phase'] 
        self._name_to_top_map = {
            'data': 0,
            'kps': 1}
        
        self._imdb = self.get_imdb(self._im_list_file, self._num_keypoints)
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

