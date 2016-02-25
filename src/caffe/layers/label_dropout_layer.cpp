// Copyright 2014 BVLC and contributors.

#include <vector>
#include <algorithm>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

namespace caffe {

int myrandom (int i) { return caffe_rng_rand()%i;}

template <typename Dtype>
void LabelDropoutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  //CHECK_EQ(bottom.size(), 1) << "IP Layer takes a single blob as input.";
  //CHECK_EQ(top->size(), 1) << "IP Layer takes a single blob as output.";

  //int num = bottom[0]->num();
  //int out_channel = bottom[0]->channels();
  //int out_height = bottom[0]->height();
  //int out_width = bottom[0]->width();

  //(*top)[0]->Reshape(num, out_channel, out_height, out_width);
  //mask_.Reshape(bottom[0]->num(), out_channel, out_height, out_width);

}

template <typename Dtype>
void LabelDropoutLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
     //CHECK_EQ(topsize(), 1) << "IP Layer takes a single blob as output.";
     int num = bottom[0]->num();
     int out_channel = bottom[0]->channels();
     int out_height = bottom[0]->height();
     int out_width = bottom[0]->width();
                
     top[0]->Reshape(num, out_channel, out_height, out_width);
     mask_.Reshape(bottom[0]->num(), out_channel, out_height, out_width);
}
 


template <typename Dtype>
void LabelDropoutLayer<Dtype>::set_mask(const vector<Blob<Dtype>*>& bottom)
{
	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* label = bottom[1]->cpu_data();

	int dim = bottom[0]->count() / bottom[0]->num();
	int count = bottom[0]->count();
	int num = bottom[0]->num();
	int out_channel = bottom[0]->channels();
	int out_height = bottom[0]->height();
	int out_width = bottom[0]->width();
	int mapsize = out_height * out_width;

	LabelDropParameter label_drop_param = this->layer_param_.label_drop_param();
	float drop_neg_ratio = label_drop_param.drop_neg_ratio();
	float hard_ratio = label_drop_param.hard_ratio();
	float rand_ratio = label_drop_param.rand_ratio();
	Dtype* mask_data = mask_.mutable_cpu_data();

	vector<pair<float, int> > negpairs;
	vector<int> sid1;
	vector<int> sid2;

	/*vector<int> idxs;
	for(int i = 0; i < 100; i ++) idxs.push_back(i);
	std::random_shuffle(idxs.begin(), idxs.end(), myrandom);*/

	for(int i = 0; i < count; i ++) mask_data[i] = 0;

	for(int i = 0; i < num; i ++)
	{

		for(int j = 0; j < out_channel; j ++)
		{
			negpairs.clear();
			sid1.clear();
			sid2.clear();
			int pos_num = 0;
			int neg_num = 0;
			for(int k = 0; k < mapsize; k ++)
			{
				int nowid = i * dim + j * mapsize + k;
                                //std::cout << label[nowid];
				if(label[nowid] > 0.01)
				{
					mask_data[nowid] = 1;
					pos_num ++;
				}
				else//if(label[nowid] == 0)
				{
					float ts = fabs(bottom_data[nowid]);
					negpairs.push_back(make_pair(ts, nowid));
					neg_num ++;
				}
			}
                        std::cout << pos_num;
			int use_neg_num = pos_num * drop_neg_ratio;
			if(use_neg_num >= neg_num)
			{
				for(int k = 0; k < negpairs.size(); k ++)
				{
					mask_data[negpairs[k].second] = 1;
				}
				continue;
			}

			sort(negpairs.begin(), negpairs.end());
			for(int k = 0; k < use_neg_num; k ++)
			{
				sid1.push_back(negpairs[neg_num - k - 1].second);
			}
			for(int k = 0; k < neg_num - use_neg_num; k ++)
			{
				sid2.push_back(negpairs[k].second);
			}
			std::random_shuffle(sid1.begin(), sid1.end(), myrandom);
			int hardNum = use_neg_num * hard_ratio;
			int randNum = use_neg_num * rand_ratio;
			for(int k = 0; k < hardNum; k ++)
			{
				mask_data[sid1[k]] = 1;
			}
			for(int k = 0; k < randNum; k ++)
			{
				sid2.push_back(sid1[hardNum + k]);
			}
			std::random_shuffle(sid2.begin(), sid2.end(), myrandom);
			for(int k = 0; k < randNum; k ++)
			{
				mask_data[sid2[k]] = 1;
			}

		}

	}

}


template <typename Dtype>
void LabelDropoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {

    Dtype* top_data = top[0]->mutable_cpu_data();
    const Dtype * bottom_data = bottom[0]->cpu_data();
    const Dtype * label = bottom[1]->cpu_data();

    int dim = bottom[0]->count() / bottom[0]->num();
    int num = bottom[0]->num();
    int out_channel = bottom[0]->channels();
    int out_height = bottom[0]->height();
    int out_width = bottom[0]->width();

    set_mask(bottom);
    Dtype* mask_data = mask_.mutable_cpu_data();
    for(int i = 0; i < top[0]->count(); i ++)
	{
    	top_data[i] = label[i];
    	if (mask_data[i] > 0)
    	{
    		top_data[i] = bottom_data[i];
    	}
	}

    // debug
    if(0)
    {

    char ss1[1010];
    Mat imgout(Size(out_width, out_height), CV_8UC3);
    for(int i = 0; i < num; i ++)
    {
    	for(int c = 0; c < out_channel; c ++)
    	{
    		for(int h = 0; h < out_height; h ++ )
    			for(int w = 0; w < out_width; w ++)
    			{
    				for(int ch = 0; ch < 3; ch ++)
    				imgout.at<cv::Vec3b>(h, w)[ch] =(uchar)( bottom_data[i * dim + c * out_height * out_width + h * out_width + w] * 255 );
    			}

    	    sprintf(ss1,"/home/dragon123/cnncode/showimg/%d_%d_pred.jpg",i,c);
    	    imwrite(ss1, imgout);
    	}
    }

    }

}

template <typename Dtype>
void LabelDropoutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

	const Dtype* top_diff = top[0]->cpu_diff();
	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* label = bottom[1]->cpu_data();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	int dim = top[0]->count() / top[0]->num();
	int count = top[0]->count();

	Dtype* mask_data = mask_.mutable_cpu_data();

	for(int i = 0; i < count; i ++)
	{
		bottom_diff[i] = 0;
		if (mask_data[i] > 0)
		{
			bottom_diff[i] = top_diff[i];
		}
	}

}

INSTANTIATE_CLASS(LabelDropoutLayer);
REGISTER_LAYER_CLASS(LabelDropout);
}  // namespace caffe
