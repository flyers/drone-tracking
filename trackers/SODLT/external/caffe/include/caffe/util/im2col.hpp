// Copyright 2013 Yangqing Jia

#ifndef _CAFFE_UTIL_IM2COL_HPP_
#define _CAFFE_UTIL_IM2COL_HPP_

namespace caffe {

template <typename Dtype>
void im2col_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int ksize, const int pad, const int stride,
    Dtype* data_col);

template <typename Dtype>
void col2im_cpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int psize, const int pad, const int stride,
    Dtype* data_im);

template <typename Dtype>
void im2col_gpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int ksize, const int pad, const int stride,
    Dtype* data_col, int id);

template <typename Dtype>
void col2im_gpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int psize, const int pad, const int stride,
    Dtype* data_im, int id);

template <typename Dtype>
void im2col_uw_cpu(const Dtype* data_im, const int begin, const int end, const int channels,
    const int height, const int width, const int ksize, const int pad, const int stride,
    Dtype* data_col);

template <typename Dtype>
void col2im_uw_cpu(const Dtype* data_col, const int begin, const int end, const int channels,
    const int height, const int width, const int psize, const int pad, const int stride,
    Dtype* data_im);

template <typename Dtype>
void im2col_uw_gpu(const Dtype* data_im, const int begin, const int end, const int channels,
    const int height, const int width, const int ksize, const int pad, const int stride,
    Dtype* data_col, int id);

template <typename Dtype>
void col2im_uw_gpu(const Dtype* data_col, const int begin, const int end, const int channels,
    const int height, const int width, const int psize, const int pad, const int stride,
    Dtype* data_im, int id);

}  // namespace caffe

#endif  // CAFFE_UTIL_IM2COL_HPP_
