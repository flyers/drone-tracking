// Copyright 2014 BVLC and contributors.
// Modified by Naiyan Wang
#include <stdio.h>  // for snprintf
#include <cuda_runtime.h>
#include <google/protobuf/text_format.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv);

int main(int argc, char** argv) {
  return feature_extraction_pipeline<float>(argc, argv);
//  return feature_extraction_pipeline<double>(argc, argv);
}

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv) {
  const int num_required_args = 6;
  if (argc < num_required_args) {
    LOG(ERROR)<<
    "This program takes in a trained network and an input data layer, and then"
    " extract features of the input data produced by the net.\n"
    "Usage: extract_features  pretrained_net_param"
    "  feature_extraction_proto_file  extract_feature_blob_name"
    "  save_feature_txt_name num_mini_batches  [CPU/GPU]  [DEVICE_ID=0] [data/diff]";
    return 1;
  }
  int arg_pos = num_required_args;

  arg_pos = num_required_args;
  if (argc > arg_pos && strcmp(argv[arg_pos], "GPU") == 0) {
    LOG(ERROR)<< "Using GPU";
    uint device_id = 0;
    if (argc > arg_pos + 1) {
      device_id = atoi(argv[arg_pos + 1]);
      CHECK_GE(device_id, 0);
    }
    LOG(ERROR) << "Using Device_id=" << device_id;
    Caffe::SetDevice(device_id);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(ERROR) << "Using CPU";
    Caffe::set_mode(Caffe::CPU);
  }
  Caffe::set_phase(Caffe::TEST);
  bool useData;
  if (string(argv[arg_pos + 2]) == "diff"){
    useData = false;
  } else if (string(argv[arg_pos + 2]) == "data"){
    useData = true;
  } else {
    LOG(FATAL) << "Unknown type of data to extract.";
  }
  arg_pos = 0;  // the name of the executable
  string pretrained_binary_proto(argv[++arg_pos]);

//  Caffe::set_phase(Caffe::VAL);
  // Expected prototxt contains at least one data layer such as
  //  the layer data_layer_name and one feature blob such as the
  //  fc7 top blob to extract features.
  /*
   layers {
     name: "data_layer_name"
     type: DATA
     data_param {
       source: "/path/to/your/images/to/extract/feature/images_leveldb"
       mean_file: "/path/to/your/image_mean.binaryproto"
       batch_size: 128
       crop_size: 227
       mirror: false
     }
     top: "data_blob_name"
     top: "label_blob_name"
   }
   layers {
     name: "drop7"
     type: DROPOUT
     dropout_param {
       dropout_ratio: 0.5
     }
     bottom: "fc7"
     top: "fc7"
   }
   */
  string feature_extraction_proto(argv[++arg_pos]);
  shared_ptr<Net<Dtype> > feature_extraction_net(
      new Net<Dtype>(feature_extraction_proto));
  feature_extraction_net->CopyTrainedLayersFrom(pretrained_binary_proto);

  string extract_feature_blob_name(argv[++arg_pos]);
  CHECK(feature_extraction_net->has_blob(extract_feature_blob_name))
      << "Unknown feature blob name " << extract_feature_blob_name
      << " in the network " << feature_extraction_proto;

  string save_feature_txt_name(argv[++arg_pos]);
  FILE* featureFile = fopen(save_feature_txt_name.c_str(), "w");
  if (!featureFile){
    LOG(FATAL) << "Failed to open dumped feature file.";
  }
  int num_mini_batches = atoi(argv[++arg_pos]);

  LOG(ERROR)<< "Extacting Features";

  Datum datum;
  int num_bytes_of_binary_code = sizeof(Dtype);
  vector<Blob<float>*> input_vec;
  int image_index = 0;
  for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index) {
    if (useData){
      feature_extraction_net->Forward(input_vec);
    } else {
      feature_extraction_net->ForwardBackward(input_vec);
    }
    const shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net
        ->blob_by_name(extract_feature_blob_name);
    // const shared_ptr<Blob<Dtype> > label_blob = feature_extraction_net
    //     ->blob_by_name("label");
    int num_features = feature_blob->num();
    int dim_features = feature_blob->count() / num_features;
    Dtype* feature_blob_data;

    for (int n = 0; n < num_features; ++n) {
      if (useData){
        feature_blob_data = feature_blob->mutable_cpu_data() + feature_blob->offset(n);
      } else {
        feature_blob_data = feature_blob->mutable_cpu_diff() + feature_blob->offset(n);
      }
      //fprintf(featureFile, "%d ",  (int)label_blob->cpu_data()[n]);
      fprintf(featureFile, "%d ", 0);
      for (int d = 0; d < dim_features; ++d) {
        /*if (feature_blob_data[d] != 0){
          fprintf(featureFile, "%d:%.6f ", d + 1, feature_blob_data[d]);
        }*/
        fprintf(featureFile, "%.6f ", feature_blob_data[d]);
      }
      fprintf(featureFile, "\n");
      ++image_index;
      if (image_index % 1000 == 0) {
        LOG(ERROR)<< "Extracted features of " << image_index <<
            " query images.";
      }
    }  // for (int n = 0; n < num_features; ++n)
  }  // for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index)
  // write the last batch
  if (image_index % 1000 != 0) {
    LOG(ERROR)<< "Extracted features of " << image_index <<
        " query images.";
  }
  fclose(featureFile);
  LOG(ERROR)<< "Successfully extracted the features!";
  return 0;
}
