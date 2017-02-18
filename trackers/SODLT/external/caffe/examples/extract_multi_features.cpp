// Copyright 2014 BVLC and contributors.
// Modified by Naiyan Wang
#include <stdio.h>  // for snprintf
#include <cuda_runtime.h>
#include <google/protobuf/text_format.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <string>
#include <vector>
#include <fstream>

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
  const int num_required_args = 5;
  if (argc < num_required_args) {
    LOG(ERROR)<<
    "This program takes in a trained network and an input data layer, and then"
    " extract multiple features of the input data produced by the net.\n"
    "Usage: extract_multi_features  pretrained_net_param"
    "  feature_extraction_proto_file  config_file_name"
    " num_mini_batches  [CPU/GPU]  [DEVICE_ID=0]";
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

  string config_file_name(argv[++arg_pos]);
  std::ifstream in(config_file_name.c_str());
  string blob_name, filename;
  vector<string> blob_names;
  vector<FILE*> file_handles;
  while (in >> blob_name >> filename){
    CHECK(feature_extraction_net->has_blob(blob_name))
      << "Unknown feature blob name " << blob_name
      << " in the network " << feature_extraction_proto;
    blob_names.push_back(blob_name);
    FILE* featureFile = fopen(filename.c_str(), "w");
    if (!featureFile){
      LOG(FATAL) << "Failed to open dumped feature file.";
    }
    file_handles.push_back(featureFile);
  }
  /*string extract_feature_blob_name(argv[++arg_pos]);
  CHECK(feature_extraction_net->has_blob(extract_feature_blob_name))
      << "Unknown feature blob name " << extract_feature_blob_name
      << " in the network " << feature_extraction_proto;

  string save_feature_txt_name(argv[++arg_pos]);
  FILE* featureFile = fopen(save_feature_txt_name.c_str(), "w");
  if (!featureFile){
    LOG(FATAL) << "Failed to open dumped feature file.";
  }*/
  int num_mini_batches = atoi(argv[++arg_pos]);

  LOG(ERROR)<< "Extacting Features";

  Datum datum;
  int num_bytes_of_binary_code = sizeof(Dtype);
  vector<Blob<float>*> input_vec;
  int num_features, dim_features;
  for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index) {
    feature_extraction_net->Forward(input_vec);
    for (int i = 0; i < blob_names.size(); ++i){
      const shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net
        ->blob_by_name(blob_names[i]);
      num_features = feature_blob->num();
      dim_features = feature_blob->count() / num_features;
      const Dtype* feature_blob_data;
      for (int n = 0; n < num_features; ++n) {
        feature_blob_data = feature_blob->cpu_data() +
            feature_blob->offset(n);
        fprintf(file_handles[i], "%d ", 0);
        for (int d = 0; d < dim_features; ++d) {
          if (feature_blob_data[d] != 0){
            fprintf(file_handles[i], "%d:%.6f ", d + 1, feature_blob_data[d]);
          }
        }
        fprintf(file_handles[i], "\n");
      }
    }
    if (batch_index % 10 == 0) {
      LOG(ERROR)<< "Extracted features of " << batch_index * num_features<<
          " query images.";
    }  // for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index)
  }
  LOG(ERROR)<< "Extracted features of " << num_mini_batches * num_features <<
      " query images.";
  
  for (int i = 0; i < file_handles.size(); ++i){
    fclose(file_handles[i]);
  }
  LOG(ERROR)<< "Successfully extracted the features!";
  return 0;
}
