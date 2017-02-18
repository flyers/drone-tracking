#include <cuda_runtime.h>
#include <fcntl.h>
#include <google/protobuf/text_format.h>
#include <iostream>
#include <cstring>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/solver.hpp"

using namespace caffe;
using namespace std;
int main(int argc, char** argv){
  if (argc < 3){
    printf("Not enough parameters.\n");
    return 1;
  }
  // Read in old net parameters
  string trained_filename(argv[1]);
  NetParameter paramOld;
  ReadProtoFromBinaryFile(trained_filename, &paramOld);
  int num_source_layers = paramOld.layers_size();

  // New net parameters
  NetParameter param;
  param.Clear();

  for (int i = 0; i < num_source_layers; ++i) {
    const LayerParameter& source_layer = paramOld.layers(i).layer();
    const string& source_layer_name = source_layer.name();
    string name = source_layer_name;
    if (source_layer_name[source_layer_name.size() - 1] == '_'){
      name = source_layer_name.substr(0, source_layer_name.size() - 1);
    }
    LayerConnection* layer_connection = param.add_layers();
    layer_connection->CopyFrom(paramOld.layers(i));
    layer_connection->mutable_layer()->set_name(name);
    cout << name << endl;
    name = name + "_";
    layer_connection = param.add_layers();
    layer_connection->CopyFrom(paramOld.layers(i));
    layer_connection->mutable_layer()->set_name(name);
  }

  WriteProtoToBinaryFile(param, argv[2]);

	return 0;
}