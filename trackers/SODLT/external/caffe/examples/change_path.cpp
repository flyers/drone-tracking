#include <fstream>
#include <vector>
#include "caffe/common.hpp"

using namespace caffe;
using namespace std;
int main(){
  std::ifstream infile("/media/windisk/imagenet-DET/window_file_full.txt");
  std::ofstream outfile("/media/windisk/imagenet-DET/window_file.txt");
  string hashtag;
  int image_index, channels;
  if (!(infile >> hashtag >> image_index)) {
    LOG(FATAL) << "Window file is empty";
  }
  
  do {
    CHECK_EQ(hashtag, "#");
    outfile << hashtag << " " << image_index << '\n';
    // read image path
    string image_path;
    infile >> image_path;
    for (int i = 0; i < image_path.size(); ++i){
      if (image_path[i] == '\\') image_path[i] = '/';
    }
    int pos = image_path.find("train");
    if (pos != -1){
      outfile << "/media/windisk/imagenet-DET/ILSVRC2014_DET_train/" + image_path.substr(pos + 6) << '\n';
    } else {
      pos = image_path.find("val");
      outfile << "/media/windisk/imagenet-DET/ILSVRC2013_DET_val/" + image_path.substr(pos + 4) << '\n';
    }

    // read image dimensions
    vector<int> image_size(3);
    infile >> image_size[0] >> image_size[1] >> image_size[2];
    outfile << image_size[0] << "\n" << image_size[1] << " " << image_size[2] << '\n';
    // read each box
    int num_windows;
    infile >> num_windows;
    outfile << num_windows << '\n';
    for (int i = 0; i < num_windows; ++i) {
      int label, x1, y1, x2, y2;
      float overlap;
      infile >> label >> overlap >> x1 >> y1 >> x2 >> y2;
      outfile << label << " " << overlap << " " << x1 << " " << y1 << " " <<
         x2 << " " << y2 << '\n';
    }

    if (image_index % 100 == 0) {
      LOG(INFO) << "num: " << image_index << " "
          << image_path << " "
          << image_size[0] << " "
          << image_size[1] << " "
          << image_size[2] << " "
          << "windows to process: " << num_windows;
    }
  } while (infile >> hashtag >> image_index);
  return 0;
}