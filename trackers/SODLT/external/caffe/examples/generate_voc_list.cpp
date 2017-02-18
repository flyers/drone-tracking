#include <fstream>
#include <string>
#include <map>
#include <vector>

using namespace std;
const int classNum = 20;
const int MAX_IMAGE_NUM = 5000;
const string classList[] = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
	"chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"};
const string outFileName = "voc_2007_train.txt";
const string fullPath = "/home/winsty/VOC07/VOCdevkit/VOC2007/";
const string phase = "train";
vector<string> filenames;
map<string, int> fileMapping;
int labels[MAX_IMAGE_NUM][classNum];
int tryGetId(const string& name){
	if (fileMapping.find(name) == fileMapping.end()){
		fileMapping[name] = filenames.size();
		filenames.push_back(name);
		return filenames.size() - 1;
	} else {
		return fileMapping[name];
	}
}
int main(){
	ofstream outFile(outFileName.c_str());
	outFile << classNum << endl;
	for (int i = 0; i < classNum; ++i){
		string fullName = fullPath + "ImageSets/Main/" + classList[i] + "_" + phase + ".txt";
		ifstream infile(fullName.c_str());
		string filename;
		int label;
		while (infile >> filename >> label){
			labels[tryGetId(filename)][i] = (label != -1);
		}
		infile.close();
	}
	for (int i = 0; i < filenames.size(); ++i){
		outFile << fullPath + "JPEGImages/" + filenames[i] + ".jpg";
		for (int j = 0; j < classNum; ++j){
			outFile << " " << labels[i][j];
		}
		outFile << endl;
	}
	outFile.close();
	return 0;
}
