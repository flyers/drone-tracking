#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
using namespace std;

int main(int argc, char** argv){
	if (argc < 7){
		printf("Usage: aggregate OUTPUT_FILE OUTPUT_LABEL SAMPLE_NUM VIEW_NUM CLASS_NUM ENSEMBLE_OUTPUT\n");
		return 0;
	}
	FILE* outputDataFile = fopen(argv[1], "r");
	FILE* outputLabelFile = fopen(argv[2], "r");
	FILE* ensembleOutputFile = fopen(argv[6], "w");
	int sampleNum = atoi(argv[3]);
	int viewNum = atoi(argv[4]);
	int classNum = atoi(argv[5]);
	int top1Correct = 0;
	int top5Correct = 0;
	for (int i = 0; i < sampleNum; ++i){
		vector<double> aggregate(classNum);
		vector<pair<double, int> > forSort;
		double temp;
		int label;
		for (int j = 0; j < viewNum; ++j){
			for (int k = 0; k < classNum; ++k){
				fscanf(outputDataFile, "%lf", &temp);
				aggregate[k] += temp;
			}
			fscanf(outputLabelFile, "%d", &label);
		}
		for (int j = 0; j < classNum; ++j){
      aggregate[j] /= viewNum;
			forSort.push_back(make_pair(aggregate[j], j));
		}
    sort(forSort.begin(), forSort.end());
    //printf("%lf %d %d\n", forSort[forSort.size() - 1].first, forSort[forSort.size() - 1].second, label);
		if (forSort[classNum - 1].second == label) ++top1Correct;
		for (int i = 0; i < 5; ++i){
			if (forSort[classNum - i - 1].second == label){
				++top5Correct;
				break;
			}
		}
		for (int i = 0; i < classNum; ++i){
			fprintf(ensembleOutputFile, "%e ", aggregate[i]);
		}
		fprintf(ensembleOutputFile, "\n");
	}
	printf("Top 1 correct: %lf\n", top1Correct / double(sampleNum));
	printf("Top 5 correct: %lf\n", top5Correct / double(sampleNum));
	fclose(outputDataFile);
	fclose(outputLabelFile);
	fclose(ensembleOutputFile);
	return 0;
}
