## Visual object tracking for unmanned aerial vehicles: A benchmark and new motion models

Created by [Siyi Li](http://lisiyi.me) and [Dit-Yan Yeung](http://home.cse.ust.hk/~dyyeung) at HKUST.

### Introduction
Drone Tracking Benchmark (DTB70) is a unified tracking benchmark on the drone platform. 
In this benchmark, we provide an extensive study of the state-of-the-art trackers and their various motion model variants on the DTB70 dataset.
Detailed description of the benchmark can be found in our [paper](http://lisiyi.me/paper/AAAI17_UAV.pdf).

### Citation
If you are using this code in a publication, please cite our paper.

    @inproceedings{drone-tracking,
	    title={Visual object tracking for unmanned aerial vehicles: A benchmark and new motion models},
	    author={Li, Siyi and Yeung, Dit-Yan},
	    booktitle = {AAAI},
	    year={2017}
    }
    
### Requirements
* MATLAB
* OpenCV 2.4
* [mexopencv 2.4](https://github.com/kyamagu/mexopencv/tree/v2.4)

For the installation of specific trackers, please refer to the corresponding documentation.

### Download dataset
Download the dataset from [Baiduyun link](https://pan.baidu.com/s/1SftGHD7SyIFyBXExHbbYAQ).
Put the unzipped file under the *data* directory.
Also, change the dataset path config in file *experiments/util/configDTBSeqs.m*.

The dataset format follows [OTB50](http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html).

### How to install individual trackers
* DAT tracker is ready to run.
* DSST, HOG_LR, KCF, MEEM, SRDCF all need to compile mex files. Just use the compilation script in the corresponding directories.
* SODLT and MDNet are deep learning based trackers. Please refer to the detailed documentation.

### Run demo examples
Run the *run_demo.m* script.

### Run evaluation toolkit
Run the *main_running.m* script under the *experiments* directory. You can config the trackers list in file *experiments/util/configTrackers.m*.

For any problems, feel free to propose issues or contact the author sliay@connect.ust.hk.
