function [ results ] = run_MDNet_RANSAC( seq, res_path, bSaveImage )

% if matlabpool('size') == 0
% 	matlabpool open;
% end

project_dir = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(project_dir, 'trackers', 'MDNet', 'pretraining'));
addpath(fullfile(project_dir, 'trackers', 'MDNet', 'tracking'));
addpath(fullfile(project_dir, 'trackers', 'MDNet', 'utils'));
% setup matconvnet
gpuDevice();
matconvnet = fullfile(project_dir, 'trackers', 'MDNet', 'matconvnet', 'matlab', 'vl_setupnn');
run(matconvnet);
% run /home/sliay/Documents/MDNet/matconvnet/matlab/vl_setupnn ;

% set pretrain model file path
net = fullfile(project_dir, 'trackers', 'MDNet', 'models', 'mdnet_vot-otb.mat');
% net = '/home/sliay/Documents/MDNet/models/mdnet_vot-otb.mat';

rects = mdnet_run_ransac(seq.s_frames, seq.init_rect, net, false);

results.res = rects;
results.type='rect';

end

