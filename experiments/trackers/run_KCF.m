function [ res ] = run_KCF( seq, res_path, bSaveImage )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
project_dir = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(project_dir, 'trackers', 'KCF'));
addpath(fullfile(project_dir, 'trackers', 'KCF', 'mex'));


kernel_type = 'gaussian';
kernel.type = kernel_type;
	
features.gray = false;
features.hog = false;

padding = 1.5;  %extra area surrounding the target
lambda = 1e-4;  %regularization
output_sigma_factor = 0.1;  %spatial bandwidth (proportional to target)

interp_factor = 0.02;
		
kernel.sigma = 0.5;

kernel.poly_a = 1;
kernel.poly_b = 9;

features.hog = true;
features.hog_orientations = 9;
cell_size = 4;


target_sz = seq.init_rect(1,[4,3]);
pos = seq.init_rect(1,[2,1]) + floor(target_sz/2);
img_files = seq.s_frames;
video_path = [];
visualization = 0;

%call tracker function with all the relevant parameters
positions = tracker(video_path, img_files, pos, target_sz, ...
    padding, kernel, lambda, output_sigma_factor, interp_factor, ...
    cell_size, features, visualization);

%return results to benchmark, in a workspace variable
rects = [positions(:,2) - target_sz(2)/2, positions(:,1) - target_sz(1)/2];
rects(:,3) = target_sz(2);
rects(:,4) = target_sz(1);
res.type = 'rect';
res.res = rects;

end

