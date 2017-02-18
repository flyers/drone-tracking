function [ results ] = run_DSST( seq, res_path, bSaveImage )


project_dir = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(project_dir, 'trackers', 'DSST'));
addpath(fullfile(project_dir, 'trackers', 'DSST', 'mex'));

rng(0);
%parameters according to the paper
params.padding = 1;         			% extra area surrounding the target
params.output_sigma_factor = 1/16;		% standard deviation for the desired translation filter output
params.scale_sigma_factor = 1/4;        % standard deviation for the desired scale filter output
params.lambda = 1e-2;					% regularization weight (denoted "lambda" in the paper)
params.learning_rate = 0.025;			% tracking model learning rate (denoted "eta" in the paper)
params.number_of_scales = 33;           % number of scale levels (denoted "S" in the paper)
params.scale_step = 1.02;               % Scale increment factor (denoted "a" in the paper)
params.scale_model_max_area = 512;      % the maximum size of scale examples

params.visualization = 0;

params.use_normal_size = true;
params.normal_width = 640;
params.normal_height = 360;

rect = seq.init_rect;
pos = [rect(2), rect(1)];
target_sz = [rect(4), rect(3)];
video_path = '';
img_files = seq.s_frames;

params.init_pos = floor(pos) + floor(target_sz/2);
params.wsize = floor(target_sz);
params.img_files = img_files;
params.video_path = video_path;

[positions, fps] = dsst(params);

rects(:, 1) = positions(:, 2) - positions(:, 4)/2;
rects(:, 2) = positions(:, 1) - positions(:, 3)/2;
rects(:, 3:4) = positions(:, 4:-1:3);

results.res = rects;
results.type='rect';
results.fps = fps;

end

