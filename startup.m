function startup()
% startup function

cur_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(cur_dir, 'experiments'));
addpath(fullfile(cur_dir, 'experiments', 'rstEval'));
addpath(fullfile(cur_dir, 'experiments', 'util'));
addpath(fullfile(cur_dir, 'util'));

% add mexopencv path
addpath('/home/sliay/Documents/mexopencv');
fprintf('startup config finished.\n');