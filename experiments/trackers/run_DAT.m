function [ results ] = run_DAT( seq, res_path, bSaveImage )

project_dir = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(project_dir, 'trackers', 'DAT', 'src'));

visualization = 1;
use_normal_size = 1;
normal_height = 360;
normal_width = 640;
I = imread(seq.s_frames{1});
if use_normal_size
  scale_height = size(I, 1) / normal_height;
  scale_width = size(I, 2) / normal_width;
else
  scale_height = 1;
  scale_width = 1;
end
% Load annotations
groundtruth = seq.init_rect;
groundtruth = groundtruth ./ [scale_width, scale_height, scale_width, scale_height];
% groundtruth = groundtruth / 2;

% Load default settings
cfg = default_parameters_dat();

% Tracking
frames = 1:length(seq.s_frames);
times = zeros(size(frames));
rects = zeros(length(frames), 4);
spf = tic;
for frame = frames
  I = imread(seq.s_frames{frame});
  if use_normal_size
    I = imresize(I, [normal_height, normal_width]);
  end
  
  ttrack = tic;
  if frame == 1
    [state, location] = tracker_dat_initialize(I, groundtruth, cfg); 
  else
    [state, location] = tracker_dat_update(state, I, cfg);
  end
  times(frame) = toc(ttrack);
  
  % Visualization
  % rects(frame, :) = location*2;
  rects(frame, :) = location .* [scale_width, scale_height, scale_width, scale_height];
  if visualization
    figure(1), clf
    imshow(I)
    rectangle('Position', location, 'EdgeColor', 'b', 'LineWidth', 2);
    drawnow
  end
end
spf = toc(spf);

results.res = rects;
results.type='rect';
results.fps = length(frames)/spf;

end

