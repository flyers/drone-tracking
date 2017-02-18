close all;
name = 'MotorRolling'; 
dataPath = ['./data/', name];

opt.use_normalize = 1;
opt.normal_width = 320;
opt.normal_height = 240;
opt.show_img = 1;

% Load data
disp('Loading data...');
fullPath = [dataPath, '/img/'];
d = dir([fullPath, '*.jpg']);
if size(d, 1) == 0
    d = dir([fullPath, '*.png']);
end
if size(d, 1) == 0
    d = dir([fullPath, '*.bmp']);
end
if strcmp(name, 'Jogging') == 0
    rects = importdata([dataPath, '/groundtruth_rect.txt']);
else
    rects = importdata([dataPath, '/groundtruth_rect.1.txt']);
    % rects = importdata([dataPath, '/groundtruth_rect.2.txt']);
end
p = rects(1,:);

seq.init_rect = [p(1), p(2), p(3), p(4)];
im = imread([fullPath, d(1).name]);
data = zeros(size(im, 1), size(im, 2), size(d, 1));
seq.s_frames = cell(size(d, 1), 1);
for i = 1 : size(d, 1)
    seq.s_frames{i} = [fullPath, d(i).name];
end
seq.opt = opt;
results = run_SODLT(seq, '', false);
% save([title '_res'], 'results');
