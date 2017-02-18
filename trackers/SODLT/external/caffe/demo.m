addpath(genpath('./matlab'));
caffe('init_finetune', '/home/winsty/caffe/examples/objectness/imagenet_deploy_solver.prototxt', '/media/windisk/imagenet_snapshot/caffe_objectness_train_iter_100000');
tic;

% test single frame
caffe('set_batch_size', 1);
im = imread('/home/winsty/caffe/dump/90.jpg');
im = single(im);
im = im(:, :, [3 2 1]) - 120;
box = zeros(1, 1, 4, 8, 'single');
image_batch = zeros(100, 100, 3, 8, 'single');
image_batch(:, :, :, 1) = permute(im, [2, 1, 3]);
% dummy target
box(:, :, :, 1) = [0, 0, -1, -1];
input = {image_batch; box};
caffe('forward', input);
fea = caffe('extract_feature', 'fc11');
fea = fea{1};
fea = reshape(fea(1 : 2500), [50, 50]);
fea = 1 ./ (1 + exp(-fea));

% update model
% for i = 1 : 5
%    caffe('update', input);
% end
% toc;