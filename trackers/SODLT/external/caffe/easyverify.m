% addpath(genpath('./matlab'));
% caffe('init_finetune', '/home/winsty/caffe/examples/objectness/imagenet_deploy_solver.prototxt', '/media/windisk/imagenet_snapshot/caffe_objectness_train_iter_100000');
% tic;

% test single frame
caffe('set_batch_size', 1);


track_dir = dir('../trackingDataset');
track_dir = track_dir(3:52);
for i = 1:50
    dataname = track_dir(i).name;
    im = imread(['../trackingDataset/' dataname '/img/0001.jpg' ]);
    if strcmp(dataname,'Jogging')
        grt = importdata(['../trackingDataset/' dataname '/groundtruth_rect.1.txt']);
        %[x,y,width,height] = importdata(['../trackingDataset/' dataname '/groundtruth_rect.2.txt']);
        x = grt(1,1); y = grt(1,2);
        width = grt(1,3); height = grt(1,4);
    else
        grt = importdata(['../trackingDataset/' dataname '/groundtruth_rect.txt']);
        x = grt(1,1); y = grt(1,2);
        width = grt(1,3); height = grt(1,4);
    end

    padding_x = max(x-width/2,1); padding_y = max(y-height/2,1);
    w = min(2*width, size(im,2)-padding_x);
    h = min(2*height, size(im,1)-padding_y);
    crop_img = im(padding_y:(padding_y+h), padding_x:(padding_x+w), :);

    crop_img = im;
    crop_img = imresize(crop_img, [100,100]);
    crop_img = single(crop_img);
    if(size(crop_img,3) == 1)
    	crop_img = repmat(crop_img, [1,1,3]);
    end
    ori_img = crop_img/255;
    crop_img = crop_img(:,:,[3 2 1]) - 120;
    box = zeros(1,1,4,8,'single');
    image_batch = zeros(100,100,3,8,'single');
    image_batch(:,:,:,1) = permute(crop_img, [2,1,3]);

    % dummy target
    box(:, :, :, 1) = [0, 0, -1, -1];
    input = {image_batch; box};
    caffe('forward', input);
    fea = caffe('extract_feature', 'fc11');
    fea = fea{1};
    fea = reshape(fea(1 : 2500), [50, 50]);
    fea = 1 ./ (1 + exp(-fea));
    mask = imresize(fea',[100,100]);
    for k = 1:3
        masked(:,:,k) = min(ori_img(:,:,k), mask);
    end
    h  = figure; imshow([ori_img, masked]);
    saveas(h, [num2str(i) dataname '.jpg']);

end
