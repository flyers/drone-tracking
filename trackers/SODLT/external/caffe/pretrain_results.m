% test single frame
caffe('set_batch_size', 8);
for i = 1 : 8 : 128
    box = zeros(1, 1, 4, 8, 'single');
    image_batch = zeros(100, 100, 3, 8, 'single');
    for j = i : i + 8
        im = imread(['/home/winsty/caffe/dump/' num2str(j) '.jpg']);
        im = single(im);
        im = im(:, :, [3 2 1]) - 120;
        image_batch(:, :, :, j - i + 1) = permute(im, [2, 1, 3]);
        % dummy target
        box(:, :, :, j - i + 1) = [0, 0, -1, -1];
    end
    input = {image_batch; box};
    caffe('forward', input);
    fea = caffe('extract_feature', 'fc11');
    fea = fea{1};
    fea = reshape(fea, [50, 50, 8]);
    fea = 1 ./ (1 + exp(-fea));
    j = i + 7;
    im = single(imread(['/home/winsty/caffe/dump/' num2str(j) '.jpg'])) / 255;
    mask = imresize(fea(:, :, j - i + 1)', [100, 100]);
    for k = 1 : 3
        masked(:, :, k) = min(im(:, :, k), mask);
    end
    h = figure; imshow([im, masked]);
    saveas(h, [num2str(j) '.jpg']);
end