function fea = forward_cnn(tmpl, frame, num)
	caffe('set_batch_size', num);
    box = zeros(1, 1, 4, 40, 'single');
    image_batch = zeros(100, 100, 3, 40, 'single');
    for i = 1:num
    	im = single(reshape(tmpl.basis(:,i), [100,100,3]));
        im = im(:,:,[3,2,1]) - 120;
        image_batch(:, :, :, i) = permute(im, [2, 1, 3]);
        box(:, :, :, i) = [0, 0, -1, -1];
    end
    input = {image_batch; box};
    caffe('set_phase_test');
    caffe('forward', input);
%     caffe('set_phase_train');
    fea = caffe('extract_feature', 'fc11');
    fea = fea{1};
    fea = reshape(fea(1 : 50*50*num), [50, 50, num]);
    fea = 1 ./ (1 + exp(-fea));
    for i = 1 : num
        fea(:, :, i) = fea(:, :, i)';
    end
end
