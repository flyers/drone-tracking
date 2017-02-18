function newNN = update_cnn(tmpl, num, negNum, step)
    box = zeros(1, 1, 4, 40, 'single');
    image_batch = zeros(100, 100, 3, 40, 'single');
    for i = 1 : num
        im = single(reshape(tmpl.basis(:,i), [100,100,3]));
        im = im(:,:,[3,2,1]) - 120;
        image_batch(:, :, :, i) = permute(im, [2, 1, 3]);
        
        if i > num - negNum
            box(:, :, :, i) = [0, 0, -1, -1];
        else
            box(:, :, :, i) = floor([tmpl.cord(i,1), tmpl.cord(i,2), tmpl.cord(i,3), tmpl.cord(i,4)] / 2 );
        end
    end
    input = {image_batch; box};
    caffe('set_batch_size', num);
    for iter = 1 : step
        l = caffe('update', input)
    end
end