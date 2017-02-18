function [tmpl, cord] = sample_pos(frame, p, out_size, random, num_scale)
    %% sample positive samples with different scales and translations

    len = (p(3)+p(4)) / 2;
    for iter = 1 : 2
        for i = 1 : num_scale
            context_scale = 0.5 * (i);
            translationX = randi(floor(len / 2 * context_scale)) - 1;
            translationY = randi(floor(len / 2 * context_scale)) - 1;
            if randi(2) == 1
                translationX = -translationX;
            end
            if randi(2) == 1
                translationY = -translationY;
            end
            if random == 0
                translationX = 0;
                translationY = 0;
            end
            p_sample = [p(1) + translationX, p(2) + translationY, p(3) + len * context_scale, p(4) + len * context_scale];
            crop_img = im_crop(frame, p_sample);
            scale_height = size(crop_img,1) / out_size;
            scale_width = size(crop_img,2) / out_size;
            center_x = (1+out_size) / 2 - translationX / scale_width;
            center_y = (1+out_size) / 2 - translationY / scale_height;
            x1 = floor(center_x - (p(3) / 2 ) / scale_width);
            y1 = floor(center_y - (p(4) / 2 ) / scale_height);
            x2 = ceil(center_x + (p(3) / 2 ) / scale_width);
            y2 = ceil(center_y + (p(4) / 2 ) / scale_height);
            %minus 1 for numeric reasons
            cord(iter * num_scale - num_scale + i, :) = [x1, y1, x2, y2] - 1 ;
            crop_img = imresize(crop_img, [out_size, out_size], 'Antialiasing', false);
            tmpl(:,iter * num_scale - num_scale + i) = crop_img(:);
        end
    end
end
