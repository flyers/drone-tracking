function neg_tmpl =  sample_neg(frame, p, out_size, margin)
    %% Sample 16 negative image samples arround the estimated bounding box with 2 different scales each
    % the parameter margin specifices the distance scale between the original bounding box and the sampled negative boxes
    count = 1;
    p_ori = p;
    len = (p(3) + p(4)) / 2;
    for context_scale = [0.5, 1]   
        for i = -1 : 1
            for j = -1 : 1
                if i == 0 && j == 0
                    continue;
                end
                tx = i * (p_ori(3) / 2 + len * (context_scale + min(margin / len, 0.1)));
                ty = j * (p_ori(4) / 2 + len * (context_scale + min(margin / len, 0.1)));
                p = p_ori;
                p(1) = p(1) + tx;
                p(2) = p(2) + ty;
                p(3) = 2 * len * context_scale;
                p(4) = 2 * len * context_scale;
                crop_img = im_crop(frame, round(p), out_size);
                neg_tmpl(:, count) = crop_img(:);
                count = count + 1;
                
                if i == 0
                    p(1) = p(1) - len * context_scale;
                    crop_img = im_crop(frame, round(p), out_size);
                    neg_tmpl(:, count) = crop_img(:);
                    count = count + 1;
                    
                    p(1) = p(1) + 2 * len * context_scale;
                    crop_img = im_crop(frame, round(p), out_size);
                    neg_tmpl(:, count) = crop_img(:);
                    count = count + 1;
                end
                
                if j == 0
                    p(2) = p(2) - len * context_scale;
                    crop_img = im_crop(frame, round(p), out_size);
                    neg_tmpl(:, count) = crop_img(:);
                    count = count + 1;
                    
                    p(2) = p(2) + 2 * len * context_scale;
                    crop_img = im_crop(frame, round(p), out_size);
                    neg_tmpl(:, count) = crop_img(:);
                    count = count + 1;
                end
            end
        end
    end
  

end