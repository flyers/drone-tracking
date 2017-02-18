function param = bbox_search_cnn(frame, param, tmpl, background_threshold, frameNum, netId)

num_scale = 4;
neg_num = 16;
p = param.est;
ave_len = (p(3) + p(4)) / 2;


% test in batch to get the probability map of both the postive samples and the negative samples
feas = forward_cnn(tmpl, frame, num_scale + neg_num);


% Collect neg samples with high response on the CNN, which are used for finetune the cnn afterwards
neg_sample = [];
for i = num_scale + 1 : num_scale + neg_num
    if (sum(sum(feas(:, :, i))) > background_threshold)
        neg_sample = [neg_sample, tmpl.basis(:, i)];
    end
end

% bestConf = -1e9;
% bestCord = zeros(1,4);

% hierarchical search in the probability map of different scales to find the best bounding box, details of the searching scheme can be found in the paper
for scale = 1 : num_scale
    curfea = feas(:, :, scale);
    i = 1;
    clear x1_ y1_ x2_ y2_;
    % first get a rough estimation of the bounding box by thresholding the probability map
    for threshold = 0.1 : 0.1 : 0.7
        tmp = curfea > threshold;
        if sum(sum(tmp)) < 10
            break;
        end
        [x1_(i),y1_(i),x2_(i),y2_(i)] = find_box_in_map(tmp);
        i = i + 1;
    end
    if ~exist('x1_', 'var')
        continue;
    end
    cur_cord = mean([x1_', y1_', x2_', y2_'], 1);
    fea = feas(:, :, scale);
    x1 = cur_cord(1); x2 = cur_cord(3);
    y1 = cur_cord(2); y2 = cur_cord(4);

    % param.lastUpdate = 1;
    
    % save the average response of the region, which is then used to determine whether the cnn will be updated
    curconf = mean(mean(fea(x1 : x2, y1 : y2)));

    %% based on the above estimated center location, we then find a more accurate scale of the bounding box
    % calculate the location in the original frame
    scale_ratio = scale * 0.5;
    scale_width = (p(3) + scale_ratio * ave_len ) / 50;
    scale_height = (p(4) + scale_ratio * ave_len ) / 50;
    new_center = [(x1+x2)/2, (y1+y2)/2];
    center_x = (1 + 50)/2;
    center_y = (1 + 50)/2;
    new_p = p;
    new_p(1) = (p(1) + (new_center(1) - center_x ) * scale_width);
    new_p(2) = (p(2) + (new_center(2) - center_y ) * scale_height);


    % search again in the probability map for a more accurate scale.
    h = p(3) + scale_ratio * ave_len;
    w = p(4) + scale_ratio * ave_len;
    oriResizedMap = imresize(fea, [w, h]);
   
    total_scale = 0;
    total_ratio = 0;
    for th = 1 : 0.05 : 1.3
        resizedMap = max(oriResizedMap, 0) * 2 - th;
        % penalize boundary values, otherwise the estimated scale turns to be bigger
        resizedMap(resizedMap < -th + 0.1) = -2;
        iter_center(1) = new_center(1) * h / 50;
        iter_center(2) = new_center(2) * w / 50;
        forDecision = -1e9;
        curRes = -1e9;
        bestScale = 0;
        bestRatio = 0;
        sumMap = integral(double(resizedMap));
        for i = -0.02 : 0.001 : 0.02
            for j = 0.00 : 0.002 : 0.00
                newH = p(3) * (1 + i);
                newW = p(4) * (1 + i) * (1 + j);
                x1 = round(max(1, iter_center(2) - newW / 2));
                x2 = round(min(w, iter_center(2) + newW / 2));
                y1 = round(max(1, iter_center(1) - newH / 2));
                y2 = round(min(h, iter_center(1) + newH / 2));
                x1 = x1 - 1;
                y1 = y1 - 1;
                inConf = get(sumMap, x2, y2) - get(sumMap,x1, y2) - ...
                    get(sumMap, x2, y1) + get(sumMap, x1, y1);
                fmeasure = inConf * (newH * newW) / p(3) / p(4);

                if fmeasure > curRes 
                    curRes = fmeasure;
                    bestScale = i;
                    bestRatio = j;
                    forDecision = inConf / newH / newW;
                end
            end
        end
        if forDecision < 0
            bestScale = 0;
            bestRatio = 0;
        end
        total_scale = total_scale + bestScale;
        total_ratio = total_ratio + bestRatio;
    end
    total_scale = total_scale / 7;
    total_ratio = total_ratio / 7;
    new_p(3) = ( p(3) * (1 + total_scale) );
    new_p(4) = ( p(4) * (1 + total_scale) * (1 + total_ratio));
  
    best_p = new_p;
    param.conf = curconf;
    param.neg_sample = neg_sample;
    param.est = best_p;
    return;
end

% if the code runs here, it means we cannot find the target in any scale, thus the target is reported missing.
param.neg_sample = [];
param.conf = 0;
end

function res = get(map, x, y)
    if x == 0 || y == 0
        res = 0;
    else 
        res = map(x, y);
    end
end
   

function [x1,y1,x2,y2] = find_box_in_map(tmp)

x1 = find( sum(tmp, 1) > 0, 1 );
y1 = find( sum(tmp, 2) > 0, 1 );
x2 = size(tmp, 2) + 1 - find( fliplr(sum(tmp, 1)) > 0, 1 );
y2 = size(tmp, 1) + 1 - find( flipud(sum(tmp, 2)) > 0, 1 );

end