function img = im_crop(ori, p, out_size)
    %% crop part of the image according to the given bounding box coordinates
    %% for out-of-boundary part, just use an constance value
    sz = size(ori);
    p(1) = floor(p(1)); p(2) = floor(p(2));
    p(3) = ceil(p(3)); p(4) = ceil(p(4));
    img = ((ones(p(4), p(3), 3) * 120));
    
    x1 = round(p(1) - p(3) / 2);
    y1 = round(p(2) - p(4) / 2);
    x2 = x1 + p(3) - 1;
    y2 = y1 + p(4) - 1;
    crop_x1 = max(1, -x1 + 2); x1 = max(1, x1);
    crop_y1 = max(1, -y1 + 2); y1 = max(1, y1);
    crop_x2 = min(p(3), p(3) + sz(2) - x2); x2 = min(x2, sz(2));
    crop_y2 = min(p(4), p(4) + sz(1) - y2); y2 = min(y2, sz(1));
    img( crop_y1 : crop_y2, crop_x1 : crop_x2, :) = ori(y1 : y2, x1 : x2, :);
    
    if nargin > 2
        img = imresize(img, [out_size, out_size], 'Antialiasing', false);
    end
end
