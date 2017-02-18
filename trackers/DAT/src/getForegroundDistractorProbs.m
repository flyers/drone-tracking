function prob_lut = getForegroundDistractorProbs(frame, obj_rect, distractors, num_bins, bin_mapping)
%GETFOREGROUNDDISTRACTORPROBS Computes the probability lookup table 
%for the object vs distractor model.
% Parameters:
%   frame       Input (color) image 
%   obj_rect    Rectangular object region
%   distractors Nx4 matrix where each row corresponds to a rectangular
%               distractor region
%   num_bins    Number of bins per channel (scalar)
%   bin_mapping Maps intensity values to num_bins bins

  [rows,cols,layers] = size(frame);
  obj_rect = round(obj_rect);

  num_distr = size(distractors,1);

  if layers == 3 % Color image
    obj_hist = zeros(num_bins, num_bins, num_bins, 'double');
    distr_hist = zeros(num_bins, num_bins, num_bins, 'double');

    % Mask object and distracting regions
    Md = false(size(frame,1),size(frame,2));
    Mo = false(size(frame,1),size(frame,2));
    Mo(obj_rect(2):obj_rect(2)+obj_rect(4),obj_rect(1):obj_rect(1)+obj_rect(3)) = true;
    for i = 1:num_distr
      Md(distractors(i,2):distractors(i,2)+distractors(i,4),distractors(i,1):distractors(i,1)+distractors(i,3)) = true;
    end

    [x,y] = meshgrid(1:cols, 1:rows);
    xo = x(Mo);
    yo = y(Mo);
    xd = x(Md);
    yd = y(Md);

    % Distractor histogram
    idx1 = sub2ind([rows, cols, layers], yd, xd, ones(numel(xd), 1));
    idx2 = sub2ind([rows, cols, layers], yd, xd, 2.*ones(numel(xd), 1));
    idx3 = sub2ind([rows, cols, layers], yd, xd, 3.*ones(numel(xd), 1));

    bin1 = bin_mapping(frame(idx1)+1);
    bin2 = bin_mapping(frame(idx2)+1);
    bin3 = bin_mapping(frame(idx3)+1);

    idx_hist_dist = sub2ind(size(distr_hist), bin1, bin2, bin3);
    bins = unique(idx_hist_dist);
    for b = bins
      distr_hist(b) = nnz(idx_hist_dist == b);
    end

    % Object histogram
    idx1 = sub2ind([rows, cols, layers], yo, xo, ones(numel(xo), 1));
    idx2 = sub2ind([rows, cols, layers], yo, xo, 2.*ones(numel(xo), 1));
    idx3 = sub2ind([rows, cols, layers], yo, xo, 3.*ones(numel(xo), 1));

    bin1 = bin_mapping(frame(idx1)+1);
    bin2 = bin_mapping(frame(idx2)+1);
    bin3 = bin_mapping(frame(idx3)+1);

    idx_hist_obj = sub2ind(size(obj_hist), bin1, bin2, bin3);
    bins = unique(idx_hist_obj);
    for b = bins
      n = nnz(idx_hist_obj == b);
      obj_hist(b) = n * num_distr;
      distr_hist(b) = distr_hist(b) + n * num_distr;
    end

    prob_lut = (obj_hist + 1) ./ (distr_hist + 2);
  elseif layers == 1
    obj_hist = zeros(num_bins, 1, 'double');
    distr_hist = zeros(num_bins, 1, 'double');

    % Mask object and distracting regions
    Md = false(size(frame,1),size(frame,2));
    Mo = false(size(frame,1),size(frame,2));
    Mo(obj_rect(2):obj_rect(2)+obj_rect(4),obj_rect(1):obj_rect(1)+obj_rect(3)) = true;
    for i = 1:num_distr
      Md(distractors(i,2):distractors(i,2)+distractors(i,4),distractors(i,1):distractors(i,1)+distractors(i,3)) = true;
    end

    [x,y] = meshgrid(1:cols, 1:rows);
    xo = x(Mo);
    yo = y(Mo);
    xd = x(Md);
    yd = y(Md);

    % Distractor histogram
    idx1 = sub2ind([rows, cols], yd, xd);

    bin1 = bin_mapping(frame(idx1)+1);

    idx_hist_dist = sub2ind(size(distr_hist), bin1);
    bins = unique(idx_hist_dist);
    for b = bins
      distr_hist(b) = nnz(idx_hist_dist == b);
    end

    % Object histogram
    idx1 = sub2ind([rows, cols, layers], yo, xo, ones(numel(xo), 1));

    bin1 = bin_mapping(frame(idx1)+1);

    idx_hist_obj = sub2ind(size(obj_hist), bin1);
    bins = unique(idx_hist_obj);
    for b = bins
      n = nnz(idx_hist_obj == b);
      obj_hist(b) = n * num_distr;
      distr_hist(b) = distr_hist(b) + n * num_distr;
    end

    prob_lut = (obj_hist + 1) ./ (distr_hist + 2);
  else
    error('Color space not supported');
  end
end




 



