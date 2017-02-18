function [prob_lut, prob_map] = getForegroundBackgroundProbsMasked(frame, mask, num_bins, bin_mapping)
%GETFOREGROUNDBACKGROUNDPROBSMASKED Computes the probability lookup table 
%for the object vs surrounding region model given a masked foreground
%region.
% Parameters:
%   frame       Input (color) image cropped to contain only the surrounding
%               region
%   mask        Binary mask where object pixels are set to 1
%   num_bins    Number of bins per channel (scalar)
%   bin_mapping Maps intensity values to num_bins bins

[rows, cols, layers] = size(frame);

if nargout > 1
  prob_map = zeros(rows, cols);
end

if layers == 3
  % Color image
  obj_hist = zeros(num_bins, num_bins, num_bins, 'double');
  surr_hist = zeros(num_bins, num_bins, num_bins, 'double');

  % Histogram over full image
  [x,y] = meshgrid(1:cols, 1:rows);
  idx_map = sub2ind([rows, cols], y(:), x(:));
  idx_1 = sub2ind([rows, cols, layers], y(:), x(:), ones(numel(x), 1));
  idx_2 = sub2ind([rows, cols, layers], y(:), x(:), 2.*ones(numel(x), 1));
  idx_3 = sub2ind([rows, cols, layers], y(:), x(:), 3.*ones(numel(x), 1));

  bin_1 = bin_mapping(frame(idx_1)+1);
  bin_2 = bin_mapping(frame(idx_2)+1);
  bin_3 = bin_mapping(frame(idx_3)+1);

  idx_hist_full = sub2ind(size(surr_hist), bin_1, bin_2, bin_3);
  bins = unique(idx_hist_full);
  for b = bins
    surr_hist(b) = nnz(idx_hist_full == b);
  end

  % Histogram over object region
  idx_1 = sub2ind([rows, cols, layers], y(mask), x(mask), ones(nnz(mask), 1));
  idx_2 = sub2ind([rows, cols, layers], y(mask), x(mask), 2.*ones(nnz(mask), 1));
  idx_3 = sub2ind([rows, cols, layers], y(mask), x(mask), 3.*ones(nnz(mask), 1));

  bin_1o = bin_mapping(frame(idx_1)+1);
  bin_2o = bin_mapping(frame(idx_2)+1);
  bin_3o = bin_mapping(frame(idx_3)+1);

  idx_hist_obj = sub2ind(size(obj_hist), bin_1o, bin_2o, bin_3o);
  bins = unique(idx_hist_obj);
  for b = bins
    obj_hist(b) = nnz(idx_hist_obj == b);
  end
  
  prob_lut = (obj_hist + 1) ./ (surr_hist + 2);
  if nargout > 1
    prob_map(idx_map) = prob_lut(idx_hist_full);
  end

else
  error('Not supported\n');
end

end

