function threshold = getAdaptiveThreshold(prob_map, obj_coords, cfg)
%GETADAPTIVETHRESHOLD Returns the threshold to separate foreground from
%background/surroundings based on cumulative histograms
% Parameters:
%   prob_map   [MxNx1] probability map
%   obj_coords Object rectangle defined as a 4 element vector: [x,y,w,h]
%   cfg        DAT configuration

% Object region
obj_prob_map = imcrop(prob_map, obj_coords);
H_obj =  hist(obj_prob_map(:), cfg.adapt_thresh_prob_bins);
H_obj = H_obj./sum(H_obj);
cum_H_obj = cumsum(H_obj);

% Surroundings
H_dist = hist(prob_map(:), cfg.adapt_thresh_prob_bins);
% Remove object information
H_dist = H_dist - H_obj;
H_dist = H_dist./sum(H_dist);
cum_H_dist = cumsum(H_dist);

k = zeros(size(cum_H_obj));
for i = 1:length(k)-1, k(i) = cum_H_obj(i+1) - cum_H_obj(i); end
x = abs(cum_H_obj - (1 - cum_H_dist)) + (cum_H_obj < 1 - cum_H_dist) + (1 - k);
[~,i] = min(x);
%Final threshold result should lie between 0.4 and 0.7 to be not too restrictive
threshold = max(.4,min(.7, cfg.adapt_thresh_prob_bins(i)));

    
    

