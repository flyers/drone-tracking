function mapping = getBinMapping(num_bins)
%GETBINMAPPING Returns a vector to map intensity values in [0 255] to 
% num_bins bins with uniform bin width.
mapping = arrayfun(@(i) fix(i/(256/num_bins))+1, 0:255, 'UniformOutput', true);
end

