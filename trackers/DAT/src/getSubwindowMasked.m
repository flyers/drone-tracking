function [out, mask] = getSubwindowMasked(im, pos, sz)
%GETSUBWINDOWMASKED Extract sub-window with replication-padding centered at pos.
%
% Adapted from GETSUBWINDOW (originally by CSK/KCF (Joao F. Henriques, 
% 2014 - available online: http://www.isr.uc.pt/~henriques).
% 
% Changes: works with (x,y) and (w,h); returns mask indicating padded image
% regions.
	
  xs = floor(pos(1)) + (1:sz(1)) - floor(sz(1)/2);
	ys = floor(pos(2)) + (1:sz(2)) - floor(sz(2)/2);
	
	% Replicate border
  padded_x = xs < 1 | xs > size(im,2);
  padded_y = ys < 1 | ys > size(im,1);
	xs(xs < 1) = 1;
	ys(ys < 1) = 1;
	xs(xs > size(im,2)) = size(im,2);
	ys(ys > size(im,1)) = size(im,1);
	
	% Extract image
	out = im(ys, xs, :);

  % Mark padded regions
  mask = false(size(out,1),size(out,2));
  mask(padded_y,:) = true;
  mask(:,padded_x) = true;
end

