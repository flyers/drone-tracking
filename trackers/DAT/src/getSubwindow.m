function out = getSubwindow(im, pos, sz)
%GETSUBWINDOWMASKED Extract sub-window with replication-padding centered at pos.
%
% Adapted from CSK/KCF (Joao F. Henriques, 2014 - available online:
% http://www.isr.uc.pt/~henriques).
% 
% Changes: works with (x,y) and (w,h);

	xs = floor(pos(1)) + (1:sz(1)) - floor(sz(1)/2);
  ys = floor(pos(2)) + (1:sz(2)) - floor(sz(2)/2);
	
	% Replicate border pixels
	xs(xs < 1) = 1;
	ys(ys < 1) = 1;
	xs(xs > size(im,2)) = size(im,2);
	ys(ys > size(im,1)) = size(im,1);
	
	% Extract image
	out = im(ys, xs, :);
end

