function rects = poly2rect(polygon)
%POLY2RECT Convert polygon to rectangle
% Compute axis aligned bounding boxes with correct area and center
cx = mean(polygon(:, 1:2:end), 2);
cy = mean(polygon(:, 2:2:end), 2);
x1 = min(polygon(:, 1:2:end), [], 2);
x2 = max(polygon(:, 1:2:end), [], 2);
y1 = min(polygon(:, 2:2:end), [], 2);
y2 = max(polygon(:, 2:2:end), [], 2);
A1 = sqrt(sum((polygon(:, 1:2) - polygon(:, 3:4)).^2, 2)) .* sqrt(sum((polygon(:, 3:4) - polygon(:, 5:6)).^2, 2));
A2 = (x2 - x1) .* (y2 - y1);
s = sqrt(A1./A2);
w = s .* (x2 - x1) + 1;
h = s .* (y2 - y1) + 1;
rects = round([[cx cy] - [w h]./2, w, h]);
