function res = project_t(pts, H)
%% apply projective transformation H on points pts
%% pts, n x 2
%% H, 3 x 3
%% res, n x 2

pts = [pts, ones(size(pts, 1), 1)];
res = H * pts';
res = [ res(1,:)./res(3,:); res(2,:)./res(3,:) ];
res = res';
