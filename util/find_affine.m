function [feat, H] = find_affine(feat, im, target_pos, target_size)
%% estimate the affine transformation from the previous image to current image

cur_keypoints = feat.detector.detect(im);
cur_descriptors = feat.extractor.compute(im, cur_keypoints);

feat.matcher.clear();
feat.matcher.add(feat.last_descriptors);
feat.matcher.train();
matches = feat.matcher.match(cur_descriptors);

[~, odx] = sort([matches.distance]);
good_matches = matches(odx);
good_matches = good_matches(1:min(100, numel(matches)));

last_pts = cat(1, feat.last_keypoints([good_matches.trainIdx] + 1).pt);
cur_pts = cat(1, cur_keypoints([good_matches.queryIdx] + 1).pt);

%% discard the feature points lie in target area
idx1 = ( last_pts(:, 1) > target_pos(2) - target_size(2)*0.6 ) & ( last_pts(:, 1) < target_pos(2) + target_size(2)*0.6 );
idx2 = ( last_pts(:, 2) > target_pos(1) - target_size(1)*0.6 ) & ( last_pts(:, 2) < target_pos(1) + target_size(1)*0.6 );
idx = ~(idx1 & idx2);

last_pts = num2cell(last_pts(idx, :), 2);
cur_pts = num2cell(cur_pts(idx, :), 2);

%% if there is less than 50 points, H is hard assigned as I
if size(last_pts, 1) < 50
	H = eye(3);
else
	H = cv.estimateRigidTransform(last_pts, cur_pts, 'FullAffine', true);
	if isempty(H)
		H = eye(3);
	else
		H = [H; 0, 0, 1];
	end
end

feat.last_keypoints = cur_keypoints;
feat.last_descriptors = cur_descriptors;

% target_pos = H * [target_pos(2); target_pos(1); 1];
% target_pos = [ target_pos(2)/target_pos(3), target_pos(1)/target_pos(3) ];
