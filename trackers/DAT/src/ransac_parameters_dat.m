function [ cfg ] = ransac_parameters_dat()
%DEFAULT_PARAMETERS_DAT Default parametrization
  cfg = struct('show_figures', false);
  
  % Image scaling
  cfg.img_scale_target_diagonal = 75; % Length of object hypothesis diagonal (Used to downscale image)
  
  % Search and surrounding regions
  cfg.search_win_padding = 2; % Search win = hypothesis + X * max(w,h of hypothesis)
  cfg.surr_win_factor = 1.9;  % Surrounding win = X * surr_win_factor
  
  % Appearance model
  cfg.color_space = 'rgb';                % 'rgb', 'hsv', or 'lab'
  cfg.num_bins = 16;                      % Number of bins per channel
  cfg.bin_mapping = getBinMapping(cfg.num_bins); % Maps pixel values from [0, 255] to the corresponding bins
  cfg.prob_lut_update_rate = .05;         % Update rate for LUT
  cfg.distractor_aware = true;            % Toggle distractor-awareness
  cfg.adapt_thresh_prob_bins = 0:0.05:1;  % Bins for adaptive threshold. 
  
  % Motion estimation
  cfg.motion_estimation_history_size = 5;  % Motion based on past X frames (Set to 0 to disable motion estimation)
  
  % NMS-based localization
  cfg.nms_scale = 1;                  % NMS yields rects scaled by X - no scaling typically achieves best localization (Lower scales yield more distractors - so choose wisely)
  cfg.nms_overlap = .9;               % Overlap between candidate rectangles for NMS
  cfg.nms_score_factor = .5;          % Report all rectangles with score >= X * best_score
  cfg.nms_include_center_vote = true; % Prefer hypothesis with highly confident center regions
  
  % camera correction part
  % cfg.transform = '';
  cfg.transform = 'projective';
%   cfg.transform = 'affine';
end

