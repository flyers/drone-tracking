function [state, location] = tracker_dat_update(state, I, varargin)
  % Configuration parameter
  if nargin < 3
    error('No configuration provided');
  else
    cfg = varargin{1};
  end
  
  %% Resize & preprocess input image
  img_preprocessed = state.imgFunc(imresize(I, state.scale_factor));
  switch cfg.color_space
    case 'rgb'
      img = uint8(255.*img_preprocessed);
    case 'lab'
      img = lab2uint8(applycform(img_preprocessed, state.lab_transform));
    case 'hsv'
      img = uint8(255.*rgb2hsv(img_preprocessed));
    case 'gray'
      img = uint8(255.*rgb2gray(img_preprocessed));
    otherwise
      error('Color space not supported');
  end

  %% Localization
  % Check if we need a refinement, if we explicitly estimate the camera motion, then such a refinement is not needed.
  prev_pos = state.target_pos_history(end,:);
  prev_sz = state.target_sz_history(end,:);
  % if cfg.motion_estimation_history_size > 0 && strcmp(cfg.transform, '')
  %   prev_pos = prev_pos + getMotionPrediction(state.target_pos_history, cfg.motion_estimation_history_size);
  % end
  
  % Previous object location (possibly downscaled)
  target_pos = prev_pos .* state.scale_factor;
  target_sz = prev_sz .* state.scale_factor;
  
  % added code for camera correction
  if strcmp(cfg.transform, 'projective')
    [feat, H] = find_homography(state.feat, img, target_pos([2,1]), target_sz([2,1]));
    state.feat = feat;
  elseif strcmp(cfg.transform, 'affine')
    [feat, H] = find_affine(state.feat, img, target_pos([2,1]), target_sz([2,1]));
    state.feat = feat;
  else
    H = eye(3);
  end
  target_pos = project_t(target_pos, H);
  target_pos = min(target_pos, [size(img, 2), size(img, 1)]);
  target_pos = max(target_pos, [0,0]);
  
  % Search region
  search_sz   = floor(target_sz + cfg.search_win_padding*max(target_sz));
  search_rect = pos2rect(target_pos, search_sz);
  [search_win, padded_search_win] = getSubwindowMasked(img, target_pos, search_sz);
  
  % Apply probability LUT
  pm_search = getForegroundProb(search_win, state.prob_lut, cfg.bin_mapping);
  if cfg.distractor_aware
    pm_search_dist = getForegroundProb(search_win, state.prob_lut_distractor, cfg.bin_mapping);
    pm_search = .5 .* pm_search + .5 .* pm_search_dist;
  end
  pm_search(padded_search_win) = 0;
  
  % Cosine/Hanning window
  cos_win = hann(search_sz(2)) * hann(search_sz(1))';
  
  % Localize
  [hypotheses, vote_scores, dist_scores] = getNMSRects(pm_search, target_sz, cfg.nms_scale, ...
    cfg.nms_overlap, cfg.nms_score_factor, cos_win, cfg.nms_include_center_vote);
  candidate_centers = hypotheses(:,1:2) + hypotheses(:,3:4)./2;
  candidate_scores = vote_scores .* dist_scores;
  [~, best_candidate] = max(candidate_scores);
  
  target_pos = candidate_centers(best_candidate,:);


  if size(hypotheses,1) > 1
    idx = 1:size(hypotheses,1);
    idx(best_candidate) = []; % Remove current object location
    distractors = hypotheses(idx,:);
    target_rect = pos2rect(target_pos, target_sz, [size(pm_search,2) size(pm_search,1)]);
    distractor_overlap = intersectionOverUnion(target_rect, distractors);
  else
    distractors = [];
    distractor_overlap = [];
  end
    
  % Localization visualization
  if cfg.show_figures
    figure(2), clf
    imagesc(pm_search,[0 1]);
    axis image
    title('Search Window')
    
    for i = 1:size(hypotheses,1)
      if i == best_candidate, color = 'r'; else color = 'y'; end
      rectangle('Position',hypotheses(i,:),'EdgeColor',color,'LineWidth',2);
    end
  end
    
  
  %% Appearance update
  % Get current target position within full (possibly downscaled) image coorinates
  target_pos_img = target_pos + search_rect(1:2)-1;
  if cfg.prob_lut_update_rate > 0
    % Extract surrounding region
    surr_sz = floor(cfg.surr_win_factor * target_sz);
    surr_rect = pos2rect(target_pos_img, surr_sz, [size(img,2) size(img,1)]);
    obj_rect_surr = pos2rect(target_pos_img, target_sz, [size(img,2) size(img,1)]) - [surr_rect(1:2)-1, 0, 0];
    surr_win = getSubwindow(img, target_pos_img, surr_sz);
    
    prob_lut_bg = getForegroundBackgroundProbs(surr_win, obj_rect_surr, cfg.num_bins, cfg.bin_mapping);
   
    if cfg.distractor_aware
      % Handle distractors
      if size(distractors,1) > 1
        obj_rect = pos2rect(target_pos, target_sz, [size(search_win,2) size(search_win,1)]);
        prob_lut_dist = getForegroundDistractorProbs(search_win, obj_rect, distractors, cfg.num_bins, cfg.bin_mapping);

        state.prob_lut_distractor = (1-cfg.prob_lut_update_rate).*state.prob_lut_distractor + cfg.prob_lut_update_rate .* prob_lut_dist;
      else
        % If there are no distractors, trigger decay of distractor LUT
        state.prob_lut_distractor = (1-cfg.prob_lut_update_rate).*state.prob_lut_distractor + cfg.prob_lut_update_rate .* prob_lut_bg;
      end

      if (isempty(distractors) || all(distractor_overlap < .1)) % Only update if distractors are not overlapping too much
        state.prob_lut = (1-cfg.prob_lut_update_rate) .* state.prob_lut + cfg.prob_lut_update_rate .* prob_lut_bg;
      end
      
      prob_map = getForegroundProb(surr_win, state.prob_lut, cfg.bin_mapping);
      dist_map = getForegroundProb(surr_win, state.prob_lut_distractor, cfg.bin_mapping);
      prob_map = .5.*prob_map + .5.*dist_map;
    else % No distractor-awareness
      state.prob_lut = (1-cfg.prob_lut_update_rate) .* state.prob_lut + cfg.prob_lut_update_rate .* prob_lut_bg;
      prob_map = getForegroundProb(surr_win, state.prob_lut, cfg.bin_mapping);
    end
    % Update adaptive threshold  
    state.adaptive_threshold = getAdaptiveThreshold(prob_map, obj_rect_surr, cfg);
  end
  
  % Store current location
  target_pos = target_pos + search_rect(1:2)-1;
  target_pos_original = target_pos ./ state.scale_factor;
  target_sz_original = target_sz ./ state.scale_factor;
    
  state.target_pos_history = [state.target_pos_history; target_pos_original];
  state.target_sz_history = [state.target_sz_history; target_sz_original];
  
  % Report current location
  location = pos2rect(state.target_pos_history(end,:), state.target_sz_history(end,:), [size(I,2) size(I,1)]);
  
  if state.report_poly
    location = rect2poly(location);
  end
  
  % Adapt image scale factor
  state.scale_factor = min(1, round(10*cfg.img_scale_target_diagonal/sqrt(sum(target_sz_original.^2)))/10);
end

function pred = getMotionPrediction(values, maxNumFrames)
  if ~exist('maxNumFrames','var')
    maxNumFrames = 5;
  end
  
  if isempty(values)
    pred = [0,0];
  else
    if size(values,1) < 3
      pred = [0,0];
    else
      maxNumFrames = maxNumFrames + 2;
     
      A1 = 0.8;
      A2 = -1;
      V = values(max(1,end-maxNumFrames):end,:);
      P = zeros(size(V,1)-2, size(V,2));
      for i = 3:size(V,1)
        P(i-2,:) = A1 .* (V(i,:) - V(i-2,:)) + A2 .* (V(i-1,:) - V(i-2,:));
      end
      
      pred = mean(P,1);
    end
  end
end

% target_rect Single 1x4 rect
function iou = intersectionOverUnion(target_rect, candidates)
  assert(size(target_rect,1) == 1)
  inA = rectint(candidates,target_rect);
  unA = prod(target_rect(3:4)) + prod(candidates(:,3:4),2) - inA;
  iou = inA ./ max(eps,unA);
end

