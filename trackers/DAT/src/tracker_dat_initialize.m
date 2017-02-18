function [state, location] = tracker_dat_initialize(I, region, varargin)
  if nargin < 3 
    % No tracker configuration provided, use default values
    cfg = default_parameters_datv2();
  else
    assert(isstruct(varargin{1}));
    cfg = varargin{1};
  end
  
  state = struct('show_figures', cfg.show_figures);

 %% Prepare init region (axis-aligned bbox)
  % If the provided region is a polygon ...
  if numel(region) > 4
    state.report_poly = true;
    % Init with an axis aligned bounding box with correct area and center
    cx = mean(region(1:2:end));
    cy = mean(region(2:2:end));
    x1 = min(region(1:2:end));
    x2 = max(region(1:2:end));
    y1 = min(region(2:2:end));
    y2 = max(region(2:2:end));
    A1 = norm(region(1:2) - region(3:4)) * norm(region(3:4) - region(5:6));
    A2 = (x2 - x1) * (y2 - y1);
    s = sqrt(A1/A2);
    w = s * (x2 - x1) + 1;
    h = s * (y2 - y1) + 1;
  else
    state.report_poly = false;
    cx = region(1) + (region(3) - 1)/2;
    cy = region(2) + (region(4) - 1)/2;
    w = region(3);
    h = region(4);
  end
  target_pos = round([cx cy]);
  target_sz = round([w h]);
  
  % Scale data
  state.scale_factor = min(1, round(10*cfg.img_scale_target_diagonal/sqrt(sum(target_sz.^2)))/10);
  target_pos = target_pos .* state.scale_factor;
  target_sz = target_sz .* state.scale_factor;
  
  % Resize/preprocess input image
  state.imgFunc = @(I) im2double(I);
  img = state.imgFunc(imresize(I, state.scale_factor));
  switch cfg.color_space
    case 'rgb'
      img = uint8(255.*img);
    case 'lab'
      state.lab_transform = makecform('srgb2lab');
      img = lab2uint8(applycform(img, state.lab_transform));
    case 'hsv'
      img = uint8(255.*rgb2hsv(img));
    case 'gray'
      img = uint8(255.*rgb2gray(img));
    otherwise
      error('Not supported');
  end

  % specify the low-level features used for camera motion estimation
  feat.detector = cv.FeatureDetector('SURF');
  feat.extractor = cv.DescriptorExtractor('SURF');
  feat.matcher = cv.DescriptorMatcher('FlannBased');
  feat.last_keypoints = feat.detector.detect(img);
  feat.last_descriptors = feat.extractor.compute(img, feat.last_keypoints);
  state.feat = feat;
  
  % Object vs surrounding
  surr_sz = floor(cfg.surr_win_factor * target_sz);
  surr_rect = pos2rect(target_pos, surr_sz, [size(img,2) size(img,1)]);
  obj_rect_surr = pos2rect(target_pos, target_sz, [size(img,2) size(img,1)]) - [surr_rect(1:2)-1, 0, 0];% or replace -1 by extra + [1,1,0,0];
  surr_win = getSubwindow(img, target_pos, surr_sz);
  [state.prob_lut, prob_map] = getForegroundBackgroundProbs(surr_win, obj_rect_surr, cfg.num_bins, cfg.bin_mapping);
  state.prob_lut_distractor = state.prob_lut; % Copy initial discriminative model
  state.prob_lut_masked = state.prob_lut; 
  state.adaptive_threshold = getAdaptiveThreshold(prob_map, obj_rect_surr, cfg);
   
  if state.show_figures
    figure(2), clf
    subplot(121);
    imshow(img);
    rectangle('Position', pos2rect(target_pos, target_sz, [size(img,2) size(img,1)]),'EdgeColor','y','LineWidth',2);
  end
  
  % Store current location
  state.target_pos_history = target_pos ./ state.scale_factor;
  state.target_sz_history = target_sz ./ state.scale_factor;
  
  % Report current location
  location = pos2rect(state.target_pos_history(end,:), state.target_sz_history(end,:), [size(I,2) size(I,1)]);
  
  if state.report_poly
    location = rect2poly(location);
  end
end
