function [top_rects, top_vote_scores, top_dist_scores] = getNMSRects(prob_map, obj_sz, scale, overlap, score_frac, dist_map, include_inner)
%GETNMSRECTS Perform NMS on given probability map
% Parameters:
%   prob_map      Object likelihood within search region
%   obj_sz        Currently estimated object size
%   scale         Optionally scale hypotheses (e.g. search smaller rects)
%   overlap       Overlap percentage of hypotheses
%   score_frac    return all boxes with score >= score_frac * highest-score
%   dist_map      Distance prior (e.g. cosine/hanning window)
%   include_inner Add extra inner rect scores to favor hypotheses with 
%                 highly confident center regions
% 
% Returns:
% top_rects       Rectangles
% top_vote_scores Scores based on the likelihood map
% top_dist_scores Scores based on the distance prior
  [height, width, l] = size(prob_map);
  
  if ~exist('scale', 'var'), scale = 1; end
  if ~exist('overlap','var'), overlap = .5; end
  if ~exist('score_frac','var'), score_frac = .25; end
  if ~exist('dist_map','var'), dist_map = ones(height,width); end

  rect_sz = floor(obj_sz .* scale);
  if include_inner
    o_x = round(max([1, rect_sz(1)*0.2]));
    o_y = round(max([1, rect_sz(2)*0.2]));
  end 
  
  stepx = max([1,round(rect_sz(1) .* (1-overlap))]);
  stepy = max([1, round(rect_sz(2) .* (1-overlap))]);

  posx = 1:stepx:width-rect_sz(1);
  posy = 1:stepy:height-rect_sz(2);

  [x,y] = meshgrid(posx, posy);
  r = x(:) + rect_sz(1);
  b = y(:) + rect_sz(2);
  r(r > width) = width;
  b(b > height) = height;

  boxes = [x(:), y(:), r-x(:), b-y(:)];
  if include_inner
    boxes_inner = [x(:)+o_x, y(:)+o_y, (r-2*o_x)-x(:), (b-2*o_y)-y(:)];
  end

  % Linear indices
  l = boxes(:,1); t = boxes(:,2);
  h = height+1;
  w = width+1;
  bl = sub2ind([h w],b,l);
  br = sub2ind([h w],b,r);
  tl = sub2ind([h w],t,l);
  tr = sub2ind([h w],t,r);

  if include_inner
    rect_sz_inner = rect_sz - 2.*[o_x,o_y];%[r - l - 2*o_x, b-t-2*o_y];
    bl_inner = sub2ind([h w],b-o_y,l+o_x);
    br_inner = sub2ind([h w],b-o_y,r-o_x);
    tl_inner = sub2ind([h w],t+o_y,l+o_x);
    tr_inner = sub2ind([h w],t+o_y,r-o_x);
  end

  intProbMap = integralImage(prob_map);
  intDistMap = integralImage(dist_map);
  v_scores = intProbMap(br) - intProbMap(bl) - intProbMap(tr) + intProbMap(tl);
  d_scores = intDistMap(br) - intDistMap(bl) - intDistMap(tr) + intDistMap(tl);
  if include_inner
    scores_inner = intProbMap(br_inner) - intProbMap(bl_inner) - intProbMap(tr_inner) + intProbMap(tl_inner);
    v_scores = v_scores ./ prod(rect_sz) + scores_inner ./ prod(rect_sz_inner);
  end

  top_rects = [];
  top_vote_scores = [];
  top_dist_scores = [];
  [ms, midx] = max(v_scores);
  best_score = ms;

  while ms > score_frac*best_score
    prob_map(boxes(midx,2):boxes(midx,2)+boxes(midx,4), boxes(midx,1):boxes(midx,1)+boxes(midx,3)) = 0;
    top_rects = [top_rects; boxes(midx,:)];
    top_vote_scores = [top_vote_scores; v_scores(midx)];
    top_dist_scores = [top_dist_scores; d_scores(midx)];
    boxes(midx,:) = [];
    if include_inner
      boxes_inner(midx,:) = [];
    end

    bl(midx) = []; br(midx) = [];
    tl(midx) = []; tr(midx) = [];
    if include_inner
      bl_inner(midx) = []; br_inner(midx) = [];
      tl_inner(midx) = []; tr_inner(midx) = [];
    end

    intProbMap = integralImage(prob_map);
    intDistMap = integralImage(dist_map);
    v_scores = intProbMap(br) - intProbMap(bl) - intProbMap(tr) + intProbMap(tl);
    d_scores = intDistMap(br) - intDistMap(bl) - intDistMap(tr) + intDistMap(tl);
    if include_inner
      scores_inner = intProbMap(br_inner) - intProbMap(bl_inner) - intProbMap(tr_inner) + intProbMap(tl_inner);
      v_scores = v_scores ./ prod(rect_sz) + scores_inner ./ prod(rect_sz_inner);
    end

    [ms, midx] = max(v_scores);
  end
end

function [Integral] = integralImage(probMap)
  outputSize = size(probMap) + 1;
  Integral = zeros(outputSize);
  Integral(2:end, 2:end) = cumsum(cumsum(double(probMap),1),2);
end

