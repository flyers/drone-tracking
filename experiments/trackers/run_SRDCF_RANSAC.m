% This function runs the SRDCF tracker on the video specified in "seq".
% It can be integrated directly in the Online Tracking Benchmark (OTB).
% The parameters are set as in the ICCV 2015 paper.

function results=run_SRDCF_RANSAC(seq, res_path, bSaveImage, parameters)


project_dir = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(project_dir, 'trackers', 'SRDCF'));

% Default parameters used in the ICCV 2015 paper

% HOG feature parameters
hog_params.nDim = 31;

% Grayscale feature parameters
grayscale_params.colorspace='gray';
grayscale_params.nDim = 1;

% Global feature parameters 
params.t_features = {
    ...struct('getFeature',@get_colorspace, 'fparams',grayscale_params),...  % Grayscale is not used as default
    struct('getFeature',@get_fhog,'fparams',hog_params),...
};
params.t_global.cell_size = 4;                  % Feature cell size
params.t_global.cell_selection_thresh = 0.75^2; % Threshold for reducing the cell size in low-resolution cases

% Filter parameters
params.search_area_shape = 'square';    % the shape of the training/detection window: 'proportional', 'square' or 'fix_padding'
params.search_area_scale = 4.0;         % the size of the training/detection area proportional to the target size
params.filter_max_area = 50^2;          % the size of the training/detection area in feature grid cells

% Learning parameters
params.learning_rate = 0.025;			% learning rate
params.output_sigma_factor = 1/16;		% standard deviation of the desired correlation output (proportional to target)
params.init_strategy = 'indep';         % strategy for initializing the filter: 'const_reg' or 'indep'
params.num_GS_iter = 4;                 % number of Gauss-Seidel iterations in the learning

% Detection parameters
params.refinement_iterations = 1;       % number of iterations used to refine the resulting position in a frame
params.interpolate_response = 4;        % correlation score interpolation strategy: 0 - off, 1 - feature grid, 2 - pixel grid, 4 - Newton's method
params.newton_iterations = 5;           % number of Newton's iteration to maximize the detection scores

% Regularization window parameters
params.use_reg_window = 1;              % wather to use windowed regularization or not
params.reg_window_min = 0.1;			% the minimum value of the regularization window
params.reg_window_edge = 3.0;           % the impact of the spatial regularization (value at the target border), depends on the detection size and the feature dimensionality
params.reg_window_power = 2;            % the degree of the polynomial to use (e.g. 2 is a quadratic window)
params.reg_sparsity_threshold = 0.05;   % a relative threshold of which DFT coefficients that should be set to zero
params.lambda = 1e-2;					% the weight of the standard (uniform) regularization, only used when params.use_reg_window == 0

% Scale parameters
params.number_of_scales = 7;
params.scale_step = 1.01;

% Debug and visualization
params.visualization = 0;
params.debug = 0;

% Camera estimation
params.transform = 'projective';
params.use_normal_size = 1;
params.normal_height = 360;
params.normal_width = 640;

params.wsize = [seq.init_rect(1,4), seq.init_rect(1,3)];
params.init_pos = [seq.init_rect(1,2), seq.init_rect(1,1)] + floor(params.wsize/2);
params.s_frames = seq.s_frames;

results = SRDCF_tracker(params);
