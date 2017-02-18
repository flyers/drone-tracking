function [ results ] = run_MEEM_RANSAC( seq, res_path, bSaveImage )


project_dir = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(genpath(fullfile(project_dir, 'trackers', 'MEEM')));
addpath(genpath(fullfile(project_dir, 'trackers', 'MEEM', 'base_tracker')));
addpath(genpath(fullfile(project_dir, 'trackers', 'MEEM', 'expert_ensemble')));
addpath(genpath(fullfile(project_dir, 'trackers', 'MEEM', 'mex')));
addpath(genpath(fullfile(project_dir, 'trackers', 'MEEM', 'utils')));

visualization = 0;
results = MEEMTrack_ransac(seq.path, seq.ext, visualization, seq.init_rect, seq.startFrame, seq.endFrame, seq);

end

