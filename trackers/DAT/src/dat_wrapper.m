function dat_wrapper()
% VOT integration wrapper
% *************************************************************
% VOT: Always call exit command at the end to terminate Matlab!
% *************************************************************
cleanup = onCleanup(@() exit() );

% *************************************************************
% VOT: Set random seed to a different value every time.
% *************************************************************
RandStream.setGlobalStream(RandStream('mt19937ar', 'Seed', sum(clock)));

% **********************************
% VOT: Get initialization data
% **********************************
[handle, image, region] = vot('polygon');

% Initialize the tracker
cfg = default_parameters_dat();
[state, ~] = tracker_dat_initialize(imread(image), region, cfg);

while true
  % VOT: Get next frame
  [handle, image] = handle.frame(handle);

  if isempty(image)
      break;
  end;

  % Perform a tracking step, obtain new region
  [state, region] = tracker_dat_update(state, imread(image), cfg);

  % VOT: Report position for frame
  handle = handle.report(handle, region);
end;

% VOT: Output the results
handle.quit(handle);
end

