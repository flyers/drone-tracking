function results = run_HOGLR_RANSAC(seq, res_path, bSaveImage)

    project_dir = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    addpath(genpath(fullfile(project_dir, 'trackers', 'HOG_LR')));
    addpath(genpath(fullfile(project_dir, 'trackers', 'HOG_LR', 'FeatureExtractor')));
    addpath(genpath(fullfile(project_dir, 'trackers', 'HOG_LR', 'MotionModel')));
    addpath(genpath(fullfile(project_dir, 'trackers', 'HOG_LR', 'ObservationModel')));
    addpath(genpath(fullfile(project_dir, 'trackers', 'HOG_LR', 'sampler')));
    addpath(genpath(fullfile(project_dir, 'trackers', 'HOG_LR', 'UpdateJudger')));
    addpath(genpath(fullfile(project_dir, 'trackers', 'HOG_LR', 'Utils')));

    config_RANSAC;
    rng(0);
    seq.opt = opt;
    rect=seq.init_rect;
    p = [rect(1)+rect(3)/2, rect(2)+rect(4)/2, rect(3), rect(4)]; % Center x, Center y, height, width
    frame = imread(seq.s_frames{1});
    if size(frame,3)==1
        frame = repmat(frame,[1,1,3]);
    end
    frame = rgb2gray(frame);
    if (seq.opt.useNormalSize)
        scaleHeight = size(frame, 1) / seq.opt.normalHeight;
        scaleWidth = size(frame, 2) / seq.opt.normalWidth;
        p(1) = p(1) / scaleWidth;
        p(3) = p(3) / scaleWidth;
        p(2) = p(2) / scaleHeight;
        p(4) = p(4) / scaleHeight;
    end
    
    duration = 0;
    tic;
    reportRes = [];

    %% compute feature points
    cvfeat.detector = cv.FeatureDetector('SURF');
    cvfeat.extractor = cv.DescriptorExtractor('SURF');
    cvfeat.matcher = cv.DescriptorMatcher('FlannBased');

    for f = 1:length(seq.s_frames)
        
        frame = imread(seq.s_frames{f});
        if size(frame,3)==1
            frame = repmat(frame,[1,1,3]);
        end

        if (seq.opt.useNormalSize)
%           frame = imresize(frame, [seq.opt.normalHeight, seq.opt.normalWidth]);
            frame = mexResize(frame, [seq.opt.normalHeight, seq.opt.normalWidth], 'auto');
        end

        %% added code for camera motion estimation
        % resize_im = mexResize(frame, [opt.feature.normalHeight, opt.feature.normalWidth]);
        if f ~= 1
            s_pos = [p(2), p(1)];
            s_size = [p(4), p(3)];
            [cvfeat, H] = find_homography(cvfeat, frame, s_pos, s_size);
        else
            cvfeat.last_keypoints = cvfeat.detector.detect(frame);
            cvfeat.last_descriptors = cvfeat.extractor.compute(frame, cvfeat.last_keypoints);
        end

        frame = im2double(frame);
        
        if (f ~= 1)
            % first apply the H transformation
            tmpl(:, 1:2) = project_t(tmpl(:, 1:2), H);
            % in case the transformed bounding box get out the image
            % boundary, manually bound them by the size of the bounding box
            tmpl(:, 1) = max(tmpl(:, 1), -tmpl(:, 3)/2);
            tmpl(:, 2) = max(tmpl(:, 2), -tmpl(:, 4)/2);
            tmpl(:, 1) = min(tmpl(:, 1), size(frame, 2)+tmpl(:, 3)/2);
            tmpl(:, 2) = min(tmpl(:, 2), size(frame, 1)+tmpl(:, 4)/2);
            tmpl    = globalParam.MotionModel(tmpl, prob, seq.opt);
            [feat, seq.opt] = globalParam.FeatureExtractor(frame, tmpl, seq.opt);
            prob    = globalParam.ObservationModelTest(feat, model);    
            
            [maxProb, maxIdx] = max(prob); 
            p = tmpl(maxIdx, :);
            model.lastOutput = p;
            model.lastProb = maxProb;
        else
            tmpl = globalParam.MotionModel(p, 1, seq.opt);
            prob = ones(1, size(tmpl, 1));
        end     
        
        if (f == 1)
            tmplPos = globalParam.PosSampler(p, seq.opt);
            tmplNeg = globalParam.NegSampler(p, seq.opt);
            [dataPos, seq.opt] = globalParam.FeatureExtractor(frame, tmplPos, seq.opt);
            [dataNeg, seq.opt] = globalParam.FeatureExtractor(frame, tmplNeg, seq.opt);
            model   = globalParam.ObservationModelTrain(dataPos, dataNeg, seq.opt);  
            if (seq.opt.useFirstFrame)
                assert(~strcmp(func2str(globalParam.ObservationModelTrain), 'SOSVMTrain'), ...
                    'SOSVM does not support useFirstFrame option!!');
                dataPosFirstFrame = dataPos;
            end
        else
            if (globalParam.ConfidenceJudger(model, seq.opt))
                tmplPos = globalParam.PosSampler(p, seq.opt);
                tmplNeg = globalParam.NegSampler(p, seq.opt);
                [dataPos, seq.opt] = globalParam.FeatureExtractor(frame, tmplPos, seq.opt);
                [dataNeg, seq.opt] = globalParam.FeatureExtractor(frame, tmplNeg, seq.opt);
%                 disp(f);
                if (seq.opt.useFirstFrame)
                    dataPos.feat = [dataPosFirstFrame.feat, dataPos.feat];
                    dataPos.tmpl = [zeros(size(dataPosFirstFrame.tmpl)); dataPos.tmpl];
                end
                model   = globalParam.ObservationModelTrain(dataPos, dataNeg, seq.opt, model);  
            end
        end
%       
        if opt.visualization
            figure(1),imagesc(frame);
    %         pause(0.1);
            imshow(frame); 
            rectangle('position', [p(1) - p(3) / 2, p(2) - p(4) / 2, p(3), p(4)], ...
                'EdgeColor','r', 'LineWidth',2);
            drawnow;
        end
        if (seq.opt.useNormalSize)
            p(1) = p(1) * scaleWidth;
            p(3) = p(3) * scaleWidth;
            p(2) = p(2) * scaleHeight;
            p(4) = p(4) * scaleHeight;
        end
        rect = [p(1) - p(3) / 2, p(2) - p(4) / 2, p(3), p(4)];
        reportRes = [reportRes; round(rect)];
        fprintf('frame %d\n', f);
    end
    
    duration = duration + toc;
    fprintf('%d frames took %.3f seconds : %.3fps\n',f,duration,f/duration);
    results.res=reportRes;
    results.type='rect';
    results.fps = f/duration;
    
end
