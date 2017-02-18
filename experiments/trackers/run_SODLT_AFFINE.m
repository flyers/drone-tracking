%% Copyright (C) Naiyan Wang, Siyi Li, Abhinav Gupta and Dit-Yan Yeung.
%% Transferring Rich Feature Hierarchies for Robust Visual Tracking
%% All rights reserved.

function results=run_SODLT_AFFINE(seq, res_path, bSaveImage)
    project_dir = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    addpath(fullfile(project_dir, 'trackers', 'SODLT'));
    addpath(fullfile(project_dir, 'trackers', 'SODLT', 'mex'));
    % add the caffe toolbox path
    addpath(fullfile(project_dir, 'trackers', 'SODLT', 'external', 'caffe', 'matlab', 'caffe'));
    caffe_proto_path = fullfile(project_dir, 'trackers', 'SODLT', 'external', 'caffe', 'examples', 'objectness', 'imagenet_deploy_solver.prototxt');
    caffe_model_path = fullfile(project_dir, 'trackers', 'SODLT', 'external', 'caffe_objectness_train_iter_100000');

    rng('default');
    warning off;

    if isfield(seq, 'opt')
        opt = seq.opt;
    else
        trackparam_SODLT;
    end
    rect = seq.init_rect;
    p = [rect(1)+rect(3)/2, rect(2)+rect(4)/2, rect(3), rect(4)];
    frame = imread(seq.s_frames{1});
    if size(frame,3)==1
        frame = repmat(frame,[1,1,3]);
    end
    
    % normalize the image, if any
    if opt.use_normalize == 1
        scale_height = size(frame, 1) / opt.normal_height;
        scale_width = size(frame, 2) / opt.normal_width;
        p(1) = p(1) / scale_width;
        p(3) = p(3) / scale_width;
        p(2) = p(2) / scale_height;
        p(4) = p(4) / scale_height;
        frame = imresize(frame, [opt.normal_height, opt.normal_width]);
    else
        scale_height = 1;
        scale_width = 1;
    end

    %% specify the low-level features used for camera motion estimation
    feat.detector = cv.FeatureDetector('SURF');
    feat.extractor = cv.DescriptorExtractor('SURF');
    feat.matcher = cv.DescriptorMatcher('FlannBased');
    feat.last_keypoints = feat.detector.detect(frame);
    feat.last_descriptors = feat.extractor.compute(frame, feat.last_keypoints);

    frame = double(frame);    
    pos_num = 8;
    fixed_neg_num = 32;
    
    % Sample negative templates for finetuning CNN
    tmpl.basis(:, pos_num + 1 : pos_num + fixed_neg_num) = sample_neg(frame, p, 100, 0);
    
    param_fusion.est = p;
    % maintain two CNN nets with different update strategies
    % both the two nets are finetuned in the first frame
    caffe('set_net_id', 0);
    caffe('init_finetune', caffe_proto_path, caffe_model_path);
    for i = 1 : 4
        % Sample positive templates
        [tmpl.basis(:, 1:pos_num), tmpl.cord(1:pos_num, :)] = sample_pos(frame, p, 100, 1,4);
        update_cnn(tmpl, pos_num + fixed_neg_num, fixed_neg_num, 5);
    end
    caffe('set_net_id', 1);
    caffe('init_finetune', caffe_proto_path, caffe_model_path);
    for i = 1 : 4
        % Sample positive templates
        [tmpl.basis(:, 1:pos_num), tmpl.cord(1:pos_num, :)] = sample_pos(frame, p, 100, 1,4);
        update_cnn(tmpl, pos_num + fixed_neg_num, fixed_neg_num, 5);
    end

    % conserve the positive sample in the first frame, always use them to as the finetuning data afterwards
    fixed_pos.basis(:, 1:pos_num) = tmpl.basis(:, 1:pos_num);
    fixed_pos.cord(1:pos_num, :) = tmpl.cord( 1:pos_num, :);

    % track the sequence from frame 1 onward
    duration = 0; tic;
    report_res = [];
    for f = 1:size(seq.s_frames, 1)  
        frame = imread(seq.s_frames{f});
        if size(frame,3)==1
            frame = repmat(frame, [1,1,3]);
        end
        if opt.use_normalize == 1
            frame = imresize(frame, [opt.normal_height, opt.normal_width]);
        end
        if f > 1
            [feat, H] = find_affine(feat, frame, param_fusion.est([2,1]), param_fusion.est([4,3]));
            param_fusion.est(1:2) = project_t(param_fusion.est(1:2), H);
            param_fusion.est(1:2) = min(param_fusion.est(1:2), [size(frame, 2), size(frame, 1)]);
            param_fusion.est(1:2) = max(param_fusion.est(1:2), [0,0]);
        end
        frame = double(frame);
        % test with the cropped region with 4 different scales
        num_scale = 4;
        [tmpl.basis] = sample_pos(frame, param_fusion.est, 100, 0, num_scale);
        % also crop the negative samples
        tmpl.basis(:, num_scale + 1 : num_scale + fixed_neg_num) = sample_neg(frame, param_fusion.est, 100, 0);

        % estimate the bounding box using the two networks
        caffe('set_net_id', 0);
        param_cnn_long = bbox_search_cnn(frame, param_fusion, tmpl, 100);
        caffe('set_net_id', 1);
        param_cnn_short = bbox_search_cnn(frame, param_fusion, tmpl, 100);

        if param_cnn_long.conf > param_cnn_short.conf
            param_fusion = param_cnn_long;
            disp('using long term net');
        else
            param_fusion = param_cnn_short;
            disp('using short term net');
        end
        fprintf('param_cnn_long conf \t%f\nparam_cnn_short conf \t%f\t %d\n', param_cnn_long.conf, param_cnn_short.conf, f);

        %% update the long term net
        if param_cnn_long.conf > 0.8 && size(param_cnn_long.neg_sample, 2) > 0
            caffe('set_net_id', 0);
            fprintf('updating long term net\n');
            clear tmpl;
            [tmpl.basis, tmpl.cord] = sample_pos(frame, param_cnn_long.est, 100, 1, 4);
            tmpl.basis = [tmpl.basis fixed_pos.basis];
            tmpl.cord = [tmpl.cord; fixed_pos.cord];
            start = pos_num * 2;
            %         start = pos_num;
            neg_num = size(param_cnn_long.neg_sample, 2);
            tmpl.basis(:, start + 1 : start + neg_num) = param_cnn_long.neg_sample;
            update_cnn(tmpl, start + neg_num, neg_num, 1);
        end

        p = param_fusion.est;
        %% update the short term net
        if  size(param_cnn_short.neg_sample, 2) > 0
            caffe('set_net_id', 1);
            fprintf('updating short term net\n');
            clear tmpl;
            if param_fusion.conf > 1e-4
                [tmpl.basis, tmpl.cord] = sample_pos(frame, p, 100, 1, 4);
                tmpl.basis = [tmpl.basis fixed_pos.basis];
                tmpl.cord = [tmpl.cord; fixed_pos.cord];
                start = pos_num * 2;
            else 
                start = 0;
            end
            neg_num = size(param_cnn_short.neg_sample, 2);
            tmpl.basis(:, start + 1 : start + neg_num) = param_cnn_short.neg_sample;
            update_cnn(tmpl, start + neg_num, neg_num, 1);
        end

        % draw the tracking result
        if opt.show_img
            rect_position = [p(1) - p(3) / 2, p(2) - p(4) / 2, p(3), p(4)];
            if f == 1  %first frame, create GUI
                figure( 'Name',['SODLT']);
                im_handle = imshow(uint8(frame), 'Border','tight', 'InitialMag', 100 + 100 * (length(frame) < 500));
                rect_handle = rectangle('Position',rect_position, 'EdgeColor','r', 'LineWidth', 2);
                text_handle = text(10, 10, int2str(f));
                set(text_handle, 'color', [0 1 1]);
            else
                try  %subsequent frames, update GUI
                    set(im_handle, 'CData', uint8(frame));
                    set(rect_handle, 'Position', rect_position)
                    set(text_handle, 'string', int2str(f));
                catch
                    return
                end
            end
            
            drawnow;
        end

        p(1) = p(1) * scale_width;
        p(3) = p(3) * scale_width;
        p(2) = p(2) * scale_height;
        p(4) = p(4) * scale_height;
        rect = [p(1) - p(3) / 2, p(2) - p(4) / 2, p(3), p(4)];
        report_res = [report_res; round(rect)];

    end
    duration = duration + toc;
    fprintf('%d frames took %.3f seconds : %.3fps\n',f,duration,f/duration);
    results.res=report_res;
    results.type='rect';
    results.fps = f/duration;
end
