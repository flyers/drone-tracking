function model = OnlineSVMTrain(dataPos, dataNeg, opt, model)

featPos         = dataPos.feat;
featNeg         = dataNeg.feat;
% tmplPos         = dataPos.tmpl;
% tmplNeg         = dataNeg.tmpl;

numPos          = size(featPos, 2);
numNeg          = size(featNeg, 2);
% featDim         = size(featPos, 1);
num             = numPos + numNeg;

feat            = [featPos, featNeg];
tmpl            = [dataPos.tmpl; dataNeg.tmpl];
% feat            = (feat - 128) / 128;
label           = zeros(num, 1);
label(1:numPos) = 1;

config.verbose   = false;
config.svm_thresh = -0.7; % for detecting the tracking failure

%% Model Initialization
if nargin <= 3
    model = CreateSVMTracker();
    fuzzy_weight = ones(size(label));
    model = initSVMTracker(feat',label,fuzzy_weight,model);
else 
%% update svm classifier
    model.temp_count = model.temp_count + 1;
    
    fuzzy_weight = ones(size(label));
    costs = ComputeOverlapCost(tmpl, model.lastOutput);
    fuzzy_weight(~label) = 2*costs(~label)-1;
    model = updateSvmTracker (feat', label, fuzzy_weight, model, config);  
end
end

%% compute cost table 
function costs = ComputeOverlapCost(tmpl, rect)
left    = max(round(tmpl(:,1)),round(rect(1)));
top     = max(round(tmpl(:,2)),round(rect(2)));
right   = min(round(tmpl(:,1)+tmpl(:,3)),round(rect(1)+rect(3)));
bottom  = min(round(tmpl(:,2)+tmpl(:,4)),round(rect(2)+rect(4)));
ovlp    = max(right - left,0).*max(bottom - top, 0);
costs   = 1 - ovlp./(2*rect(3)*rect(4)-ovlp);
end

%%
function tracker = CreateSVMTracker()

tracker = [];
tracker.sv_size = 100;% maxial 100 cvs
tracker.C = 100;
tracker.B = 80;% for tvm
tracker.B_p = 10;% for positive sv
tracker.lambda = 1;% for whitening
tracker.m1 = 1;% for tvm
tracker.m2 = 2;% for tvm
tracker.w = [];
tracker.w_smooth_rate = 0.0;
tracker.confidence = 1;
tracker.state = 0;
tracker.temp_count = 0;
tracker.output_feat_record = [];
tracker.feat_cache = [];
% tracker.experts = {};
tracker.confidence_exp = 1;
tracker.confidence = 1;
tracker.best_expert_idx = 1;
tracker.failure = false;
tracker.update_count = 0;

end

%%
function [svm_tracker] = initSVMTracker(sample,label,fuzzy_weight,svm_tracker)

% sample_w = fuzzy_weight;
%        
% pos_mask = label>0.5;
% neg_mask = ~pos_mask;
% s1 = sum(sample_w(pos_mask));
% s2 = sum(sample_w(neg_mask));
%         
% sample_w(pos_mask) = sample_w(pos_mask)*s2;
% sample_w(neg_mask) = sample_w(neg_mask)*s1;
%         
% C = max(svm_tracker.C*sample_w/sum(sample_w),0.001);
C = svm_tracker.C;

svm_tracker.clsf = svmtrain( sample, label,'boxconstraint',C,'autoscale','false');
        
svm_tracker.clsf.w = svm_tracker.clsf.Alpha'*svm_tracker.clsf.SupportVectors;
svm_tracker.w = svm_tracker.clsf.w;
svm_tracker.Bias = svm_tracker.clsf.Bias;
svm_tracker.sv_label = label(svm_tracker.clsf.SupportVectorIndices,:);
svm_tracker.sv_full = sample(svm_tracker.clsf.SupportVectorIndices,:);
        
svm_tracker.pos_sv = svm_tracker.sv_full(svm_tracker.sv_label>0.5,:);
svm_tracker.pos_w = ones(size(svm_tracker.pos_sv,1),1);
svm_tracker.neg_sv = svm_tracker.sv_full(svm_tracker.sv_label<0.5,:);
svm_tracker.neg_w = ones(size(svm_tracker.neg_sv,1),1);
        
% compute real margin
pos2plane = -svm_tracker.pos_sv*svm_tracker.w';
neg2plane = -svm_tracker.neg_sv*svm_tracker.w';
svm_tracker.margin = (min(pos2plane) - max(neg2plane))/norm(svm_tracker.w);
        
% calculate distance matrix
if size(svm_tracker.pos_sv,1)>1
    svm_tracker.pos_dis = squareform(pdist(svm_tracker.pos_sv));
else
    svm_tracker.pos_dis = inf;
end
svm_tracker.neg_dis = squareform(pdist(svm_tracker.neg_sv)); 
        
% %% intialize tracker experts
% experts{1}.w = svm_tracker.w;
% experts{1}.Bias = svm_tracker.Bias;
% experts{1}.score = [];
% experts{1}.snapshot = svm_tracker;
%         
% experts{2} = experts{1};

end

%%
function [svm_tracker] = updateSvmTracker(sample,label,fuzzy_weight,svm_tracker,config)

sample = [svm_tracker.pos_sv;svm_tracker.neg_sv; sample];
label = [ones(size(svm_tracker.pos_sv,1),1);zeros(size(svm_tracker.neg_sv,1),1);label];% positive:1 negative:0
% sample_w = [svm_tracker.pos_w;svm_tracker.neg_w;fuzzy_weight];
       
% pos_mask = label>0.5;
% neg_mask = ~pos_mask;
% s1 = sum(sample_w(pos_mask));
% s2 = sum(sample_w(neg_mask));
%         
% sample_w(pos_mask) = sample_w(pos_mask)*s2;
% sample_w(neg_mask) = sample_w(neg_mask)*s1;
        
% C = max(svm_tracker.C*sample_w/sum(sample_w),0.001);
C = svm_tracker.C;
if config.verbose
    svm_tracker.clsf = svmtrain( sample, label, ...%'kernel_function',@kfun,'kfunargs',{svm_tracker.struct_mat},...
       'boxconstraint',C,'autoscale','false','options',statset('Display','final','MaxIter',5000));
    fprintf('feat_d: %d; train_num: %d; sv_num: %d \n',size(sample,2),size(sample,1),size(svm_tracker.clsf.Alpha,1)); 
else
    svm_tracker.clsf = svmtrain( sample, label, ...%'kernel_function',@kfun,'kfunargs',{svm_tracker.struct_mat},...
       'boxconstraint',C,'autoscale','false','options',statset('MaxIter',5000));
end
%**************************
svm_tracker.w = svm_tracker.clsf.Alpha'*svm_tracker.clsf.SupportVectors;
svm_tracker.Bias = svm_tracker.clsf.Bias;
svm_tracker.clsf.w = svm_tracker.w;
% get the idx of new svs
sv_idx = svm_tracker.clsf.SupportVectorIndices;
sv_old_sz = size(svm_tracker.pos_sv,1)+size(svm_tracker.neg_sv,1);
sv_new_idx = sv_idx(sv_idx>sv_old_sz);
sv_new = sample(sv_new_idx,:);
sv_new_label = label(sv_new_idx,:);
        
num_sv_pos_new = sum(sv_new_label);
        
% update pos_dis, pos_w and pos_sv
pos_sv_new = sv_new(sv_new_label>0.5,:);
if ~isempty(pos_sv_new)
    if size(pos_sv_new,1)>1
        pos_dis_new = squareform(pdist(pos_sv_new));
    else
        pos_dis_new = 0;
    end
    pos_dis_cro = pdist2(svm_tracker.pos_sv,pos_sv_new);
    svm_tracker.pos_dis = [svm_tracker.pos_dis, pos_dis_cro; pos_dis_cro', pos_dis_new];
    svm_tracker.pos_sv = [svm_tracker.pos_sv;pos_sv_new];
    svm_tracker.pos_w = [svm_tracker.pos_w;ones(num_sv_pos_new,1)];
end
        
% update neg_dis, neg_w and neg_sv
neg_sv_new = sv_new(sv_new_label<0.5,:);
if ~isempty(neg_sv_new)
    if size(neg_sv_new,1)>1
        neg_dis_new = squareform(pdist(neg_sv_new));
    else
        neg_dis_new = 0;
    end
    neg_dis_cro = pdist2(svm_tracker.neg_sv,neg_sv_new);
    svm_tracker.neg_dis = [svm_tracker.neg_dis, neg_dis_cro; neg_dis_cro', neg_dis_new];
    svm_tracker.neg_sv = [svm_tracker.neg_sv;neg_sv_new];
    svm_tracker.neg_w = [svm_tracker.neg_w;ones(size(sv_new,1)-num_sv_pos_new,1)];
end
        
svm_tracker.pos_dis = svm_tracker.pos_dis + diag(inf*ones(size(svm_tracker.pos_dis,1),1));
svm_tracker.neg_dis = svm_tracker.neg_dis + diag(inf*ones(size(svm_tracker.neg_dis,1),1));
        
        
% compute real margin
pos2plane = -svm_tracker.pos_sv*svm_tracker.w';
neg2plane = -svm_tracker.neg_sv*svm_tracker.w';
svm_tracker.margin = (min(pos2plane) - max(neg2plane))/norm(svm_tracker.w);
        
% shrink svs
% check if to remove
if size(svm_tracker.pos_sv,1)+size(svm_tracker.neg_sv,1)>svm_tracker.B
    pos_score_sv = -(svm_tracker.pos_sv*svm_tracker.w'+svm_tracker.Bias);
    neg_score_sv = -(svm_tracker.neg_sv*svm_tracker.w'+svm_tracker.Bias);
    m_pos = abs(pos_score_sv) < svm_tracker.m2;
    m_neg = abs(neg_score_sv) < svm_tracker.m2;
            
    if config.verbose
        fprintf('remove svs: pos %d, neg %d \n',sum(~m_pos),sum(~m_neg));
    end
    if sum(m_pos) > 0
        svm_tracker.pos_sv = svm_tracker.pos_sv(m_pos,:);
        svm_tracker.pos_w = svm_tracker.pos_w(m_pos,:);
        svm_tracker.pos_dis = svm_tracker.pos_dis(m_pos,m_pos);
    end

    if sum(m_neg)>0
        svm_tracker.neg_sv = svm_tracker.neg_sv(m_neg,:);
        svm_tracker.neg_w = svm_tracker.neg_w(m_neg,:);
        svm_tracker.neg_dis = svm_tracker.neg_dis(m_neg,m_neg);
    end
end
        
% check if to merge
while size(svm_tracker.pos_sv,1)+size(svm_tracker.neg_sv,1)>svm_tracker.B
    [mm_pos,idx_pos] = min(svm_tracker.pos_dis(:));
    [mm_neg,idx_neg] = min(svm_tracker.neg_dis(:));
            
    if mm_pos > mm_neg || size(svm_tracker.pos_sv,1) <= svm_tracker.B_p% merge negative samples
        if config.verbose
            fprintf('merge negative samples: %d \n', size(svm_tracker.neg_w,1))
        end
                
        [i,j] = ind2sub(size(svm_tracker.neg_dis),idx_neg);
        w_i= svm_tracker.neg_w(i);
        w_j= svm_tracker.neg_w(j);
        merge_sample = (w_i*svm_tracker.neg_sv(i,:)+w_j*svm_tracker.neg_sv(j,:))/(w_i+w_j);                
                
        svm_tracker.neg_sv([i,j],:) = []; svm_tracker.neg_sv(end+1,:) = merge_sample;
        svm_tracker.neg_w([i,j]) = []; svm_tracker.neg_w(end+1,1) = w_i + w_j;
                
        svm_tracker.neg_dis([i,j],:)=[]; svm_tracker.neg_dis(:,[i,j])=[];
        neg_dis_cro = pdist2(svm_tracker.neg_sv(1:end-1,:),merge_sample);
        svm_tracker.neg_dis = [svm_tracker.neg_dis, neg_dis_cro;neg_dis_cro',inf];                
    else
        if config.verbose
            fprintf('merge positive samples: %d \n', size(svm_tracker.pos_w,1))
        end

        [i,j] = ind2sub(size(svm_tracker.pos_dis),idx_pos);
        w_i= svm_tracker.pos_w(i);
        w_j= svm_tracker.pos_w(j);
        merge_sample = (w_i*svm_tracker.pos_sv(i,:)+w_j*svm_tracker.pos_sv(j,:))/(w_i+w_j);                

        svm_tracker.pos_sv([i,j],:) = []; svm_tracker.pos_sv(end+1,:) = merge_sample;
        svm_tracker.pos_w([i,j]) = []; svm_tracker.pos_w(end+1,1) = w_i + w_j;
                
        svm_tracker.pos_dis([i,j],:)=[]; svm_tracker.pos_dis(:,[i,j])=[];
        pos_dis_cro = pdist2(svm_tracker.pos_sv(1:end-1,:),merge_sample);
        svm_tracker.pos_dis = [svm_tracker.pos_dis, pos_dis_cro;pos_dis_cro',inf]; 
                
                
    end
            
end
        
% % update experts
% experts{end}.w = svm_tracker.w;
% experts{end}.Bias = svm_tracker.Bias;
%         
% svm_tracker.update_count = svm_tracker.update_count + 1;
end
