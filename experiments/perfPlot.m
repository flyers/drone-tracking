clear
close all;
clc

cur_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(cur_dir, 'util'));

drawOverall = false;
drawAtt = true;

% The folder that contains the annotation files for sequence attributes
attPath = fullfile(cur_dir, 'anno', 'att');

attName={'scale variation' 'aspect ratio variation' 'occlusion'	'deformation' 'fast camera motion'	'in-plane rotation' 'out-of-plane rotation'  'out of view'	'background clutter' 'Similar Objects Around'};

attFigName={'scale_variation' 'aspect_ratio_variation' 'occlusion'	'deformation' 'fast_camera_motion'	'in-plane_rotation' 'out-of-plane_rotation'  'out_of_view'	'background_clutter' 'Similar_Objects_Around'};


plotDrawStyleAll={   struct('color',[1,0,0],'lineStyle','-'),...
    struct('color',[0,1,0],'lineStyle','-'),...
    struct('color',[0,0,1],'lineStyle','-'),...
    struct('color',[0,0,0],'lineStyle','-'),...%    struct('color',[1,1,0],'lineStyle','-'),...%yellow
    struct('color',[1,0,1],'lineStyle','-'),...%pink
    struct('color',[0,1,1],'lineStyle','-'),...
    struct('color',[0.5,0.5,0.5],'lineStyle','-'),...%gray-25%
    struct('color',[136,0,21]/255,'lineStyle','-'),...%dark red
    struct('color',[255,127,39]/255,'lineStyle','-'),...%orange
    struct('color',[0,162,232]/255,'lineStyle','-'),...%Turquoise
    struct('color',[163,73,164]/255,'lineStyle','-'),...%purple    %%%%%%%%%%%%%%%%%%%%
    struct('color',[1,0,0],'lineStyle','--'),...
    struct('color',[0,1,0],'lineStyle','--'),...
    struct('color',[0,0,1],'lineStyle','--'),...
    struct('color',[0,0,0],'lineStyle','--'),...%    struct('color',[1,1,0],'lineStyle','--'),...%yellow
    struct('color',[1,0,1],'lineStyle','--'),...%pink
    struct('color',[0,1,1],'lineStyle','--'),...
    struct('color',[0.5,0.5,0.5],'lineStyle','--'),...%gray-25%
    struct('color',[136,0,21]/255,'lineStyle','--'),...%dark red
    struct('color',[255,127,39]/255,'lineStyle','--'),...%orange
    struct('color',[0,162,232]/255,'lineStyle','--'),...%Turquoise
    struct('color',[163,73,164]/255,'lineStyle','--'),...%purple    %%%%%%%%%%%%%%%%%%%
    struct('color',[1,0,0],'lineStyle','-.'),...
    struct('color',[0,1,0],'lineStyle','-.'),...
    struct('color',[0,0,1],'lineStyle','-.'),...
    struct('color',[0,0,0],'lineStyle','-.'),...%    struct('color',[1,1,0],'lineStyle',':'),...%yellow
    struct('color',[1,0,1],'lineStyle','-.'),...%pink
    struct('color',[0,1,1],'lineStyle','-.'),...
    struct('color',[0.5,0.5,0.5],'lineStyle','-.'),...%gray-25%
    struct('color',[136,0,21]/255,'lineStyle','-.'),...%dark red
    struct('color',[255,127,39]/255,'lineStyle','-.'),...%orange
    struct('color',[0,162,232]/255,'lineStyle','-.'),...%Turquoise
    struct('color',[163,73,164]/255,'lineStyle','-.'),...%purple
    };

plotDrawStyle10={   struct('color',[1,0,0],'lineStyle','-'),...
    struct('color',[0,1,0],'lineStyle','--'),...
    struct('color',[0,0,1],'lineStyle',':'),...
    struct('color',[0,0,0],'lineStyle','-'),...%    struct('color',[1,1,0],'lineStyle','-'),...%yellow
    struct('color',[1,0,1],'lineStyle','--'),...%pink
    struct('color',[0,1,1],'lineStyle',':'),...
    struct('color',[0.5,0.5,0.5],'lineStyle','-'),...%gray-25%
    struct('color',[136,0,21]/255,'lineStyle','--'),...%dark red
    struct('color',[255,127,39]/255,'lineStyle',':'),...%orange
    struct('color',[0,162,232]/255,'lineStyle','-'),...%Turquoise
    };

% seqs=configUAVSeqs100;
seqs = configUAVSeqs;


trackers=configTrackers;

% seqs = seqs(1:10);
% trackers = trackers(1:10);

numSeq=length(seqs);
numTrk=length(trackers);

nameTrkAll=cell(numTrk,1);
for idxTrk=1:numTrk
    t = trackers{idxTrk};
    nameTrkAll{idxTrk}=t.namePaper;
end

nameSeqAll=cell(numSeq,1);
numAllSeq=zeros(numSeq,1);

att=[];
for idxSeq=1:numSeq
    s = seqs{idxSeq};
    nameSeqAll{idxSeq}=s.name;
    
    s.len = s.endFrame - s.startFrame + 1;
    
    numAllSeq(idxSeq) = s.len;
    
    % att(idxSeq,:)=load([attPath s.name '.txt']);
    att(idxSeq,:)=load(fullfile(attPath, [s.name, '.txt']));
end

attNum = size(att,2);

figPath = fullfile(cur_dir, 'fig', 'overall');
perfMatPath = fullfile(cur_dir, 'perfMat', 'overall');

if ~exist(figPath,'dir')
    mkdir(figPath);
end

metricTypeSet = {'error', 'overlap'};
% metricTypeSet = {'error'};
% evalTypeSet = {'SRE', 'TRE', 'OPE'};
evalTypeSet = {'OPE'};
rankingType = 'threshold';%AUC, threshod
% rankingType = 'threshold';

rankNum = 10;%number of plots to show

if rankNum == 10
    plotDrawStyle=plotDrawStyle10;
else
    plotDrawStyle=plotDrawStyleAll;
end

thresholdSetOverlap = 0:0.05:1;
thresholdSetError = 0:50;

for i=1:length(metricTypeSet)
    metricType = metricTypeSet{i};%error,overlap
    
    switch metricType
        case 'overlap'
            thresholdSet = thresholdSetOverlap;
            rankIdx = 11;
            xLabelName = 'Overlap threshold';
            yLabelName = 'Success rate';
            rankingType = 'AUC';
        case 'error'
            thresholdSet = thresholdSetError;
            rankIdx = 21;
            xLabelName = 'Location error threshold';
            yLabelName = 'Precision';
            rankingType = 'threshold';
    end  
        
    if strcmp(metricType,'error')&strcmp(rankingType,'AUC')
        continue;
    end
    
    tNum = length(thresholdSet);
    
    for j=1:length(evalTypeSet)
        
        evalType = evalTypeSet{j};%SRE, TRE, OPE
        
        plotType = [metricType '_' evalType];
        
        switch metricType
            case 'overlap'
                titleName = ['Success plots of ' evalType];
            case 'error'
                titleName = ['Precision plots of ' evalType];
        end

        dataName = fullfile(perfMatPath, ['aveSuccessRatePlot_' num2str(numTrk) 'alg_'  plotType '.mat']);
        % dataName = [perfMatPath 'aveSuccessRatePlot_' num2str(numTrk) 'alg_'  plotType '.mat'];
        
        % If the performance Mat file, dataName, does not exist, it will call
        % genPerfMat to generate the file.
        if ~exist(dataName)
            genPerfMat(seqs, trackers, evalType, nameTrkAll, perfMatPath);
        end        
        
        load(dataName);
        numTrk = size(aveSuccessRatePlot,1);        
        
        if rankNum > numTrk | rankNum <0
            rankNum = numTrk;
        end
        
        % figName= [figPath 'quality_plot_' plotType '_' rankingType];
        figName= fullfile(figPath, ['quality_plot_' plotType '_' rankingType]);
        idxSeqSet = 1:length(seqs);
        
        if drawOverall
            % draw and save the overall performance plot
            plotDrawSave(numTrk,plotDrawStyle,aveSuccessRatePlot,idxSeqSet,rankNum,rankingType,rankIdx,nameTrkAll,thresholdSet,titleName, xLabelName,yLabelName,figName,metricType);
        end
        
%         tmpName = [figPath 'quality_plot_' plotType '_' rankingType];
%         for idxSeqSet = 1:51
%             figName = [tmpName, '_', seqs{idxSeqSet}.name];
%             tmptitle = [titleName, ' for ', seqs{idxSeqSet}.name];
%             plotDrawSave(numTrk,plotDrawStyle,aveSuccessRatePlot,idxSeqSet,rankNum,rankingType,rankIdx,nameTrkAll,thresholdSet,tmptitle, xLabelName,yLabelName,figName,metricType);
%         end
        
        
        
%         draw and save the performance plot for each attribute
        if drawAtt
            attTrld = 0;
            for attIdx=1:attNum
                
                idxSeqSet=find(att(:,attIdx)>attTrld);
                
                if length(idxSeqSet) < 2
                    continue;
                end
                % disp([attName{attIdx} ' ' num2str(length(idxSeqSet))]);
                
                % figName=[figPath attFigName{attIdx} '_'  plotType '_' rankingType];
                figName=fullfile(figPath, [attFigName{attIdx} '_'  plotType '_' rankingType]);
                titleName = ['Plots of ' evalType ': ' attName{attIdx} ' (' num2str(length(idxSeqSet)) ')'];
                
                switch metricType
                    case 'overlap'
                        titleName = ['Success plots of ' evalType ' - ' attName{attIdx} ' (' num2str(length(idxSeqSet)) ')'];
                    case 'error'
                        titleName = ['Precision plots of ' evalType ' - ' attName{attIdx} ' (' num2str(length(idxSeqSet)) ')'];
                end
                
                % plotDrawSave(numTrk,plotDrawStyle,aveSuccessRatePlot,idxSeqSet,rankNum,rankingType,rankIdx,nameTrkAll,thresholdSet,titleName, xLabelName,yLabelName,figName,metricType);
                % show the attribute based score
                for idxTrk=1:numTrk
                    %each row is the sr plot of one sequence
                    tmp=aveSuccessRatePlot(idxTrk, idxSeqSet,:);
                    aa=reshape(tmp,[length(idxSeqSet),size(aveSuccessRatePlot,3)]);
                    aa=aa(sum(aa,2)>eps,:);
                    bb=mean(aa, 1);
                    switch rankingType
                        case 'AUC'
                            perf(idxTrk) = mean(bb);
                        case 'threshold'
                            perf(idxTrk) = bb(rankIdx);
                    end
                    fprintf('%.4f\t', perf(idxTrk));
                end
                fprintf('\n');
            end
        end      
    end
end
