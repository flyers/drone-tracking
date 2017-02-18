clear
close all

cur_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(cur_dir, 'util'));
addpath(fullfile(cur_dir, 'rstEval'));
addpath(fullfile(cur_dir, 'trackers'));

seqs = configDTBSeqs;

trackers=configTrackers;

shiftTypeSet = {'left','right','up','down','topLeft','topRight','bottomLeft','bottomRight','scale_8','scale_9','scale_11','scale_12'};

evalType='OPE'; %'OPE','SRE','TRE'

diary(fullfile(cur_dir, 'tmp', [evalType, '.txt']));

numSeq=length(seqs);
numTrk=length(trackers);

finalPath = fullfile(cur_dir, 'results', ['results_', evalType]);

if ~exist(finalPath,'dir')
    mkdir(finalPath);
end

tmpRes_path = fullfile(cur_dir, 'tmp', evalType);
bSaveImage=0;

if ~exist(tmpRes_path,'dir')
    mkdir(tmpRes_path);
end

pathAnno = fullfile(cur_dir, 'anno');

for idxSeq=1:length(seqs)
    s = seqs{idxSeq};
    s.len = s.endFrame - s.startFrame + 1;
    s.s_frames = cell(s.len,1);
    nz	= strcat('%0',num2str(s.nz),'d'); %number of zeros in the name of image
    for i=1:s.len
        image_no = s.startFrame + (i-1);
        id = sprintf(nz,image_no);
        s.s_frames{i} = strcat(s.path,id,'.',s.ext);
    end
    img = imread(s.s_frames{1});
    [imgH,imgW,ch]=size(img);
    
    rect_anno = dlmread(fullfile(pathAnno, [s.name, '.txt']));
    numSeg = 20;
    
    [subSeqs, subAnno]=splitSeqTRE(s,numSeg,rect_anno);
    
    switch evalType
        case 'SRE'
            subS = subSeqs{1};
            subA = subAnno{1};
            subSeqs=[];
            subAnno=[];
            r=subS.init_rect;
            
            for i=1:length(shiftTypeSet)
                subSeqs{i} = subS;
                shiftType = shiftTypeSet{i};
                subSeqs{i}.init_rect=shiftInitBB(subS.init_rect,shiftType,imgH,imgW);
                subSeqs{i}.shiftType = shiftType;
                
                subAnno{i} = subA;
            end
        case 'OPE'
            subS = subSeqs{1};
            subSeqs=[];
            subSeqs{1} = subS;
            
            subA = subAnno{1};
            subAnno=[];
            subAnno{1} = subA;
        otherwise
    end
            
    for idxTrk=1:numTrk
        t = trackers{idxTrk};

        % validate the results if exists already
        if exist(fullfile(finalPath, [s.name '_' t.name '.mat']))
            load(fullfile(finalPath, [s.name '_' t.name '.mat']));
            bfail=checkResult(results, subAnno);
            if bfail
                disp([s.name ' '  t.name]);
            end
            continue;
        end

        results = [];
        for idx=1:length(subSeqs)
            disp([num2str(idxTrk) '_' t.name ', ' num2str(idxSeq) '_' s.name ': ' num2str(idx) '/' num2str(length(subSeqs))])       

            rp = fullfile(tmpRes_path, [s.name '_' t.name '_' num2str(idx)]);
            if bSaveImage&~exist(rp,'dir')
                mkdir(rp);
            end
            subS = subSeqs{idx};
            subS.name = [subS.name '_' num2str(idx)];
            funcName = ['res=run_' t.name '(subS, rp, bSaveImage);'];
            eval(funcName);
            res.len = subS.len;
            res.annoBegin = subS.annoBegin;
            res.startFrame = subS.startFrame;
                    
            switch evalType
                case 'SRE'
                    res.shiftType = shiftTypeSet{idx};
            end
            
            results{idx} = res;
        end
        save(fullfile(finalPath, [s.name '_' t.name '.mat']), 'results');
    end
end

t=clock;
t=uint8(t(2:end));
disp([num2str(t(1)) '/' num2str(t(2)) ' ' num2str(t(3)) ':' num2str(t(4)) ':' num2str(t(5))]);

