function bfail=checkResult(results, subAnno)
% validate the results
% 
% Some trackers may fail on some sequences so before evaulating the results
% we have to check them.

bfail=0;
if isempty(results)
    disp(['Empty all'])
    bfail=1;
else
    for i=1:length(results) 
        if isempty(results{i}.res)
            disp(['Empty sub ' num2str(i)])
            bfail=1;
        else
            if size(results{i}.res,1)<size(subAnno{i},1)
                disp(['Size sub ' num2str(i)])
                bfail=1;
            end
        end
    end    
end