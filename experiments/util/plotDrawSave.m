function plotDrawSave(numTrk,plotDrawStyle,aveSuccessRatePlot,idxSeqSet,rankNum,rankingType,rankIdx,nameTrkAll,thresholdSet,titleName,xLabelName,yLabelName,figName,metricType)

aveSuccessRate11=[];

scrsz = get(0,'ScreenSize');

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
    fprintf('%.3f\n', perf(idxTrk));
end
fprintf('\n');

[tmp,indexSort]=sort(perf,'descend');

% indexSort = [7, 6, 3, 4, 5, 1, 2];

i=1;
AUC=[];

fontSize = 16;
fontSizeLegend = 10;

figure1 = figure;

axes1 = axes('Parent',figure1,'FontSize',14);
for idxTrk=indexSort(1:rankNum)

    tmp=aveSuccessRatePlot(idxTrk,idxSeqSet,:);
    aa=reshape(tmp,[length(idxSeqSet),size(aveSuccessRatePlot,3)]);
    aa=aa(sum(aa,2)>eps,:);
    bb=mean(aa, 1);
    
    switch rankingType
        case 'AUC'
            score = mean(bb);
            tmp=sprintf('%.3f', score);
        case 'threshold'
            score = bb(rankIdx);
            tmp=sprintf('%.3f', score);
    end    
    
    tmpName{i} = [nameTrkAll{idxTrk} ' [' tmp ']'];
    h(i) = plot(thresholdSet,bb,'color',plotDrawStyle{i}.color, 'lineStyle', plotDrawStyle{i}.lineStyle,'lineWidth', 4,'Parent',axes1);
    hold on
    i=i+1;
end


legend1=legend(tmpName,'Interpreter', 'none','fontsize',fontSizeLegend, 'Location', 'southeast');
title(titleName,'fontsize',fontSize);
xlabel(xLabelName,'fontsize',fontSize);
ylabel(yLabelName,'fontsize',fontSize);
grid on;
% legend1=legend(tmpName(1:round(length(tmpName)/2)),'Interpreter', 'none','fontsize',fontSizeLegend,'Location','NorthWest');
% legend2=legend(tmpName(round(length(tmpName)/2)+1:end),'Interpreter', 'none','fontsize',fontSizeLegend,'Location','SouthEast');

%for too many
%plots--------------------------------------------------------------

% switch metricType
%     case 'error'
%         ah1 = gca;
%         %         legend1=legend(ah1,h(1:round(length(tmpName)/2)),tmpName(1:round(length(tmpName)/2)),'Interpreter', 'none','fontsize',fontSizeLegend,'Location','NorthWest');
%         legend1=legend(ah1,h(1:5),tmpName(1:5),'Interpreter', 'none','fontsize',fontSizeLegend,'Location','NorthWest');
%         
%         title(titleName,'fontsize',fontSize);
%         xlabel(xLabelName,'fontsize',fontSize);
%         ylabel(yLabelName,'fontsize',fontSize);
%         
%         ah2=axes('position',get(gca,'position'), 'visible','off');
%         legend2=legend(ah2,h(6:end),tmpName(6:end),'Interpreter', 'none','fontsize',fontSizeLegend,'Location','SouthEast');
%     case 'overlap'
%         ah1 = gca;
%         legend1=legend(ah1,h(1:5),tmpName(1:5),'Interpreter', 'none','fontsize',fontSizeLegend,'Location','NorthEast');
%         title(titleName,'fontsize',fontSize);
%         xlabel(xLabelName,'fontsize',fontSize);
%         ylabel(yLabelName,'fontsize',fontSize);
%         
%         ah2=axes('position',get(gca,'position'), 'visible','off');
%         legend2=legend(ah2,h(6:end),tmpName(6:end),'Interpreter', 'none','fontsize',fontSizeLegend,'Location','SouthWest');
% end

%     switch metricType
%         case 'error'
%             ah1 = gca;
%             %         legend1=legend(ah1,h(1:round(length(tmpName)/2)),tmpName(1:round(length(tmpName)/2)),'Interpreter', 'none','fontsize',fontSizeLegend,'Location','NorthWest');
% %             legend1=legend(ah1,h(1:round(length(tmpName)/2)),tmpName(1:round(length(tmpName)/2)),'Interpreter', 'none','fontsize',fontSizeLegend,'Position',[0.55 0.1 0.1 0.4]);
% 
%             legend1=legend(ah1,h,tmpName,'Interpreter', 'none','fontsize',fontSizeLegend,'Location','SouthEast');
%             title(titleName,'fontsize',fontSize);
%             xlabel(xLabelName,'fontsize',fontSize);
%             ylabel(yLabelName,'fontsize',fontSize);
% 
% %             ah2=axes('position',get(gca,'position'), 'visible','off');
% %             legend2=legend(ah2,h(round(length(tmpName)/2)+1:end),tmpName(round(length(tmpName)/2)+1:end),'Interpreter', 'none','fontsize',fontSizeLegend,'Location','SouthEast');
%         case 'overlap'
%             ah1 = gca;
% %             legend1=legend(ah1,h(1:round(length(tmpName)/2)),tmpName(1:round(length(tmpName)/2)),'Interpreter', 'none','fontsize',fontSizeLegend,'Location','NorthEast');
%             legend1=legend(ah1,h,tmpName,'Interpreter', 'none','fontsize',fontSizeLegend,'Location','SouthWest');
%             title(titleName,'fontsize',fontSize);
%             xlabel(xLabelName,'fontsize',fontSize);
%             ylabel(yLabelName,'fontsize',fontSize);
% 
% %             ah2=axes('position',get(gca,'position'), 'visible','off');
% %             legend2=legend(ah2,h(round(length(tmpName)/2)+1:end),tmpName(round(length(tmpName)/2)+1:end),'Interpreter', 'none','fontsize',fontSizeLegend,'Location','SouthWest');
%     end
%     
%     axes('fontsize',14);
%     set(legend1,'FontSize',fontSizeLegend);
%     set(legend2,'FontSize',fontSizeLegend);
%     set(legend1,'Interpreter','none',...
%     'Position',[0.800694444444435 0.117313517441224 0.0984375 0.521055753262159],...
%     'FontSize',fontSizeLegend);
hold off

% saveas(gcf,figName,'eps');
% print('-depsc', [figName, '.eps']);

end