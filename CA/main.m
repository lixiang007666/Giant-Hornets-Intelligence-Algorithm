clear
clc
close all

%% ����
load pos.mat

pSurvive = 0.1;     % ÿ�������ɵķ䳲�Ĵ���
pSurvive2 = 0;  % ��Ϊ�����£��䳲�Ĵ���

load matlab.mat

figure
plot(boundary(:, 1),boundary(:, 2), 'color', 'k')
hold on

%% Ԫ���Զ�������
nestPos = posStart;
for idx = 1 : 30
    nestPosAll = [];
    for idxNest = 1 : size(nestPos, 1)
        nestNumNew = sum(rand(40, 1) < pSurvive);
        % Longitude ��0.33���ڣ�Latitude ��0.27���ڣ����Ա�֤30km����
        nestPosNew = [unifrnd(nestPos(idxNest, 1)-0.33, nestPos(idxNest, 1)+0.33, nestNumNew, 1) ...
            unifrnd(nestPos(idxNest, 2)-0.27, nestPos(idxNest, 2)+0.27, nestNumNew, 1)];
        nestPosNew = posDeleteByDist(nestPosNew, 0.07);
        
        if ~isempty(nestPosNew)
            inFlag = inpolygon(nestPosNew(:,1), nestPosNew(:, 2), boundary(:, 1), boundary(:, 2));
            nestPosNew = nestPosNew(inFlag, :);
            nestPosAll = [nestPosAll; nestPosNew];        
        end
    end
    if ~isempty(nestPosAll)
        nestSurviveIdx = rand(size(nestPosAll, 1), 1) < pSurvive2;
        nestPosAll = nestPosAll(nestSurviveIdx, :);
        nestPosAll = posDeleteByDist(nestPosAll, 0.07);
        disp(size(nestPosAll,1));
    end
    
    % ��ͼ
    if ~isempty(nestPosAll)
        nestPos = nestPosAll;
        exist p 'var';
        if ans
            delete(p);
        end
        p = plot(nestPos(:, 1), nestPos(:, 2), '*', 'color', [1 0.25 0]);
        drawnow
        pause(0.01)
    end
end
