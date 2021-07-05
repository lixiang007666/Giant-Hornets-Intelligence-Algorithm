function posNew = posDeleteByDist( pos, th )
% ====================== 函数说明 ======================
% 通过距离删除过近的位置
% ====================== 输入参数 ======================
% pos : 位置矩阵(n*2)
% th : 位置阈值
% ====================== 输出参数 ======================
% posNew : 删除之后的位置矩阵
% =====================================================

posNew = [];
posNum = size(pos, 1);
for idx = 1 : posNum
    disTemp = (pos-ones(posNum, 1)*pos(idx, :));
    disTemp = sqrt(sum(disTemp.^2, 2));
    disTemp(disTemp == 0) = [];
    if isempty(find(disTemp < th, 1))
        posNew = [posNew; pos(idx, :)];
    end
end

end

