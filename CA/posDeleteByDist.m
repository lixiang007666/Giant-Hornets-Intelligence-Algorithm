function posNew = posDeleteByDist( pos, th )
% ====================== ����˵�� ======================
% ͨ������ɾ��������λ��
% ====================== ������� ======================
% pos : λ�þ���(n*2)
% th : λ����ֵ
% ====================== ������� ======================
% posNew : ɾ��֮���λ�þ���
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

