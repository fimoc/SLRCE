function T = Gram_Schmidt_Orthogonalization(p_tr)
% 
%     % 一列为一个向量
% 
%     [row,col]= size(p_tr);
% 
%     T = zeros(row,col);
% 
%     T(:,1)=p_tr(:,1);
% 
%     for i = 2 : col
% 
%         for j = 1: i-1
% 
%             p_tr(:,i)= p_tr(:,i) - ((T(:,j)' * p_tr(:,i))/(T(:,j)' * T(:,j))) * T(:,j);
% 
%         end
% 
%         T(:,i)=p_tr(:,i);
% 
%     end
% 
%     % 向量单位化
% 
% %     for i = 1: col
% % 
% %         length=norm(T(:,i));
% % 
% %         for j = 1: row
% % 
% %             T(j,i)= T(j,i)/ length;
% % 
% %         end
% % 
% %     end
% 
% end

  
[Ahang,Alie]=size(p_tr);  %矩阵的行和列
v(:,1)=p_tr(:,1)/norm(p_tr(:,1));
for k=2:Alie
    for i=k:Alie
         p_tr(:,i)=p_tr(:,i)-p_tr(:,i)'*v(:,k-1)*v(:,k-1);%对剩余向量进行修正
    end
         v(:,k)=p_tr(:,k)/norm(p_tr(:,k));%对本次得到的正交向量进行归一化
%      v(:,k)=p_tr(:,k);
end
T=v;

