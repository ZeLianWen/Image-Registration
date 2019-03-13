function [solution,rmse,cor1,cor2]= match(im1, im2,des1,loc1,...
       des2,loc2,change_form)
%% 这里im1是待配准图像，im2是参考图像
%% 开始匹配
distRatio=0.9;
des2t = des2';

%对于参考图像中的每一个点寻找和待匹配图像中的相似点
for i = 1 : size(des1,1)
  dotprods = des1(i,:) * des2t;        
  [vals,indx] = sort(acos(dotprods));  
  if (vals(1) < distRatio * vals(2))
     match(i) = indx(1);%%match保存着des2t中的对应的特征点的索引
  else
      match(i) = 0;
  end
end

%输出参考图像和待配准图像中特征点个数
fprintf('参考图像特征描述子数目%d.\n待配准图像特征描述子数目是%d.\n', size(des2,1),size(des1,1));
num = sum(match > 0);%%匹配的个数
fprintf('初始距离比Found %d matches.\n', num);
[~,point1,point2]=find(match);
%保存【x,y,尺度，layer，角度】
cor1=loc1(point1,[1 2 3 4 5]);
cor2=loc2(point2,[1 2 3 4 5]);
cor1=[cor1 point2'];cor2=[cor2 point2'];%point2保存着最开始的特征点编号

%% 移除重复点对
uni1=[cor1(:,[1,2]),cor2(:,[1,2])];
[~,i,~]=unique(uni1,'rows','first');
cor1=cor1(sort(i)',:);cor2=cor2(sort(i)',:);
fprintf('删除重复点对后Found %d matches.\n', size(cor1,1));

%% 初始移除错误点对后使用ransac算法
[solution,rmse,cor1,cor2]=ransac(cor1,cor2,change_form,1);
fprintf('Ransac删除错误点对后 %d matches.\n', size(cor1,1));

%% 保存最后正确配对点对                                       
fprintf('最后Found %d matches.\n', size(cor1,1));
[hand_1,hand_2]=showpoints(im2,im1,cor2,cor1);
str1=['.\save_image\','参考图像正确特征点.jpg'];
saveas(hand_1,str1,'jpg');
str1=['.\save_image\','待配准图像正确配对点.jpg'];
saveas(hand_2,str1,'jpg');

fhand=appendimages(im2,im1,cor2,cor1);
str1=['.\save_image\','最后点匹配结果.jpg'];
saveas(fhand,str1,'jpg');

cor1=cor1(:,1:5);
cor2=cor2(:,1:5);



