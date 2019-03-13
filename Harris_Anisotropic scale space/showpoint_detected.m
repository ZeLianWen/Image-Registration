function showpoint_detected(im1,im2,cor1,cor2)
%% 这里im1是参考图像，im2是待配准图像,
%该函数的作用是显示在参考图像和待配准图像上检测到的角点
%因为一个特征点可能有多个主方向，因此删除重复计数的特征点
uni1=cor1(:,[1,2,3,4]);
[~,i,~]=unique(uni1,'rows','first');
cor1=cor1(sort(i)',:);
cor1_x=cor1(:,1);cor1_y=cor1(:,2);
f=figure;colormap('gray');imagesc(im1);
title(['参考图像',num2str(size(cor1_x,1)),'特征点']);hold on;
scatter(cor1_x,cor1_y,'r');hold on;%scatter可用于描绘散点图
str1=['.\save_image\','参考图像检测特征点','.jpg'];
saveas(f,str1,'jpg');
fprintf('参考图像检测到特征点个数是%d\n', size(cor1,1));

uni1=cor2(:,[1,2,3,4]);
[~,i,~]=unique(uni1,'rows','first');
cor2=cor2(sort(i)',:);
cor2_x=cor2(:,1);cor2_y=cor2(:,2);
f=figure;colormap('gray');imagesc(im2);
title(['待配准图像',num2str(size(cor2_x,1)),'特征点']);hold on;
scatter(cor2_x,cor2_y,'r');hold on;
str1=['.\save_image\','待配准图像检测特征点','.jpg'];
saveas(f,str1,'jpg');
fprintf('待配准图像检测到特征点个数是%d\n', size(cor2,1));


uni1=cor1(:,[1,2,3,4]);
[~,i,~]=unique(uni1,'rows','first');
cor1=cor1(sort(i)',:);
cor1_x=cor1(:,1);cor1_y=cor1(:,2);
figure;
imshow(im1,'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,0,size(im1,2) size(im1,1)]);
axis normal;
hold on;
scatter(cor1_x,cor1_y,'r','*');hold on;%scatter可用于描绘散点图
for i=1:size(cor1,1)
text(cor1_x(i),cor1_y(i),num2str(i),'color','b');
end

uni1=cor2(:,[1,2,3,4]);
[~,i,~]=unique(uni1,'rows','first');
cor2=cor2(sort(i)',:);
cor2_x=cor2(:,1);cor2_y=cor2(:,2);
figure;
imshow(im2,'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,0,size(im2,2) size(im2,1)]);
axis normal;
hold on;
scatter(cor2_x,cor2_y,'r','*');hold on;%scatter可用于描绘散点图
for i=1:size(cor1,1)
text(cor2_x(i),cor2_y(i),num2str(i),'color','b');
end


end



