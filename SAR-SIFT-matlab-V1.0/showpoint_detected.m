function showpoint_detected(im1,im2,cor1,cor2)

cor1_x=cor1(:,1);cor1_y=cor1(:,2);
%cor_x1=loc1(point1,1);cor_y1=loc1(point1,2);
button_1=figure;colormap('gray');imagesc(im1);
title(['Reference image ',num2str(size(cor1_x,1)),' points']);hold on;
scatter(cor1_x,cor1_y,'r');hold on;%scatter可用于描绘散点图
str1=['.\save_image\','Reference image points','.jpg'];
saveas(button_1,str1,'jpg');

cor2_x=cor2(:,1);cor2_y=cor2(:,2);
%cor_x2=loc2(point2,1);cor_y2=loc2(point2,2);
button_2=figure;colormap('gray');imagesc(im2);
title(['Image to be registered ',num2str(size(cor2_x,1)),' points']);hold on;
scatter(cor2_x,cor2_y,'r');hold on;
str1=['.\save_image\','Image to be registered points','.jpg'];
saveas(button_2,str1,'jpg');

end


