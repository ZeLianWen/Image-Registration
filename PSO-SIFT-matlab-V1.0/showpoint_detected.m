function [button_1,button_2]=showpoint_detected(im1,im2,cor1,cor2)

uni1=cor1(:,[1 2 3 4 5]);
[~,i,~]=unique(uni1,'rows','first');
cor1=cor1(sort(i)',:);
cor1_x=cor1(:,2);cor1_y=cor1(:,1);
button_1=figure;colormap('gray');imagesc(im1);
title(['Reference image ',num2str(size(cor1_x,1)),' points']);hold on;
scatter(cor1_x,cor1_y,'r');hold on;
fprintf('The number of feature points for reference image detection is %d.\n', size(cor1,1));

uni1=cor2(:,[1 2 3 4 5]);
[~,i,~]=unique(uni1,'rows','first');
cor2=cor2(sort(i)',:);
cor2_x=cor2(:,2);cor2_y=cor2(:,1);
button_2=figure;colormap('gray');imagesc(im2);
title(['Image to be registered ',num2str(size(cor2_x,1)),' points']);hold on;
scatter(cor2_x,cor2_y,'r');hold on;
fprintf('The number of feature points for image detection to be registered is %d.\n', size(cor2,1));
end



