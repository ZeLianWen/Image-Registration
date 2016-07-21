function [solution,rmse,cor1,cor2]=match(im1, im2,des1,loc1,des2,loc2,change_form)

distRatio =0.9;   
M_des2=size(des2,1);
for i = 1 : size(des1,1)
    %Euclidean distance
    temp_des1=des1(i,:);
    temp_des1=repmat(temp_des1,M_des2,1);
    diff_des1=temp_des1-des2;
    ED_distance=sqrt(sum(diff_des1.^2,2));  
    [vals,indx] = sort(ED_distance);
    
    %NNDR
    if (vals(1) < distRatio * vals(2))
     match(i) = indx(1);
    else
      match(i) = 0;
    end
end
num = sum(match > 0);
fprintf('NNDR Found %d matches.\n', num);
[~,point1,point2]=find(match);

cor1=loc1(point1,[2 1 6 7 3 4]);cor2=loc2(point2,[2 1 6 7 3 4]);
cor1=[cor1 point2'];cor2=[cor2 point2'];

%% Delete duplicate point pair
uni1=[cor1(:,[1 2]),cor2(:,[1 2])];
[~,i,~]=unique(uni1,'rows','first');
cor1=cor1(sort(i)',:);cor2=cor2(sort(i)',:);

%% FSC
[solution,rmse,cor1,cor2]=FSC(cor1,cor2,change_form,1);
button=appendimages(im2,im1,cor2,cor1);
str1=['.\save_image\','After FSC right match points','.jpg'];
saveas(button,str1,'jpg');
fprintf('After FSC Found %d matches.\n', size(cor1,1));
[button1,button2]=showpoints(im2,im1,cor2,cor1);
str1=['.\save_image\','Reference after FSC right points','.jpg'];
saveas(button1,str1,'jpg');
str1=['.\save_image\','The image to be registered FSC right points','.jpg'];
saveas(button2,str1,'jpg');

%% 输出【x,y,组，层，尺度，角度】保存下来
cor1=cor1(:,1:6);
cor2=cor2(:,1:6);





