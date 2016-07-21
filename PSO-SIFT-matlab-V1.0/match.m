function [solution,rmse,cor1,cor2]=match(im1, im2,des1,loc1,des2,loc2,change_form)

Td=0.9;
distRatio =0.9;   
des2t = des2'; 
M_des2=size(des2,1);
for i = 1 : size(des1,1)
    %Euclidean distance
    temp_des1=des1(i,:);
    temp_des1=repmat(temp_des1,M_des2,1);
    diff_des1=temp_des1-des2;
    ED_distance=sqrt(sum(diff_des1.^2,2));  
    [vals,indx] = sort(ED_distance);
    cos_dis(i,:)=ED_distance';
    
    %Range ratio phase
    if (vals(1) < distRatio * vals(2))
     match(i) = indx(1);
    else
      match(i) = 0;
    end
end
num = sum(match > 0);
fprintf('NNDR Found %d matches.\n', num);
[~,point1,point2]=find(match);

loc1(find(loc1(:,4)>180)',4)=loc1(find(loc1(:,4)>180)',4)-360;
loc2(find(loc2(:,4)>180)',4)=loc2(find(loc2(:,4)>180)',4)-360;
cor1=loc1(point1,[2 1 6 7 3 4]);cor2=loc2(point2,[2 1 6 7 3 4]);
cor1=[cor1 point2'];cor2=[cor2 point2'];

%% Delete duplicate point pair
uni1=[cor1(:,[1 2]),cor2(:,[1 2])];
[~,i,~]=unique(uni1,'rows','first');
cor1=cor1(sort(i)',:);cor2=cor2(sort(i)',:);

%% display histograms of scale ratio,angle difference,position shift
scale_ratio_diff_angle(cor2(:,[5,6,1,2]),cor1(:,[5,6,1,2]));

%% FSC
[solution_old,~,cor1,cor2]=FSC(cor1,cor2,change_form,1);%%Sub-pixel accuracy
fprintf('Initial matching points is：%d matches.\n', size(cor1,1));
button=appendimages(im2,im1,cor2,cor1);
str1=['.\save_image\','intial correct matching points','.jpg'];
saveas(button,str1,'jpg');

%% scale_orien_position_joint_restriction
[solution,cor1,cor2,rmse]=scale_orien_joint_restriction(solution_old,...
            loc1,loc2,cor1,cor2,des1,des2,Td,change_form,cos_dis);
    
fprintf('Finally, the number of correct matching points is：%d matches.\n', size(cor1,1));
button=appendimages(im2,im1,cor2,cor1);
str1=['.\save_image\','correct matching points','.jpg'];
saveas(button,str1,'jpg');

%% 输出【x,y,组，层，尺度，角度】保存下来
cor1=cor1(:,1:6);
cor2=cor2(:,1:6);





