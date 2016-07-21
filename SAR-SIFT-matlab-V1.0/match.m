function [solution,rmse,cor1,cor2]= match(im1, im2,des1,loc1,des2,loc2,change_form)

distRatio = 0.9;   
des2t = des2';   

M_des1=size(des1,1);
M_des2=size(des2,1);
ED_distance=zeros(M_des1,M_des2);
for i=1:1:M_des1
    temp_des1=des1(i,:);
    temp_des1=repmat(temp_des1,M_des2,1);
    diff_des1=temp_des1-des2;
    diff_des1=sqrt(sum(diff_des1.^2,2));
    ED_distance(i,:)=diff_des1';
end

for i = 1 : size(des1,1)
  dotprods = des1(i,:) * des2t;        
   [vals,indx] = sort(acos(dotprods));  
  %[vals,indx]=sort(ED_distance(i,:));
  if (vals(1) < distRatio * vals(2))
     match(i) = indx(1);
  else
      match(i) = 0;
  end
end
num = sum(match > 0);
fprintf('NNDR found %d matchs.\n', num);
[~,point1,point2]=find(match);
%±£´æ¡¾x,y,³ß¶È£¬layer£¬½Ç¶È¡¿
cor11=loc1(point1,[1 2 3 4 5]);cor22=loc2(point2,[1 2 3 4 5]);
cor11=[cor11 point2'];cor22=[cor22 point2'];

% Remove duplicate points
uni1=[cor11(:,[1 2]),cor22(:,[1 2])];
[~,i,~]=unique(uni1,'rows','first');
cor11=cor11(sort(i)',:);
cor22=cor22(sort(i)',:);

%% FSC
[solution,rmse,cor1,cor2]=FSC(cor11,cor22,change_form,1);
fprintf('After FSC found %d matches.\n', size(cor1,1));
button=appendimages(im2,im1,cor2,cor1);
str1=['.\save_image\','After FSC right match','.jpg'];
saveas(button,str1,'jpg');

%% 
[button1,button2]=showpoints(im2,im1,cor2,cor1);
str1=['.\save_image\','Image to be registered right points','.jpg'];
saveas(button1,str1,'jpg');
str1=['.\save_image\','Reference right points','.jpg'];
saveas(button2,str1,'jpg');

%% 
cor1=cor1(:,1:5);
cor2=cor2(:,1:5);

end