function button=disp_points_distribute(locs_1,locs_2,cor_1,cor_2,...
                            nOctaves_1,nOctaves_2,dog_center_layer)

dis_num1=zeros(nOctaves_1,dog_center_layer);
dis_num2=zeros(nOctaves_2,dog_center_layer);
dis_num11=zeros(nOctaves_1,dog_center_layer);
dis_num22=zeros(nOctaves_2,dog_center_layer);

for i=1:1:nOctaves_1
    for j=1:1:dog_center_layer
        temp1=find(locs_1(:,6)==i & locs_1(:,7)==j+1);
        dis_num1(i,j)=size(temp1,1);
        temp2=find(cor_1(:,3)==i & cor_1(:,4)==j+1);
        dis_num11(i,j)=size(temp2,1);
    end
end

for i=1:1:nOctaves_2
    for j=1:1:dog_center_layer
        temp1=find(locs_2(:,6)==i & locs_2(:,7)==j+1);
        dis_num2(i,j)=size(temp1,1);
        temp2=find(cor_2(:,3)==i & cor_2(:,4)==j+1);
        dis_num22(i,j)=size(temp2,1);
    end
end

width=0.5;
button=figure;
subplot(2,2,1);
b1=bar3(dis_num1,width);
xlabel('layer num');
ylabel('group num');
zlabel('point num');

title(['Reference point initial distribution ',num2str(size(locs_1,1)),'points']);
labely1=cell(1,nOctaves_1);
labelx1=cell(1,dog_center_layer);
for i=1:1:nOctaves_1
    labely1{1,i}=['oct-',num2str(i)];
end
for i=1:1:dog_center_layer
    labelx1{1,i}=['lay-',num2str(i)];
end
set(gca,'xticklabel',labelx1);
set(gca,'yticklabel',labely1);
height=max(max(dis_num1))*0.1;
for x=1:dog_center_layer
    for y=1:nOctaves_1
        text(x,y,dis_num1(y,x)+height,num2str(dis_num1(y,x)));
    end
end
set(gca,'FontName','Time New Roman','FontSize',7);

subplot(2,2,2);
b2=bar3(dis_num2,width);
title(['The image to be registered initial point distribution ',num2str(size(locs_2,1)),'points']);
labelx2=cell(1,nOctaves_2);
labely2=cell(1,dog_center_layer);
for i=1:1:nOctaves_2
    labely2{1,i}=['oct-',num2str(i)];
end
for i=1:1:dog_center_layer
    labelx2{1,i}=['lay-',num2str(i)];
end
set(gca,'xticklabel',labelx2);
set(gca,'yticklabel',labely2);
height=max(max(dis_num2))*0.1;
for x=1:dog_center_layer
    for y=1:nOctaves_2
        text(x,y,dis_num2(y,x)+height,num2str(dis_num2(y,x)));
    end
end
xlabel('layer nun');
ylabel('group num');
zlabel('point num');
set(gca,'FontName','Time New Roman','FontSize',7);

subplot(2,2,3);
bar3(dis_num11,width);
title(['Reference right num ',num2str(size(cor_1,1)),'points']);
set(gca,'xticklabel',labelx1);
set(gca,'yticklabel',labely1);
height=max(max(dis_num11))*0.1;
for x=1:dog_center_layer
    for y=1:nOctaves_1
        text(x,y,dis_num11(y,x)+height,num2str(dis_num11(y,x)));
    end
end
xlabel('layer');
ylabel('group');
zlabel('point num');
set(gca,'FontName','Time New Roman','FontSize',7);

%待配准正确配对的点
f4=subplot(2,2,4);
bar3(dis_num22,width);
title(['The image to be registeded right num ',num2str(size(cor_2,1)),'points']);
set(gca,'xticklabel',labelx2);
set(gca,'yticklabel',labely2);
height=max(max(dis_num22))*0.1;
for x=1:dog_center_layer
    for y=1:nOctaves_2
        text(x,y,dis_num22(y,x)+height,num2str(dis_num22(y,x)));
    end
end
xlabel('layer');
ylabel('group');
zlabel('point num');
set(gca,'FontName','Time New Roman','FontSize',7);

end



