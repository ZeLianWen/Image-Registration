function button=disp_points_distribute_1(locs_1,locs_2,cor2,cor1,Mmax)
%该函数显示初始检测到的点在每层的分布和最后正确配对的点在各层的分布
%locs_1是参考图像点，locs_2是待配准图像点
%cor2是参考图像正确检测到的点，cor2是待配准图像正确检测到的点
%Mmax是层数

%% 初始化变量
dis_num1=zeros(1,Mmax);%保存参考图像每层检测到的特征点
dis_num2=zeros(1,Mmax);%保存待配准图像每层检测到的特征点
dis_num11=zeros(1,Mmax);%保存参考图像最后每层正确配对点
dis_num22=zeros(1,Mmax);%保存待配准图像最后每层正确配对点
for i=1:1:Mmax
    %所有检测到的点
    temp1=find(locs_1(:,4)==i);
    temp2=find(locs_2(:,4)==i);
    dis_num1(1,i)=size(temp1,1);
    dis_num2(1,i)=size(temp2,1);
    %正确配对的点
    temp1=find(cor2(:,4)==i);
    temp2=find(cor1(:,4)==i);
    dis_num11(1,i)=size(temp1,1);
    dis_num22(1,i)=size(temp2,1);
end
horz=1:1:Mmax;

%% 显示在figure
%参考图像初始点分布
button=figure;
subplot(2,2,1);
bar(horz,dis_num1);
axis([0 Mmax+2 0 1.1*max(dis_num1)]);
xlabel('层号');
ylabel('点数');
title(['参考初始点分布',num2str(size(locs_1,1)),'个']);
height=max(dis_num1)*0.1;
for i=1:size(horz,2)
    text(horz(1,i),dis_num1(1,i)+height,num2str(dis_num1(1,i)));
end
%待配准图像初始点分布
subplot(2,2,2);
bar(horz,dis_num2);
axis([0 Mmax+2 0 1.1*max(dis_num2)]);
xlabel('层号');
ylabel('点数');
title(['待配准初始点分布',num2str(size(locs_2,1)),'个']);
height=max(dis_num2)*0.1;
for i=1:size(horz,2)
    text(horz(1,i),dis_num2(1,i)+height,num2str(dis_num2(1,i)));
end

%参考图像正确配对点分布
subplot(2,2,3);
bar(horz,dis_num11);
xlabel('层号');
ylabel('点数');
axis([0 Mmax+2 0 1.1*max(dis_num11)]);
title(['参考正确点分布',num2str(size(cor2,1)),'个']);
height=max(dis_num11)*0.1;
for i=1:size(horz,2)
    text(horz(1,i),dis_num11(1,i)+height,num2str(dis_num11(1,i)));
end

%待配准图像正确配对点分布
subplot(2,2,4);
bar(horz,dis_num22);
xlabel('层号');
ylabel('点数');
axis([0 Mmax+2 0 1.1*max(dis_num22)]);
title(['待配准正确点分布',num2str(size(cor1,1)),'个']);
height=max(dis_num22)*0.1;
for i=1:size(horz,2)
    text(horz(1,i),dis_num22(1,i)+height,num2str(dis_num22(1,i)));
end
end



