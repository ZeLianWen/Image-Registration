%%根据特征点集合计算特征描述子
function [descriptors,locs]=calc_descriptors...
(   gradient,...%尺度空间的梯度
    angle,...%尺度空间的角度
    key_point_array,....%特征点矩阵
    sift_or_log_polar...
)
%% 参数初始化
%sift描述符参数
SIFT_DESCR_WIDTH=4;
SIFT_DESC_HIST_BINS=8;

%对数极坐标描述符参数
LOG_POLAR_WIDTH=8;%对数极坐标参数
LOG_POLAR__HIST_BINS=8;%角度方向分为8个部分，就是每45度一个部分

M=size(key_point_array,1);%%获取特征点个数
if(strcmp(sift_or_log_polar,'SIFT描述子'))
    d=SIFT_DESCR_WIDTH;
    n=SIFT_DESC_HIST_BINS;
    descriptors=zeros(M,d*d*n);%SIFT描述符
    temp=1;
elseif(strcmp(sift_or_log_polar,'对数极坐标描述子'))
    d=LOG_POLAR_WIDTH;
    n=LOG_POLAR__HIST_BINS;
    descriptors=zeros(M,(2*d+1)*n);%对数极坐标描述符
    temp=0;
end

%locs保存着特征点的信息[x,y,尺度，层，角度，梯度]，因此locs是一个大小为M*6的矩阵
locs=key_point_array;
for i=1:1:M
    x=key_point_array(i,1);%特征点的水平坐标
    y=key_point_array(i,2);%特征点的竖直坐标
    scale=key_point_array(i,3);%特征点所在的尺度
    layer=key_point_array(i,4);%特征点所在的层数
    main_angle=key_point_array(i,5);%特征点的主方向

    %得到特征点所在层的所有梯度和多有角度
    current_gradient=gradient(:,:,layer);
    current_angle=angle(:,:,layer);
    
    %% sift描述符
    if(temp==1)
        descriptors(i,:)=calc_sift_descriptor(current_gradient,current_angle,...
                                                x,y,scale,main_angle,d,n);
    elseif(temp==0)                                                                       
        descriptors(i,:)=calc_log_polar_descriptor(current_gradient,current_angle,...
                                                x,y,scale,main_angle,d,n);
    end
    
end


