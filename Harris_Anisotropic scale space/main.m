clear all;
close all;

%该函数根据各向异性扩散原理构建各向异性的尺度空间，并用harris进行角点检测

%% 读入并显示参考和待配准图像

[filename,pathname]=uigetfile({'*.*','All Files(*.*)'},'选择参考图像和待配准图像',...
                          'F:\class_file\图像配准\图像配准');%选择多个图像，这里两个
image_1=imread(strcat(pathname,filename));
[filename,pathname]=uigetfile({'*.*','All Files(*.*)'},'选择参考图像和待配准图像',...
                          'F:\class_file\图像配准\图像配准');%选择多个图像，这里两个
image_2=imread(strcat(pathname,filename));

figure;
subplot(2,1,1);
imshow(image_1);
title('参考图像');
subplot(2,1,2);
imshow(image_2);
title('待配准图像');
%button=questdlg('是否显示中间结果图像或数据？','显示选择','YES','NO','YES');
button='NO';

t1=clock;
%% 初始参数设定
sigma_1=1.6;%第一层的尺度
sigma_2=1;
ratio=2^(1/3);%尺度比
nbin=500;%计算对比度因子时候构建的梯度直方图的Bin个数
perc=0.7;%计算对比度因子时候的百分位,这个值越大，平滑越多
Mmax=8;%尺度空间的层数
which_diff=2;%选择计算扩散系数的方程
is_auto='YES';%是否自动计算对比度阈值k
first_layer=1;%极值点检测开始层数

d=0.04;%HARRIS函数任意常数默认是0.04
d_SH_1=0.1;%参考图像阈值如果是scharr滤波时候取值较大500，如果是sobel滤波取值较小
d_SH_2=0.1;%待配准图像阈值

change_form='相似变换';%可以是相似变换，仿射变换，
sift_or_log_polar='对数极坐标描述子';%可以是‘对数极坐标描述子’和‘SIFT描述子’

%% 转换输入图像格式
[~,~,num1]=size(image_1);
[~,~,num2]=size(image_2);
if(num1==3)
    image_11=rgb2gray(image_1);
else
    image_11=image_1;
end
if(num2==3)
    image_22=rgb2gray(image_2);
else
    image_22=image_2;
end

%转换为浮点数据
image_11=im2double(image_11);
image_22=im2double(image_22);                   

%% 图像加入噪声，验证算法对噪声的影响
%图像加入噪声，这里仅仅对待配准图像加入噪声
% button=questdlg('选择噪声类型','显示选择','高斯噪声',....
%     '椒盐噪声','乘性噪声','高斯噪声');
% if(strcmp(button,'高斯噪声'))
%     prompt={'输入高斯噪声均值:','输入高斯噪声方差:'};
%     dlg_title='高斯噪声参数输入';
%     def={'0','0.01'};%默认的均值是0，方差是0.01
%     numberlines=1;
%     answer=str2double(inputdlg(prompt,dlg_title,numberlines,def));
%     image_22=imnoise(image_22,'gaussian',answer(1),answer(2));
% elseif(strcmp(button,'椒盐噪声'))
%     prompt={'输入概率Pa:'};
%     dlg_title='椒盐噪声参数输入';
%     def={'0.1'};%%默认是0.05
%     numberlines=1;
%     answer=str2double(inputdlg(prompt,dlg_title,numberlines,def));
%     image_22=imnoise(image_22,'salt & pepper',answer);
% elseif(strcmp(button,'乘性噪声'))
%     prompt={'输入方差:'};
%     dlg_title='乘性噪声参数输入';
%     def={'0.1'};%默认是0.04
%     numberlines=1;
%     answer=str2double(inputdlg(prompt,dlg_title,numberlines,def));
%     image_22=imnoise(image_22,'speckle',answer);
% else 
%     image_22=image_22;
% end
% figure;
% subplot(1,2,1);
% imshow(image_2);
% title('待配准图像');
% subplot(1,2,2);
% imshow(image_22);
% title(['加入',button,'的待配准图像']);


%% 创建非线性尺度空间，这里仅仅创建了尺度空间，
tic;
[nonelinear_space_1]=create_nonlinear_scale_space(image_11,sigma_1,sigma_2,ratio,...
                                 Mmax,nbin,perc,which_diff,is_auto);
[nonelinear_space_2]=create_nonlinear_scale_space(image_22,sigma_1,sigma_2,ratio,...
                                 Mmax,nbin,perc,which_diff,is_auto);
disp(['构造各向异性尺度空间花费时间：',num2str(toc),'秒']);

%% 根据上面的各向异性尺度空间生成Harris尺度空间，尺度空间的每层图像表示harris函数
tic;
[harris_function_1,gradient_1,angle_1,]=...
    harris_scale(nonelinear_space_1,d,sigma_1,ratio);  
[harris_function_2,gradient_2,angle_2]=...
    harris_scale(nonelinear_space_2,d,sigma_1,ratio);                                                                                          
disp(['构造HARRIS函数尺度空间花费时间：',num2str(toc),'秒']);

%% 显示上面生成的结果操作
if(strcmp(button,'YES'))
    display_product_image(nonelinear_space_1,gradient_1,angle_1,harris_function_1,'参考');
    display_product_image(nonelinear_space_2,gradient_2,angle_2,harris_function_2,'待配准');                                                                                
end                                          

%% 在SAR-HARRIS函数中查找极值点
tic;
[position_1]=find_scale_extreme(harris_function_1,d_SH_1,sigma_1,ratio,...
             gradient_1,angle_1,first_layer);
[position_2]=find_scale_extreme(harris_function_2,d_SH_2,sigma_1,ratio,...
             gradient_2,angle_2,first_layer);
disp(['尺度空间查找极值点花费时间：',num2str(toc),'秒']);

%% 显示检测到的角点的位置在参考图像和待配准图像上
% showpoint_detected(image_1,image_2,position_1,position_2);

%% 计算参考图像和待配准图像的描述符
tic;
[descriptors_1,locs_1]=calc_descriptors(gradient_1,angle_1,...
                                        position_1,sift_or_log_polar);                                     
[descriptors_2,locs_2]=calc_descriptors(gradient_2,angle_2,...
                                        position_2,sift_or_log_polar);   
disp(['计算描述符花费时间：',num2str(toc),'秒']);
                                              
%% 开始匹配
tic;
[solution,rmse,cor1,cor2]=match(image_2, image_1,...
                                descriptors_2,locs_2,...
                                descriptors_1,locs_1,change_form);
    
tform=maketform('projective',solution');
% tform=maketform('affine',solution');
[M,N,P]=size(image_1);
ff=imtransform(image_2,tform, 'XData',[1 N], 'YData',[1 M]);
f=figure;
subplot(1,2,1);
imshow(image_1);
title('参考图像');
subplot(1,2,2);
imshow(ff);
title('配准后的图像');
disp(['描述符匹配花费时间：',num2str(toc),'秒']);
%保存
str1=['.\save_image\','参考和配准后的图像','.jpg'];
saveas(f,str1,'jpg');
str=['.\save_image\','配准后图像.jpg'];
imwrite(ff,str,'jpg');
str=['.\save_image\','参考图像.jpg'];
imwrite(image_1,str,'jpg');
str=['.\save_image\','待配准图像.jpg'];
imwrite(image_2,str,'jpg');

%% 图像融合
image_fusion(image_1,image_2,solution)
t2=clock;
disp(['花费总时间是：',num2str(etime(t2,t1)),'秒']);                                              

%% 显示点分布
button=disp_points_distribute_1(locs_1,locs_2,cor2,cor1,Mmax);
% %保存
% str1=['.\save_image\','检测点分布','.jpg'];
% saveas(button,str1,'jpg');
% 
% showpoint_detected(image_1,image_2,cor2,cor1);                                             
                                              
                                              
                                              
                                              
                                              
                                              
                                              
                                              
