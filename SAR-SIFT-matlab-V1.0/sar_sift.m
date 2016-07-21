%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  auther:
%  Key Laboratory of Intelligent Perception and Image Understanding of Ministry 
%  of Education, International Research Center for Intelligent Perception and 
%  Computation, Xidian University, Xian 710071,China(e-mail:zelianwen@foxmail.com).
%  Reference paper:
%  Julie Delon ; Yann Gousseau ; Julien Michel ; Florence Tupin
%  "Sar-sift: A sift-like algorithm for sar images"
%  version：SAR-SIFT-matlab-V1.0
%  In the future, we will release the c++ version.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;

%% read image
[filename,pathname]=uigetfile({'*.*','All Files(*.*)'},'选择参考图像和待配准图像',...
                          'F:\class_file\图像配准\图像配准','MultiSelect','on');
image_1=imread(strcat(pathname,filename{1}));
image_2=imread(strcat(pathname,filename{2}));

%% make file for save images
if (exist('save_image','dir')==0)%如果文件夹不存在
    mkdir('save_image');
end

%%
figure;
subplot(2,1,1);
imshow(image_1);
title('Reference image');
subplot(2,1,2);
imshow(image_2);
title('Image to be registered');
str=['.\save_image\','Reference image.jpg'];
imwrite(image_1,str,'jpg');
str=['.\save_image\','Image to be registered.jpg'];
imwrite(image_2,str,'jpg');

%% Convert input image format
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

%Converted to floating point data
image_11=im2double(image_11);
image_22=im2double(image_22);                                     

%%
t1=clock;
sigma=2;%Initial layer scale
ratio=2^(1/3);%scale ratio
Mmax=8;%layer number
d=0.04;
d_SH_1=0.8;%Harris function threshold
d_SH_2=0.8;%Harris function threshold
change_form='affine';%it can be 'similarity','afine','perspective'

%% Create SAR-HARRIS function
tic;
[sar_harris_function_1,gradient_1,angle_1]=build_scale(image_11,sigma,Mmax,ratio,d);
[sar_harris_function_2,gradient_2,angle_2]=build_scale(image_22,sigma,Mmax,ratio,d);
disp(['Create SAR-HARRIS function Spend time：',num2str(toc),'s']);

%% Feature point detection
tic;
[GR_key_array_1]=find_scale_extreme(sar_harris_function_1,d_SH_1,sigma,ratio,gradient_1,angle_1);
[GR_key_array_2]=find_scale_extreme(sar_harris_function_2,d_SH_2,sigma,ratio,gradient_2,angle_2);
disp(['Feature point detection：',num2str(toc),'s']);

%% Calculating 
tic;
[descriptors_1,locs_1]=calc_descriptors(gradient_1,angle_1,GR_key_array_1);
[descriptors_2,locs_2]=calc_descriptors(gradient_2,angle_2,GR_key_array_2);
disp(['Calculating ：',num2str(toc),'s']);

%% match
tic;
[solution,rmse,cor1,cor2]=match(image_2, image_1,descriptors_2,locs_2,descriptors_1,locs_1,change_form);
disp(['points match spend time：',num2str(toc),'s']); 

%% Calculation model parameters
tform=maketform('projective',solution');
[M,N,P]=size(image_1);
ff=imtransform(image_2,tform, 'XData',[1 N], 'YData',[1 M]);
figure;
subplot(1,2,1);
imshow(image_1);
title('reference image');
subplot(1,2,2);
imshow(ff);
title('Image after registration');
str=['.\save_image\','Image after registration.jpg'];
imwrite(ff,str,'jpg');

%% display points
showpoint_detected(image_1,image_2,GR_key_array_1,GR_key_array_2);

%% image fusion
image_fusion(image_1,image_2,solution);

t2=clock;
disp(['Total time：',num2str(etime(t2,t1)),'s']); 


