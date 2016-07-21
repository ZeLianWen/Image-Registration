%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  auther:
%  Key Laboratory of Intelligent Perception and Image Understanding of Ministry 
%  of Education, International Research Center for Intelligent Perception and 
%  Computation, Xidian University, Xian 710071,China(e-mail:zelianwen@foxmail.com).
%  version: SIFT-matlab-V1.0
%  In the future, we will release the c++ version.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all;
clear all;

%% image path
file_image='F:\class_file\图像配准\图像配准';

%% read images
[filename,pathname]=uigetfile({'*.*','All Files(*.*)'},'Reference image',...
                          file_image);
image_1=imread(strcat(pathname,filename));
[filename,pathname]=uigetfile({'*.*','All Files(*.*)'},'Image to be registered',...
                          file_image);
image_2=imread(strcat(pathname,filename));

figure;
subplot(1,2,1);
imshow(image_1);
title('Reference image');
subplot(1,2,2);
imshow(image_2);
title('Image to be registered');

%% make file for save images
if (exist('save_image','dir')==0)%如果文件夹不存在
    mkdir('save_image');
end

t1=clock;%Start time

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

%% Define the constants used
sigma=1.6;%最底层高斯金字塔的尺度
dog_center_layer=3;%定义了DOG金字塔每组中间层数，默认是3
contrast_threshold_1=0.03;%Contrast threshold
contrast_threshold_2=0.03;%Contrast threshold
edge_threshold=10;%Edge threshold
is_double_size=false;%expand image or not
change_form='affine';%change mode,'perspective','affine','similarity'
is_sift_or_log='GLOH-like';%Type of descriptor,it can be 'GLOH-like','SIFT'

%% The number of groups in Gauss Pyramid
nOctaves_1=num_octaves(image_11,is_double_size);
nOctaves_2=num_octaves(image_22,is_double_size);

%% Pyramid first layer image
image_11=create_initial_image(image_11,is_double_size,sigma);
image_22=create_initial_image(image_22,is_double_size,sigma);

%%  Gauss Pyramid of Reference image
tic;
[gaussian_pyramid_1,gaussian_gradient_1,gaussian_angle_1]=...
build_gaussian_pyramid(image_11,nOctaves_1,dog_center_layer,sigma);                                                      
disp(['参考图像创建Gauss Pyramid花费时间是：',num2str(toc),'s']);

%% DOG Pyramid of Reference image
tic;
dog_pyramid_1=build_dog_pyramid(gaussian_pyramid_1,nOctaves_1,dog_center_layer);
disp(['参考图像创建DOG Pyramid花费时间是：',num2str(toc),'s']);

%% display the Gauss Pyramid,DOG Pyramid,gradient of Reference image
display_product_image(gaussian_pyramid_1,dog_pyramid_1,gaussian_gradient_1,...
        gaussian_angle_1,nOctaves_1,dog_center_layer,'Reference image');                              
 clear gaussian_pyramid_1;
 
%% Reference image DOG Pyramid extreme point detection
tic;
[key_point_array_1]=find_scale_space_extream...
(dog_pyramid_1,nOctaves_1,dog_center_layer,contrast_threshold_1,sigma,...
edge_threshold,gaussian_gradient_1,gaussian_angle_1);
disp(['参考图像关键点定位花费时间是：',num2str(toc),'s']);
clear dog_pyramid_1;

%% descriptor generation of the reference image 
tic;
[descriptors_1,locs_1]=calc_descriptors(gaussian_gradient_1,gaussian_angle_1,...
                                key_point_array_1,is_double_size,is_sift_or_log);
disp(['参考图像描述符生成花费时间是：',num2str(toc),'s']); 
clear gaussian_gradient_1;
clear gaussian_angle_1;

%% Gauss Pyramid of the image to be registered
tic;
[gaussian_pyramid_2,gaussian_gradient_2,gaussian_angle_2]=...
build_gaussian_pyramid(image_22,nOctaves_2,dog_center_layer,sigma);                                                                                                  
disp(['待配准图像创建Gauss Pyramid花费时间是：',num2str(toc),'s']);

%% DOG of the image to be registered
tic;
dog_pyramid_2=build_dog_pyramid(gaussian_pyramid_2,nOctaves_2,dog_center_layer);
disp(['待配准图像创建DOG Pyramid花费时间是：',num2str(toc),'s']);
display_product_image(gaussian_pyramid_2,dog_pyramid_2,gaussian_gradient_2,...
        gaussian_angle_2,nOctaves_2,dog_center_layer,'Image to be registered');                              
clear gaussian_pyramid_2;

%% Image to be registered DOG Pyramid extreme point detection
tic;
[key_point_array_2]=find_scale_space_extream...
(dog_pyramid_2,nOctaves_2,dog_center_layer,contrast_threshold_2,sigma,....
edge_threshold,gaussian_gradient_2,gaussian_angle_2);
disp(['待配准图像关键点定位花费时间是：',num2str(toc),'s']);
clear dog_pyramid_2;

%% descriptor generation of the Image to be registered
tic;
[descriptors_2,locs_2]=calc_descriptors(gaussian_gradient_2,gaussian_angle_2,...
                       key_point_array_2,is_double_size,is_sift_or_log);
disp(['待配准图像描述符生成花费时间是：',num2str(toc),'s']); 
clear gaussian_gradient_2;
clear gaussian_angle_2;

%% match
tic;
[solution,rmse,cor1,cor2]=...
    match(image_2, image_1,descriptors_2,locs_2,descriptors_1,locs_1,change_form);
disp(['特征点匹配花费时间是：',num2str(toc),'s']); 

%% Transformation parameters
tform=maketform('projective',solution');
[M,N,P]=size(image_1);
ff=imtransform(image_2,tform, 'XData',[1 N], 'YData',[1 M]);
button=figure;
subplot(1,2,1);
imshow(image_1);
title('Reference image');
subplot(1,2,2);
imshow(ff);
title('Image after registration');
str1=['.\save_image\','Image after registration','.jpg'];
saveas(button,str1,'jpg');

t2=clock;
disp(['Total time：',num2str(etime(t2,t1)),'s']); 

%% Display the detected feature points
[button1,button2]=showpoint_detected(image_1,image_2,locs_1,locs_2);
str1=['.\save_image\','Reference image detection point','.jpg'];
saveas(button1,str1,'jpg');
str1=['.\save_image\','detection point in the image to be registered','.jpg'];
saveas(button2,str1,'jpg');

%% Display feature point distribution
button=disp_points_distribute(locs_1,locs_2,cor2,cor2,...
                     nOctaves_1,nOctaves_2,dog_center_layer);
str1=['.\save_image\','feature point distribution','.jpg'];
saveas(button,str1,'jpg');

%% image fusion
image_fusion(image_1,image_2,solution);



