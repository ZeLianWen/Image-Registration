function display_product_image(gaussian_image,...
                                gaussian_gradient_second,...
                                gaussian_angle_second,...
                                gauss_harris_fun,str)
%%该函数主要完成程序中间产生的图像的显示和存储操作
%gaussian_image是尺度空间每层的的高斯图像
%gaussian_harris_fun是尺度空间每层的harris函数
%str表示是参考图像还是待配准图像
[~,~,Mmax]=size(gaussian_image);
temp=Mmax/4;
if((floor(temp)-temp)~=0)
    temp=temp+1;
end
f=figure;
for i=1:1:Mmax
    subplot(temp,4,i);
    imshow(mat2gray(gaussian_image(:,:,i)));
    title([str,'各向异性',num2str(i),'层']);
    str1=['.\save_image\',str,'各向异性空间',num2str(i),'层','.png'];
    imwrite(mat2gray(gaussian_image(:,:,i)),str1,'png');
end
str1=['.\save_image\',str,'各向异性空间','.jpg'];
saveas(f,str1,'jpg');


% 显示图像的Harris函数图像
f=figure;
for i=1:1:Mmax
    subplot(temp,4,i);
%     min_temp=min(min(gauss_harris_fun(:,:,i)));
%     max_temp=max(max(gauss_harris_fun(:,:,i)));
%     imshow((gauss_harris_fun(:,:,i)-min_temp)/(max_temp-min_temp));
    imshow(mat2gray(gauss_harris_fun(:,:,i)));
    title([str,'HARRIS函数',num2str(i),'层']);
end
str1=['.\save_image\',str,'HARRIS函数','.jpg'];
saveas(f,str1,'jpg');

%显示图像的二阶高斯差分梯度
f=figure;
for i=1:1:Mmax
    subplot(temp,4,i);
    imshow(mat2gray(gaussian_gradient_second(:,:,i)));
    title([str,'二阶梯度',num2str(i),'层']);
end
str1=['.\save_image\',str,'二阶差分梯度','.jpg'];
saveas(f,str1,'jpg');
    

%显示高斯图像二阶差分角度
f=figure;
for i=1:1:Mmax
    subplot(temp,4,i);
    imshow(mat2gray(gaussian_angle_second(:,:,i)));
    title([str,'二阶角度',num2str(i),'层']);
end
str1=['.\save_image\',str,'二阶差分角度','.jpg'];
saveas(f,str1,'jpg');

end

                            

