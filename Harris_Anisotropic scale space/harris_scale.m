function [harris_function,gradient,angle]=harris_scale(nonelinear_space,d,sigma,ratio)

     
%% 初始分配存储空间
[M,N,P]=size(nonelinear_space);
harris_function=zeros(M,N,P);
gradient=zeros(M,N,P);
angle=zeros(M,N,P);

%%
h=[-1,0,1;-2,0,2;-1,0,1];%差分滤波模板

for i=1:1:P    
    %一阶差分梯度和一阶差分角度
    gradient_x_1=imfilter(nonelinear_space(:,:,i),h,'replicate');
    gradient_y_1=imfilter(nonelinear_space(:,:,i),h','replicate');
    gradient_1=sqrt(gradient_x_1.^2+gradient_y_1.^2);
    
    %新的梯度
    gradient_x_2=imfilter(gradient_1,h,'replicate');
    gradient_y_2=imfilter(gradient_1,h','replicate');
    gradient_2=sqrt(gradient_x_2.^2+gradient_y_2.^2);
    angle_2=atan2(gradient_y_2,gradient_x_2);
    angle_2=angle_2*180/pi;%转换到-180到180
    angle_2(angle_2<0)=angle_2(angle_2<0)+360;%转换到0-360
    gradient(:,:,i)=gradient_2;
    angle(:,:,i)=angle_2;
    
    %构建harris函数
    cur_scale=sigma*ratio^(i-1);%当前层的尺度
    Csh_11=cur_scale^2*gradient_x_2.*gradient_x_2;
    Csh_12=cur_scale^2*gradient_x_2.*gradient_y_2;
    Csh_22=cur_scale^2*gradient_y_2.*gradient_y_2;


    gaussian_sigma=cur_scale*sqrt(2);
    temp=round(2*gaussian_sigma);
    gaussian_width=2*temp+1;
    W=fspecial('gaussian',[gaussian_width gaussian_width],gaussian_sigma);
    %圆形区域
    [a,b]=meshgrid(1:gaussian_width,1:gaussian_width);
    index=find(((a-temp-1)^2+(b-temp-1)^2)>temp^2);
    W(index)=0;
    
    Csh_11=imfilter(Csh_11,W,'replicate');
    Csh_12=imfilter(Csh_12,W,'replicate');
    Csh_21=Csh_12;
    Csh_22=imfilter(Csh_22,W,'replicate');
    
    % 生成HARRIS函数
    harris_function(:,:,i)=Csh_11.*Csh_22-Csh_21.*Csh_12-d*(Csh_11+Csh_22).^2;
end

end



































