function [nonelinear_space]=create_nonlinear_scale_space(image,sigma_1,sigma_2,...
                                                        ratio,layers,nbin,perc,...
                                                        which_diff,is_auto)
%该函数创建非线性尺度空间
%image是输入的原始图像，这里应该是浮点类型的数据，范围是0-1
%sigma_1是第一层的图像的尺度，默认是1.6，尺度空间第一层的图像由image经过标准差
%是sigma_1的高斯滤波得到
%sigma_2是每次计算下一层图像之前，对之前层图像的高斯平滑标准差,默认是1不变
%nbin是计算对比度因子时候需要的常数，默认是300
%perc是计算对比度因子的梯度百分位，这里默认是0.7
%ratio是相隔两层的尺度比
%layers是构建的尺度空间的层数，这里没有使用下采样操作
%which_diff决定了使用哪个函数计算传到系数取值是1,2,3
%nonelinear_space是构建的尺度空间图像

%%
[M,N]=size(image);
nonelinear_space=zeros(M,N,layers);

%首先对输入图像进行高斯平滑
windows_size=2*round(2*sigma_1)+1;
W=fspecial('gaussian',[windows_size windows_size],sigma_1);
image=imfilter(image,W,'replicate');%base_image的尺度是sigma_1
nonelinear_space(:,:,1)=image;%base_image作为尺度空间的第一层图像

%获取滤波器类型
h=[-1,0,1;-2,0,2;-1,0,1];%差分滤波模板

%计算每层的尺度
sigma=zeros(1,layers);
for i=1:1:layers
    sigma(i)=sigma_1*ratio^(i-1);%每层的尺度
end

%% 构建非线性尺度空间
for i=2:1:layers
    %之前层的非线性扩散后的的图像,计算梯度之前进行平滑的目的是为了消除噪声
    prev_image=nonelinear_space(:,:,i-1);
    windows_size=2*round(2*sigma_2)+1;
    W=fspecial('gaussian',[windows_size,windows_size],sigma_2);
    prev_smooth=imfilter(prev_image,W,'replicate');
    
    %计算之前层被平滑图像的x和y方向的一阶梯度
    Lx=imfilter(prev_smooth,h,'replicate');
    Ly=imfilter(prev_smooth,h','replicate');
    %每次迭代时候都需要更新对比度因子k
    if(strcmp(is_auto,'NO'))
        [k_percentile]=compute_k_percentile(Lx,Ly,perc,nbin);
    elseif(strcmp(is_auto,'YES'))
        [k_percentile]=compute_k_percentile_auto(Lx,Ly,perc);
    end
    if(which_diff==1)
        [diff_c]=pm_g1(Lx,Ly,k_percentile);
    elseif(which_diff==2)
        [diff_c]=pm_g2(Lx,Ly,k_percentile);
    else
        [diff_c]=weickert_diffusivity(Lx,Ly,k_percentile);
    end
    
    %计算当前层尺度图像
    step=1/2*(sigma(i)^2-sigma(i-1)^2);%步长因子
    nonelinear_space(:,:,i)=AOS(prev_image,step,diff_c);
end
end


%% 扩散系数计算函数1
function [g1]=pm_g1(Lx,Ly,k)
%该函数计算PM传导系数g1,Lx是水平方向的导数，Ly是竖直方向的导数
%k是一个对比度因子参数，k的取值一般根据统计所得
%g1=exp(-(Lx^2+Ly^2)/k^2)

g1=exp(-(Lx.^2+Ly.^2)/k^2);

end

%% 扩散系数计算函数2
function [g2]=pm_g2(Lx,Ly,k)
%该函数计算PM方程的扩散系数，第二种方法
%Lx和Ly分别是水平方向和竖直方向的差分，k是对比度因子参数
%g2=1/(1+(Lx^2+Ly^2)/(k^2)),这里k值的确定一般是通过统计方法得到

g2=1./(1+(Lx.^2+Ly.^2)/(k^2));
end

%% 扩散系数计算函数3
function [g3]=weickert_diffusivity(Lx,Ly,k)
%这个函数计算weickert传导系数
%Lx和Ly是水平方向和竖直方向的一阶差分梯度，k是对比度系数
%k的取值一般通过统计方法得到
%g3=1-exp(-3.315/((Lx^2+Ly^2)/k^4))

g3=1-exp(-3.315./((Lx.^2+Ly.^2).^4/k^8));
end















