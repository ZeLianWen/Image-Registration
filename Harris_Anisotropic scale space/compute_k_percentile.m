function [k_percentile]=compute_k_percentile(gradient_x,gradient_y,perc,nbin)
%该函数计算一个对比度参数k,这个对比度参数用于计算扩散系数
%gradient_x是水平方向的梯度，gradient_y是竖直方向的梯度
%perc是梯度直方图的百分位数，默认取值是0.7，k的取值根据这个百分位数确定
%nbin是直方图bin的个数，默认是300
%传导系数函数都是k的增函数，因此对于相同的梯度值，如果k值较大，则传导系数值较大
%因此扩散大，平滑严重，因此可以看出，如果要保留细节需要较小的k值

%忽略边界计算梯度的最大值
gradient=sqrt(gradient_x.^2+gradient_y.^2);
[M,N]=size(gradient);
temp_gradient=gradient(2:M-1,2:N-1);
max_gradient=max(max(temp_gradient));

%忽略边界计算直方图
temp_gradient=temp_gradient(temp_gradient>0);
temp_gradient=floor(temp_gradient*nbin/max_gradient);
temp_gradient(temp_gradient==nbin)=nbin-1;
[M1,N1]=size(temp_gradient);
sum_pix=M1*N1;%非零像素梯度个数
hist=zeros(1,nbin);%直方图初始化

%计算直方图
for i=1:1:M1
    for j=1:1:N1
        hist(temp_gradient(i,j)+1)=hist(temp_gradient(i,j)+1)+1;
    end
end

%直方图百分位
nthreshold=perc*sum_pix;
nelements=0;
temp_i=0;
for i=1:1:nbin
    nelements=nelements+hist(i);
    if(nelements>=nthreshold)
        temp_i=i;
        break;
    end
end
 
if(temp_i==0)
    k_percentile=0.03;
else
    k_percentile=max_gradient*(temp_i)/nbin;
end

end















