function [hist,max_value]=calculate_oritation_hist(x,y,scale,gradient,angle,n)
%% 该函数计算特征点的方向直方图并获得该点的主方向
%x,y是特征点所在的层的位置
%scale是特征点的所在的尺度，决定了邻域圆的大小和高斯加权函数的标准差
%gradient是特征点所在层的所有点的梯度
%angle是特征点所在层的所有点的角度，这里直接访问索引，加快了速度
%n是直方图bin个数这里按照SIFT中36个
%输出hist是直方图，max_value是直方图的最大值，代表主方向

%% 参数初始化，包括区域半径
radius=round(6*scale);%邻域圆半径
sigma=2*scale;%高斯加权函数标准差
[M,N,~]=size(gradient);
radius_x_left=x-radius;
radius_x_right=x+radius;
radius_y_up=y-radius;
radius_y_down=y+radius;

%% 防止索引越界
if(radius_x_left<=0)
    radius_x_left=1;
end
if(radius_x_right>N)
    radius_x_right=N;
end
if(radius_y_up<=0)
    radius_y_up=1;
end
if(radius_y_down>M)
    radius_y_down=M;
end
%% 此时特征点x,y在矩形中的位置是
center_x=x-radius_x_left+1;
center_y=y-radius_y_up+1;

%% 索引特征点周围区域像素的梯度和方向
sub_gradient=gradient(radius_y_up:radius_y_down,radius_x_left:radius_x_right);
sub_angle=angle(radius_y_up:radius_y_down,radius_x_left:radius_x_right);

%% 计算高斯权重，
X=-(x-radius_x_left):1:(radius_x_right-x);
Y=-(y-radius_y_up):1:(radius_y_down-y);
[XX,YY]=meshgrid(X,Y);
gaussian_weight=1/(sqrt(2*pi)*sigma)*exp(-(XX.^2+YY.^2)/(2*sigma^2));

%W=sub_gradient.*gaussian_weight;
W=sub_gradient;%无高斯权重
bin=round(sub_angle*n/360);%确定属于直方图的哪个BIN

%直方图圆周循环
bin(bin>=n)=bin(bin>=n)-n;
bin(bin<0)=bin(bin<0)+n;

%% 计算直方图
temp_hist=zeros(1,n);
[row,col]=size(sub_angle);
for i=1:1:row
    for j=1:1:col
        %限制在圆形区域
        if(((i-center_y)^2+(j-center_x)^2)<=radius^2)
            temp_hist(bin(i,j)+1)=temp_hist(bin(i,j)+1)+W(i,j);
        end
    end
end

%% 平滑直方图
%%temp_hist(-1)=temp_hist(35)
%%temp_hist(0)=temp_hist(36)
hist=zeros(1,n);
hist(1)=(temp_hist(35)+temp_hist(3))/16+...
    4*(temp_hist(36)+temp_hist(2))/16+temp_hist(1)*6/16;
hist(2)=(temp_hist(36)+temp_hist(4))/16+...
    4*(temp_hist(1)+temp_hist(3))/16+temp_hist(2)*6/16;
% for j=3:1:n-2
%     hist(j)=(temp_hist(j-2)+temp_hist(j+2))/16+...
%     4*(temp_hist(j-1)+temp_hist(j+1))/16+temp_hist(j)*6/16;
% end

%效率高
hist(3:n-2)=(temp_hist(1:n-4)+temp_hist(5:n))/16+...
4*(temp_hist(2:n-3)+temp_hist(4:n-1))/16+temp_hist(3:n-2)*6/16;

hist(n-1)=(temp_hist(n-3)+temp_hist(1))/16+...
    4*(temp_hist(n-2)+temp_hist(n))/16+temp_hist(n-1)*6/16;
hist(n)=(temp_hist(n-2)+temp_hist(2))/16+...
    4*(temp_hist(n-1)+temp_hist(1))/16+temp_hist(n)*6/16;

%%计算直方图的主峰值
max_value=max(hist);



