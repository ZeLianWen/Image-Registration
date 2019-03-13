function [descriptor]=calc_log_polar_descriptor(gradient,angle,x,y,scale,main_angle,d,n)
%gradient是特征点所在尺度的梯度矩阵
%angle是特征点所在尺度的角度矩阵
%x,y分别是特征点的水平坐标和竖直坐标
%scale是特征点的尺度
%main_angle是特征点的主方向
%d是对数极坐标中圆周分割数目，这里是8，就是对数极坐标中每隔48度一个分割
%n是方向直方图中0-360区间分割数目这里也是8

%% 计算特征点应该旋转的方向余弦和方向正弦
cos_t=cos(-main_angle/180*pi);
sin_t=sin(-main_angle/180*pi);

%% 获得邻域半径
[M,N]=size(gradient);
%限制圆形区域半径不要超过图像尺寸的一半
radius=round(min(12*scale,min(M,N)/3));%邻域半径是12*scale

%% 获得特征点邻域区域的位置，此时位置相对于矩阵左上角
radius_x_left=x-radius;
radius_x_right=x+radius;
radius_y_up=y-radius;
radius_y_down=y+radius;

%防止索引越界
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
%% 索引特征点周围区域像素的梯度和方向，区域是一个矩形，还不是圆形
sub_gradient=gradient(radius_y_up:radius_y_down,radius_x_left:radius_x_right);
sub_angle=angle(radius_y_up:radius_y_down,radius_x_left:radius_x_right);
sub_angle=round((sub_angle-main_angle)*n/360);
sub_angle(sub_angle<=0)=sub_angle(sub_angle<=0)+n;
sub_angle(sub_angle==0)=n;

%% 特征点周围像素旋转后位置,此时的位置以特征点(x,y)为中心
X=-(x-radius_x_left):1:(radius_x_right-x);
Y=-(y-radius_y_up):1:(radius_y_down-y);
[XX,YY]=meshgrid(X,Y);
c_rot=XX*cos_t-YY*sin_t;%旋转后的x坐标的位置,此时的位置中心是特征点位置x,y
r_rot=XX*sin_t+YY*cos_t;%旋转后的y坐标的位置，此时的位置中心是特征点位置x,y

%% 计算旋转后的位置属于哪一个对数极坐标网格
log_angle=atan2(r_rot,c_rot);%得到周围像素对数极坐标角度
log_angle=log_angle/pi*180;%转换到-180-180
log_angle(log_angle<0)=log_angle(log_angle<0)+360;%转换到0-360度范围
log_amplitude=log2(sqrt(c_rot.^2+r_rot.^2));%得到周围像素对数极坐标半径

%这里角度按照45度一个区间划分为5份
log_angle=round(log_angle*d/360);
log_angle(log_angle<=0)=log_angle(log_angle<=0)+d;
log_angle(log_angle>d)=log_angle(log_angle>d)-d;

%幅度幅度按照比例关系划分为3份
r1=log2(radius*0.25);
r2=log2(radius*0.73);
log_amplitude(log_amplitude<=r1)=1;
log_amplitude(log_amplitude>r1 & log_amplitude<=r2)=2;
log_amplitude(log_amplitude>r2)=3;

%描述符生成
temp_hist=zeros(1,(2*d+1)*n);%这里描述符是136维向量
[row,col]=size(log_angle);%得到矩形区域的高度和宽度
for i=1:1:row
    for j=1:1:col
        %确定圆形区域范围
       % if(((i-center_y)^2+(j-center_x)^2)<=radius^2)
            angle_bin=log_angle(i,j);%对数极坐标角度落在那个区域
            amplitude_bin=log_amplitude(i,j);%对数极坐标半径落在那个区域
            bin_vertical=sub_angle(i,j);%三维直方图中第三维范围
            Mag=sub_gradient(i,j);%这一点的梯度值
             
            %开始生成136维描述符
            if(amplitude_bin==1)
                temp_hist(bin_vertical)=temp_hist(bin_vertical)+Mag;
            else
                temp_hist(((amplitude_bin-2)*d+angle_bin-1)*n+bin_vertical+n)=...
                    temp_hist(((amplitude_bin-2)*d+angle_bin-1)*n+bin_vertical+n)+Mag;
            end
        %end
    end
end

temp_hist=temp_hist/sqrt(temp_hist*temp_hist');
temp_hist(temp_hist>0.2)=0.2;
temp_hist=temp_hist/sqrt(temp_hist*temp_hist');
descriptor=temp_hist;

end


  



         
      
      














                                             
                                             
                                             
                                             
                                           

