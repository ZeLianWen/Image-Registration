function [descriptor]=calc_sift_descriptor(gradient,angle,x,y,scale,main_angle,d,n)
%该函数计算某一点的描述子
%gradient是特征点所在的尺度的梯度平面
%angle是特征点所在的尺度的角度平面
%x,y是特征点的水平和垂直坐标
%scale是特征点所在的尺度值
%main_angle是特征点的主方向
%d是区域宽度，默认区域大小4X4
%n是每个子区域直方图个数，默认是8

%% 初始化参数
histlen=(d+2)*(d+2)*(n+2);
hist=zeros(1,histlen);

%% 特征点方向的余弦和正弦
cos_t=cos(-main_angle/180*pi);
sin_t=sin(-main_angle/180*pi);

%高斯加权中e的指数部分
 exp_scale=-1/(0.5*d*d);%高斯函数标准差是0.5*d
%特征点子区域的宽度是3*d
hist_width=3*scale;
%特征点邻域半径
radius=round(hist_width*(d+1)*1.414/2);
%%避免邻域过大
[M,N]=size(gradient);%%特征点所在图像的长和宽
radius=min(radius,round(sqrt(M*M+N*N)));
%%归一化处理
cos_t=cos_t/hist_width;
sin_t=sin_t/hist_width;

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

%% 索引特征点周围区域像素的梯度和方向
sub_gradient=gradient(radius_y_up:radius_y_down,radius_x_left:radius_x_right);
sub_angle=angle(radius_y_up:radius_y_down,radius_x_left:radius_x_right);

%% 计算旋转后的位置计算高斯权重，
X=-(x-radius_x_left):1:(radius_x_right-x);
Y=-(y-radius_y_up):1:(radius_y_down-y);
[XX,YY]=meshgrid(X,Y);
c_rot=XX*cos_t-YY*sin_t;
r_rot=XX*sin_t+YY*cos_t;
Rbin=r_rot+d/2-0.5;
Cbin=c_rot+d/2-0.5;
gaussian_weight=exp((c_rot.^2+r_rot.^2)*exp_scale);
[row,col]=size(sub_angle);

%% 遍历所有邻域像素
for i=1:1:row
    for j=1:1:col
        %%得到上面计算的d*d邻域的坐标
        rbin=Rbin(i,j);cbin=Cbin(i,j);
        if(rbin<=-1 || rbin>=d || cbin<=-1 || cbin>=d)
            continue;
        end
        %得到幅角所属8份中的某一个等分
        obin=(sub_angle(i,j)-main_angle)*(n/360);
        %高斯加权后的幅度
        %mag=sub_gradient(i,j)*gaussian_weight(i,j);
        mag=sub_gradient(i,j);%这里不用高斯权重
        
        %%向下取整,分别表示属于哪个正方体
        r0=floor(rbin);
        c0=floor(cbin);
        o0=floor(obin);%向下取整

        %%三维坐标的小数部分
        rbin=rbin-r0;%正数
        cbin=cbin-c0;%正数
        obin=obin-o0;

        if(o0<0)
            o0=o0+n;
        end
        if(o0>=n)
            o0=o0-n;
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        v_r1 = mag*rbin; v_r0 = mag - v_r1;
        v_rc11 = v_r1*cbin; v_rc10 = v_r1 - v_rc11;
        v_rc01 = v_r0*cbin; v_rc00 = v_r0 - v_rc01;
        v_rco111 = v_rc11*obin;v_rco110 = v_rc11 -  v_rco111;
        v_rco101 = v_rc10*obin; v_rco100 = v_rc10 - v_rco101;
        v_rco011 = v_rc01*obin; v_rco010 = v_rc01 -  v_rco011;
        v_rco001 = v_rc00*obin; v_rco000 = v_rc00 -  v_rco001;
        %得到该像素点在三维直方图中的索引
        idx = ((r0+1)*(d+2) + c0+1)*(n+2) + o0+1;
        %8 个顶点对应于坐标平移前的 8 个直方图的正方体，对其进行累加求和
        hist(idx) =hist(idx)+ v_rco000;
        hist(idx+1) =hist(idx+1)+ v_rco001;
        hist(idx+(n+2))=hist(idx+(n+2)) +v_rco010;
        hist(idx+(n+3)) =hist(idx+(n+3))+ v_rco011;
        hist(idx+(d+2)*(n+2)) =hist(idx+(d+2)*(n+2))+v_rco100;
        hist(idx+(d+2)*(n+2)+1) =hist(idx+(d+2)*(n+2)+1)+ v_rco101;
        hist(idx+(d+3)*(n+2)) =hist(idx+(d+3)*(n+2))+ v_rco110;
        hist(idx+(d+3)*(n+2)+1) =hist(idx+(d+3)*(n+2)+1) +v_rco111;
    end
end

 %% 由于圆周循环的特性，对计算以后幅角小于 0 度或大于 360 度的值重新进行调整，使
 %其在 0～360 度之间
 descriptor=zeros(1,n*d*d);
    for i=0:d-1
        for j=0:d-1
             idx = ((i+1)*(d+2) + (j+1))*(n+2)+1;
             hist(idx) =hist(idx)+ hist(idx+n);
             hist(idx+1) =hist(idx+1)+ hist(idx+n+1);
            for k=1:n
                descriptor((i*d + j)*n + k) = hist(idx+k-1);
            end
        end
    end

%% 特征向量归一化处理
descriptor=descriptor/sqrt(descriptor*descriptor');
descriptor(descriptor>=0.2)=0.2;%%截断为0.2
descriptor=descriptor/sqrt(descriptor*descriptor');%%再次归一化

end


         
      
      














                                             
                                             
                                             
                                             
                                           

