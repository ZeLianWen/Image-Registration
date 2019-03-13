function [key_point_array]=find_scale_extreme...
    (sar_harris_function,...%各个尺度计算得到的SAR-HARRIS函数
    threshold,...%角点检测阈值，默认是0.04
    sigma,...%第一层的尺度，默认是2
    ratio,...%连续两层的尺度比
    gradient,...%各层梯度
    angle,...%各层的角度
    first_layer)

%% 该函数的作用是根据生成的SAR-Harris尺度空间和SAR-Harris函数寻找局部极值点
%并利用双线性插值精确定位极值点位置，输入参数sar_harris_function是sar_harris_function
%尺度空间，key_point_array是一个结构体数组，保存特征点的信息，
%输入threshold是一个用于SAR-HARRIS的阈值
%sigma是底层尺度,ratio是尺度比例
%输入harris_fun是各层的Harris函数

%% 开始 
[M,N,num]=size(sar_harris_function);
BORDER_WIDTH=2;%边界常数
HIST_BIN=36;%直方图个数，这里36个直方图，每10度一个
SIFT_ORI_PEAK_RATIO=0.9;
key_number=0;%计数特征点个数
%key_point_array=zeros(M*N*num,6);
key_point_array=zeros(M,6);
%% key_point_array是一个二维矩阵用于保存所有特征点信息，包括位置x,y,所在的层的
% 尺度scale,所在的层layer,特征点主方向的角度angle和累加的梯度gradient
for i=first_layer:1:num%从第二层开始
    temp_current=sar_harris_function(:,:,i);
    % 这里给出了本层和上下两层对比的结果，显示了harris函数很少在连续层之间取得极大值
    gradient_current=gradient(:,:,i);%获得特征点所在层的梯度
    angle_current=angle(:,:,i);%获得特征点所再次的角度
    
    for j=BORDER_WIDTH:1:M-BORDER_WIDTH%行
        for k=BORDER_WIDTH:1:N-BORDER_WIDTH%列
            temp=temp_current(j,k);
            if(temp>threshold &&...
                temp>temp_current(j-1,k-1) && temp>temp_current(j-1,k) && temp>temp_current(j-1,k+1) &&...
                temp>temp_current(j,k-1) && temp>temp_current(j,k+1) &&...
                temp>temp_current(j+1,k-1) && temp>temp_current(j+1,k) && temp>temp_current(j+1,k+1))
                
               scale=sigma*ratio^(i-1);
                %% 计算该特征点的直方图和最大值方向，即主方向
                [hist,max_value]=calculate_oritation_hist(k,j,scale,...
                        gradient_current,angle_current,HIST_BIN);                
                mag_thr=max_value*SIFT_ORI_PEAK_RATIO;%%辅助方向大小   
                for kk=1:1:HIST_BIN%k是当前直方图的索引
                    if(kk==1)
                        k1=HIST_BIN;
                    else
                        k1=kk-1;
                    end 
                    if(kk==HIST_BIN)%k2是当前直方图右边的索引
                        k2=1;
                    else
                        k2=kk+1;
                    end
                 %%当前数值大于前后数值，并且大于主峰的0.8倍
                    if(hist(kk)>hist(k1) && hist(kk)>hist(k2)...
                         && hist(kk)>mag_thr)
                        bin=kk-1+0.5*(hist(k1)-hist(k2))/(hist(k1)+hist(k2)-2*hist(kk));
                        if(bin<0)
                            bin=HIST_BIN+bin;
                        elseif(bin>=HIST_BIN)
                            bin=bin-HIST_BIN;
                        end
                        %保存特征点
                        key_number=key_number+1;
                        key_point_array(key_number,1)=k;%特征点列坐标，就是x
                        key_point_array(key_number,2)=j;%特征点行坐标，就是y
                        key_point_array(key_number,3)=sigma*ratio^(i-1);%所在层的尺度
                        key_point_array(key_number,4)=i;%所在层
                        key_point_array(key_number,5)=(360/HIST_BIN)*bin;%0-360度
                        key_point_array(key_number,6)=hist(kk);%梯度
                    end
                end
                
            end
        end
    end
end
key_point_array=key_point_array(1:key_number,:);
end

