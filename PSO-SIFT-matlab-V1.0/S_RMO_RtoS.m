function [S,RMO,delat_x,delat_y,scale_hist,scale_x,angle_hist,...
    angle_x,x_hist,x_x,y_hist,y_x]=S_RMO_RtoS(cor1,cor2)

smooth='true';
%% Computational scale ratio histogram£¬S=Reference/To be registered
ratio_cor_scale=cor1(:,1)./cor2(:,1);
ratio_cor_scale=ratio_cor_scale';
min_scale=min(ratio_cor_scale);
max_scale=max(ratio_cor_scale);
unit_scale=0.075;
scale_bin=round((max_scale-min_scale)/unit_scale);
scale_hist=zeros(1,scale_bin+1);
ratio_cor_scale_1=round((ratio_cor_scale-min_scale)/unit_scale);
[~,num_scale]=size(ratio_cor_scale_1);
for i=1:1:num_scale
    scale_hist(ratio_cor_scale_1(1,i)+1)=scale_hist(ratio_cor_scale_1(1,i)+1)+1;
end

%Interpolation histogram
if(strcmp(smooth,'true'))
    n=scale_bin+1;
    if(n>=5)
        hist=zeros(1,n);
        hist(1)=(scale_hist(n-1)+scale_hist(3))/16+...
            4*(scale_hist(n)+scale_hist(2))/16+scale_hist(1)*6/16;
        hist(2)=(scale_hist(n)+scale_hist(4))/16+...
            4*(scale_hist(1)+scale_hist(3))/16+scale_hist(2)*6/16;

        hist(3:n-2)=(scale_hist(1:n-4)+scale_hist(5:n))/16+...
        4*(scale_hist(2:n-3)+scale_hist(4:n-1))/16+scale_hist(3:n-2)*6/16;

        hist(n-1)=(scale_hist(n-3)+scale_hist(1))/16+...
            4*(scale_hist(n-2)+scale_hist(n))/16+scale_hist(n-1)*6/16;
        hist(n)=(scale_hist(n-2)+scale_hist(2))/16+...
            4*(scale_hist(n-1)+scale_hist(1))/16+scale_hist(n)*6/16;
        scale_hist=hist;
    end
end

x_length=size(scale_hist,2);
for i=1:1:x_length
    scale_x(1,i)=(i-1)*unit_scale+min_scale;
end
index=find(scale_x<=6);
scale_x=scale_x(index);
scale_hist=scale_hist(index);

%Find the maximum value in the histogram
index_scale=find(scale_hist==max(scale_hist));
n=size(scale_hist,2);
if(strcmp(smooth,'true'))
    k1=index_scale-1;k2=index_scale+1;
    k1(k1==0)=n;
    k2(k2==n+1)=1;
    index_scale=index_scale+0.5*(scale_hist(k1)-scale_hist(k2))/...
        (scale_hist(k1)+scale_hist(k2)-2*scale_hist(index_scale));
end
S=(index_scale-1)*unit_scale+scale_x(1);
S=mean(S);

%% Angle difference histogram
%Relative principal direction,RMO=Reference -To be registered
diff_cor_angle=cor1(:,2)-cor2(:,2);
diff_cor_angle=diff_cor_angle';
min_diff_cor_angle=min(diff_cor_angle);
max_diff_cor_angle=max(diff_cor_angle);
unit_angle=9;
diff_cor_angle_1=round((diff_cor_angle-min_diff_cor_angle)/unit_angle);
angle_bin=round((max_diff_cor_angle-min_diff_cor_angle)/unit_angle);
angle_hist=zeros(1,angle_bin+1);
[~,num]=size(diff_cor_angle_1);
for i=1:1:num
    angle_hist(diff_cor_angle_1(1,i)+1)=angle_hist(diff_cor_angle_1(1,i)+1)+1;
end

%Interpolation histogram
if(strcmp(smooth,'true'))
    n=angle_bin+1;
    if(n>5)
        hist=zeros(1,n);
        hist(1)=(angle_hist(n-1)+angle_hist(3))/16+...
            4*(angle_hist(n)+angle_hist(2))/16+angle_hist(1)*6/16;
        hist(2)=(angle_hist(n)+angle_hist(4))/16+...
            4*(angle_hist(1)+angle_hist(3))/16+angle_hist(2)*6/16;

        hist(3:n-2)=(angle_hist(1:n-4)+angle_hist(5:n))/16+...
        4*(angle_hist(2:n-3)+angle_hist(4:n-1))/16+angle_hist(3:n-2)*6/16;

        hist(n-1)=(angle_hist(n-3)+angle_hist(1))/16+...
            4*(angle_hist(n-2)+angle_hist(n))/16+angle_hist(n-1)*6/16;
        hist(n)=(angle_hist(n-2)+angle_hist(2))/16+...
            4*(angle_hist(n-1)+angle_hist(1))/16+angle_hist(n)*6/16;
        angle_hist=hist;
    end
end

y_length=size(angle_hist,2);
for i=1:1:y_length
    angle_x(1,i)=(i-1)*unit_angle+min_diff_cor_angle;
end

index_angle=find(angle_hist==max(angle_hist));
temp_index=index_angle-5:1:index_angle+5;

n=size(angle_hist,2);
if(strcmp(smooth,'true'))
    k1=index_angle-1;k2=index_angle+1;
    k1(k1==0)=n;
    k2(k2==n+1)=1;
    index_angle=index_angle+0.5*(angle_hist(k1)-angle_hist(k2))/...
        (angle_hist(k1)+angle_hist(k2)-2*angle_hist(index_angle));
end
RMO=(index_angle-1)*unit_angle+angle_x(1);

if(RMO>0)
    RMO=[RMO,RMO-360];
else
    RMO=[RMO,RMO+360];
end

%% position difference
cor1_xy=cor1(:,[3,4]);
cor2_xy=cor2(:,[3,4]);
temp_T=[S*cos(RMO(1)/180*pi),-S*sin(RMO(1)/180*pi);...
    S*sin(RMO(1)/180*pi),S*cos(RMO(1)/180*pi)];
diff_xy=cor1_xy'-temp_T*cor2_xy';

diff_x=diff_xy(1,:);
min_diff_x=min(diff_x);
max_diff_x=max(diff_x);
unit_x=7.5;
diff_x_1=round((diff_x-min_diff_x)/unit_x);
x_bin=round((max_diff_x-min_diff_x)/unit_x);
x_hist=zeros(1,x_bin+1);
[~,num]=size(diff_x_1);
for i=1:1:num
    x_hist(diff_x_1(1,i)+1)=x_hist(diff_x_1(1,i)+1)+1;
end

%interpolation histogram
if(strcmp(smooth,'true'))
    n=x_bin+1;
    if(n>5)
        hist=zeros(1,n);
        hist(1)=(x_hist(n-1)+x_hist(3))/16+...
            4*(x_hist(n)+x_hist(2))/16+x_hist(1)*6/16;
        hist(2)=(x_hist(n)+x_hist(4))/16+...
            4*(x_hist(1)+x_hist(3))/16+x_hist(2)*6/16;

        hist(3:n-2)=(x_hist(1:n-4)+x_hist(5:n))/16+...
        4*(x_hist(2:n-3)+x_hist(4:n-1))/16+x_hist(3:n-2)*6/16;

        hist(n-1)=(x_hist(n-3)+x_hist(1))/16+...
            4*(x_hist(n-2)+x_hist(n))/16+x_hist(n-1)*6/16;
        hist(n)=(x_hist(n-2)+x_hist(2))/16+...
            4*(x_hist(n-1)+x_hist(1))/16+x_hist(n)*6/16;
        x_hist=hist;
    end
end

y_length=size(x_hist,2);
for i=1:1:y_length
    x_x(1,i)=(i-1)*unit_x+min_diff_x;
end
index=find(x_x>-1000 & x_x<1000);
x_x=x_x(index);
x_hist=x_hist(index);

index_x=find(x_hist==max(x_hist));
n=size(x_hist,2);
if(strcmp(smooth,'true'))
    k1=index_x-1;k2=index_x+1;
    k1(k1==0)=n;
    k2(k2==n+1)=1;
    index_x=index_x+0.5*(x_hist(k1)-x_hist(k2))/...
        (x_hist(k1)+x_hist(k2)-2*x_hist(index_x));
end
delat_x=(index_x-1)*unit_x+x_x(1);

diff_y=diff_xy(2,:);
min_diff_y=min(diff_y);
max_diff_y=max(diff_y);
unit_y=7.5;
diff_y_1=round((diff_y-min_diff_y)/unit_y);
y_bin=round((max_diff_y-min_diff_y)/unit_y);
y_hist=zeros(1,y_bin+1);
[~,num]=size(diff_y_1);
for i=1:1:num
    y_hist(diff_y_1(1,i)+1)=y_hist(diff_y_1(1,i)+1)+1;
end

if(strcmp(smooth,'true'))
    n=y_bin+1;
    if(n>5)
    hist=zeros(1,n);
        hist(1)=(y_hist(n-1)+y_hist(3))/16+...
            4*(y_hist(n)+y_hist(2))/16+y_hist(1)*6/16;
        hist(2)=(y_hist(n)+y_hist(4))/16+...
            4*(y_hist(1)+y_hist(3))/16+y_hist(2)*6/16;

        hist(3:n-2)=(y_hist(1:n-4)+y_hist(5:n))/16+...
        4*(y_hist(2:n-3)+y_hist(4:n-1))/16+y_hist(3:n-2)*6/16;

        hist(n-1)=(y_hist(n-3)+y_hist(1))/16+...
            4*(y_hist(n-2)+y_hist(n))/16+y_hist(n-1)*6/16;
        hist(n)=(y_hist(n-2)+y_hist(2))/16+...
            4*(y_hist(n-1)+y_hist(1))/16+y_hist(n)*6/16;
        y_hist=hist;
    end
end

y_length=size(y_hist,2);
for i=1:1:y_length
    y_x(1,i)=(i-1)*unit_y+min_diff_y;
end
index=find(y_x>-1000 & y_x<1000);
y_x=y_x(index);
y_hist=y_hist(index);

index_y=find(y_hist==max(y_hist));
n=size(y_hist,2);
if(strcmp(smooth,'true'))
    k1=index_y-1;k2=index_y+1;
    k1(k1==0)=n;
    k2(k2==n+1)=1;
    index_y=index_y+0.5*(y_hist(k1)-y_hist(k2))/...
        (y_hist(k1)+y_hist(k2)-2*y_hist(index_y));
end
delat_y=(index_y-1)*unit_y+y_x(1);

end



