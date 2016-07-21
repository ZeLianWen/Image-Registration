function [hist,max_value]=calculate_oritation_hist(x,y,scale,gradient,angle,n)

radius=round(6*scale);%% Neighborhood radius
sigma=1.5*scale;
[M,N,~]=size(gradient);
radius_x_left=x-radius;
radius_x_right=x+radius;
radius_y_up=y-radius;
radius_y_down=y+radius;

%% To prevent the index out of bounds
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

sub_gradient=gradient(radius_y_up:radius_y_down,radius_x_left:radius_x_right);
sub_angle=angle(radius_y_up:radius_y_down,radius_x_left:radius_x_right);

%% Gauss weight calculation，
X=-(x-radius_x_left):1:(radius_x_right-x);
Y=-(y-radius_y_up):1:(radius_y_down-y);
[XX,YY]=meshgrid(X,Y);

% gaussian_weight=1/(sqrt(2*pi)*sigma)*exp(-(XX.^2+YY.^2)/(2*sigma^2));
% W=sub_gradient.*gaussian_weight;%有高斯权重

W=sub_gradient;%No Gauss weight

bin=round(sub_angle*n/360);

bin(bin>=n)=bin(bin>=n)-n;
bin(bin<0)=bin(bin<0)+n;


%% Computed histogram
temp_hist=zeros(1,n);
[row,col]=size(sub_angle);
for i=1:1:row
    for j=1:1:col
        temp_hist(bin(i,j)+1)=temp_hist(bin(i,j)+1)+W(i,j);
    end
end

%% Smoothed histogram
%%temp_hist(-1)=temp_hist(35)
%%temp_hist(0)=temp_hist(36)
hist=zeros(1,n);
hist(1)=(temp_hist(35)+temp_hist(3))/16+...
    4*(temp_hist(36)+temp_hist(2))/16+temp_hist(1)*6/16;
hist(2)=(temp_hist(36)+temp_hist(4))/16+...
    4*(temp_hist(1)+temp_hist(3))/16+temp_hist(2)*6/16;

hist(3:n-2)=(temp_hist(1:n-4)+temp_hist(5:n))/16+...
4*(temp_hist(2:n-3)+temp_hist(4:n-1))/16+temp_hist(3:n-2)*6/16;

hist(n-1)=(temp_hist(n-3)+temp_hist(1))/16+...
    4*(temp_hist(n-2)+temp_hist(n))/16+temp_hist(n-1)*6/16;
hist(n)=(temp_hist(n-2)+temp_hist(2))/16+...
    4*(temp_hist(n-1)+temp_hist(1))/16+temp_hist(n)*6/16;

max_value=max(hist);

end


