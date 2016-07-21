function [hist,max_value]=calculate_oritation_hist(x,y,scale,gradient,angle,n)

[M,N]=size(gradient);
radius=round(min(6*scale,min(M/2,N/2)));
sigma=2*scale;
[M,N,~]=size(gradient);
radius_x_left=x-radius;
radius_x_right=x+radius;
radius_y_up=y-radius;
radius_y_down=y+radius;

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

center_x=x-radius_x_left+1;
center_y=y-radius_y_up+1;

sub_gradient=gradient(radius_y_up:radius_y_down,radius_x_left:radius_x_right);
sub_angle=angle(radius_y_up:radius_y_down,radius_x_left:radius_x_right);

X=-(x-radius_x_left):1:(radius_x_right-x);
Y=-(y-radius_y_up):1:(radius_y_down-y);
[XX,YY]=meshgrid(X,Y);
%gaussian_weight=1/(sqrt(2*pi)*sigma)*exp(-(XX.^2+YY.^2)/(2*sigma^2));
%W=sub_gradient.*gaussian_weight;
W=sub_gradient;
bin=round(sub_angle*n/360);

bin(bin>=n)=bin(bin>=n)-n;
bin(bin<0)=bin(bin<0)+n;

temp_hist=zeros(1,n);
[row,col]=size(bin);
for i=1:1:row
    for j=1:1:col
       if(((i-center_y)^2+(j-center_x)^2)<=radius^2)
            temp_hist(bin(i,j)+1)=temp_hist(bin(i,j)+1)+W(i,j);
       end
    end
end

%% Æ½»¬Ö±·½Í¼
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



