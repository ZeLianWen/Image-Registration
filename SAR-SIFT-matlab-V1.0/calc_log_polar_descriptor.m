function [descriptor]=calc_log_polar_descriptor(gradient,angle,x,y,scale,main_angle,d,n)

cos_t=cos(-main_angle/180*pi);
sin_t=sin(-main_angle/180*pi);

[M,N]=size(gradient);
radius=round(min(12*scale,min(M,N)/2));

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
sub_angle=round((sub_angle-main_angle)*n/360);
sub_angle(sub_angle<=0)=sub_angle(sub_angle<=0)+n;
sub_angle(sub_angle==0)=n;

X=-(x-radius_x_left):1:(radius_x_right-x);
Y=-(y-radius_y_up):1:(radius_y_down-y);
[XX,YY]=meshgrid(X,Y);
c_rot=XX*cos_t-YY*sin_t;
r_rot=XX*sin_t+YY*cos_t;

log_angle=atan2(r_rot,c_rot);
log_angle=log_angle/pi*180;
log_angle(log_angle<0)=log_angle(log_angle<0)+360;
log_amplitude=log2(sqrt(c_rot.^2+r_rot.^2));

log_angle=round(log_angle*d/360);
log_angle(log_angle<=0)=log_angle(log_angle<=0)+d;
log_angle(log_angle>d)=log_angle(log_angle>d)-d;

r1=log2(radius*0.25);
r2=log2(radius*0.73);
log_amplitude(log_amplitude<=r1)=1;
log_amplitude(log_amplitude>r1 & log_amplitude<=r2)=2;
log_amplitude(log_amplitude>r2)=3;

temp_hist=zeros(1,(2*d+1)*n);
[row,col]=size(log_angle);
for i=1:1:row
    for j=1:1:col
        if(((i-center_y)^2+(j-center_x)^2)<=radius^2)
            angle_bin=log_angle(i,j);
            amplitude_bin=log_amplitude(i,j);
            bin_vertical=sub_angle(i,j);
            Mag=sub_gradient(i,j);
             
            if(amplitude_bin==1)
                temp_hist(bin_vertical)=temp_hist(bin_vertical)+Mag;
            else
                temp_hist(((amplitude_bin-2)*d+angle_bin-1)*n+bin_vertical+n)=...
                    temp_hist(((amplitude_bin-2)*d+angle_bin-1)*n+bin_vertical+n)+Mag;
            end
        end
    end
end

temp_hist=temp_hist/sqrt(temp_hist*temp_hist');
temp_hist(temp_hist>0.2)=0.2;
temp_hist=temp_hist/sqrt(temp_hist*temp_hist');
descriptor=temp_hist;

end


  



         
      
      














                                             
                                             
                                             
                                             
                                           

