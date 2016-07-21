function [descriptor]=calc_sift_descriptor(gradient,angle,x,y,scale,main_angle,d,n)

histlen=(d+2)*(d+2)*(n+2);
hist=zeros(1,histlen);

cos_t=cos(-main_angle/180*pi);
sin_t=sin(-main_angle/180*pi);

exp_scale=-1/(0.5*d*d);
hist_width=3*scale;
radius=round(hist_width*(d+1)*1.414/2);
[M,N]=size(gradient);
radius=min(radius,round(sqrt(M*M+N*N)));

cos_t=cos_t/hist_width;
sin_t=sin_t/hist_width;

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

sub_gradient=gradient(radius_y_up:radius_y_down,radius_x_left:radius_x_right);
sub_angle=angle(radius_y_up:radius_y_down,radius_x_left:radius_x_right);

X=-(x-radius_x_left):1:(radius_x_right-x);
Y=-(y-radius_y_up):1:(radius_y_down-y);
[XX,YY]=meshgrid(X,Y);
c_rot=XX*cos_t-YY*sin_t;
r_rot=XX*sin_t+YY*cos_t;
Rbin=r_rot+d/2-0.5;
Cbin=c_rot+d/2-0.5;
%gaussian_weight=exp((c_rot.^2+r_rot.^2)*exp_scale);
[row,col]=size(sub_angle);

for i=1:1:row
    for j=1:1:col
        rbin=Rbin(i,j);cbin=Cbin(i,j);
        if(rbin<=-1 || rbin>=d || cbin<=-1 || cbin>=d)
            continue;
        end
        obin=(sub_angle(i,j)-main_angle)*(n/360);
        %mag=sub_gradient(i,j)*gaussian_weight(i,j);
        mag=sub_gradient(i,j);%这里不用高斯权重
        
        r0=floor(rbin);
        c0=floor(cbin);
        o0=floor(obin);
        
        rbin=rbin-r0;
        cbin=cbin-c0;
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
        idx = ((r0+1)*(d+2) + c0+1)*(n+2) + o0+1;
        
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

%% Feature vector normalization
descriptor=descriptor/sqrt(descriptor*descriptor');
descriptor(descriptor>=0.2)=0.2;
descriptor=descriptor/sqrt(descriptor*descriptor');

end


         
      
      














                                             
                                             
                                             
                                             
                                           

