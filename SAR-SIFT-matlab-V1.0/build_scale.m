function [sar_harris_function,gradient,angle]=build_scale(image,sigma,Mmax,ratio,d)


[M,N]=size(image);
sar_harris_function=zeros(M,N,Mmax);
gradient=zeros(M,N,Mmax);
angle=zeros(M,N,Mmax);

for i=1:1:Mmax
    %% 
    scale=sigma*ratio^(i-1);
    radius=round(2*scale);
    j=-radius:1:radius;
    k=-radius:1:radius;
    [xarry,yarry]=meshgrid(j,k);
    W=exp(-(abs(xarry)+abs(yarry))/scale);
    W34=zeros(2*radius+1,2*radius+1);
    W12=zeros(2*radius+1,2*radius+1);
    W14=zeros(2*radius+1,2*radius+1);
    W23=zeros(2*radius+1,2*radius+1);
    
    W34(radius+2:2*radius+1,:)=W(radius+2:2*radius+1,:);
    W12(1:radius,:)=W(1:radius,:);
    W14(:,radius+2:2*radius+1)=W(:,radius+2:2*radius+1);
    W23(:,1:radius)=W(:,1:radius);
    
    M34=imfilter(image,W34,'replicate');
    M12=imfilter(image,W12,'replicate');
    M14=imfilter(image,W14,'replicate');
    M23=imfilter(image,W23,'replicate');

    Gx=log(M14./M23);
    Gy=log(M34./M12);
    
     Gx(find(imag(Gx)))=abs(Gx(find(imag(Gx))));
     Gy(find(imag(Gy)))=abs(Gy(find(imag(Gy))));
     Gx(~isfinite(Gx))=0;
     Gy(~isfinite(Gy))=0;
     
    gradient(:,:,i)=sqrt(Gx.^2+Gy.^2);   
    temp_angle=atan2(Gy,Gx);
    temp_angle=temp_angle/pi*180;
    temp_angle(temp_angle<0)=temp_angle(temp_angle<0)+360;
    angle(:,:,i)=temp_angle;
    
    Csh_11=scale^2*Gx.^2;
    Csh_12=scale^2*Gx.*Gy;
    Csh_22=scale^2*Gy.^2;
    
    gaussian_sigma=sqrt(2)*scale;
    width=round(3*gaussian_sigma);
    width_windows=2*width+1;
    W_gaussian=fspecial('gaussian',[width_windows width_windows],gaussian_sigma);
    [a,b]=meshgrid(1:width_windows,1:width_windows);
    index=find(((a-width-1)^2+(b-width-1)^2)>width^2);
    W_gaussian(index)=0;
    
    Csh_11=imfilter(Csh_11,W_gaussian,'replicate');
    Csh_12=imfilter(Csh_12,W_gaussian,'replicate');
    Csh_21=Csh_12;
    Csh_22=imfilter(Csh_22,W_gaussian,'replicate');
    
    sar_harris_function(:,:,i)=Csh_11.*Csh_22-Csh_21.*Csh_12-d*(Csh_11+Csh_22).^2;
end
  
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
