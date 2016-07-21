function [gaussian_pyramid,gaussian_gradient,gaussian_angle]=build_gaussian_pyramid(...
    base_image,nOctaves,nOctaveLayers,sigma)

sig=zeros(1,nOctaveLayers+3);
sig(1)=sigma;
k=2^(1.0/nOctaveLayers);

for i=2:1:(nOctaveLayers+3)
    sig_previous=k^(i-2)*sigma;%Image scale of the previous layer
    sig_current=k*sig_previous;%Scale of the current layer
    sig(i)=sqrt(sig_current^2-sig_previous^2);
end

gaussian_pyramid=cell(nOctaves,nOctaveLayers+3);
gaussian_gradient=cell(nOctaves,nOctaveLayers+3);%Save gradient magnitude
gaussian_angle=cell(nOctaves,nOctaveLayers+3);%Save angle
h=[-1,0,1;-2,0,2;-1,0,1];%Differential template£¨Sobel£©

for o=1:1:nOctaves%each group
    for i=1:1:(nOctaveLayers+3)%each layer
        if(o==1 && i==1)
            gaussian_pyramid{1,1}(:,:)=base_image;
        elseif(i==1)
            temp=gaussian_pyramid{(o-1),nOctaveLayers+1}(:,:);
            gaussian_pyramid{o,1}(:,:)=imresize(temp,1/2,'bilinear');
        else
            WINDOW_GAUSSIAN=round(2*sig(i));
            WINDOW_GAUSSIAN=2*WINDOW_GAUSSIAN+1;
            w=fspecial('gaussian',[WINDOW_GAUSSIAN,WINDOW_GAUSSIAN],sig(i));%Gauss filter core
            temp=gaussian_pyramid{o,i-1}(:,:);
            gaussian_pyramid{o,i}=imfilter(temp,w,'replicate');%%Gauss filter
            
            if(i>=2 && i<=nOctaveLayers+1)
                gradient_x=imfilter(gaussian_pyramid{o,i}(:,:),h,'replicate');
                gradient_y=imfilter(gaussian_pyramid{o,i}(:,:),h','replicate');
                gaussian_gradient{o,i-1}(:,:)=sqrt(gradient_x.^2+gradient_y.^2);
                
                temp_angle=atan2(gradient_y,gradient_x);
                temp_angle=temp_angle*180/pi;
                temp_angle(temp_angle<0)=temp_angle(temp_angle<0)+360;
                gaussian_angle{o,i-1}(:,:)=temp_angle;
                
                gradient_x=imfilter(gaussian_gradient{o,i-1},h,'replicate');
                gradient_y=imfilter(gaussian_gradient{o,i-1},h','replicate'); 
                temp_angle=atan2(gradient_y,gradient_x);
                temp_angle=temp_angle*180/pi;
                temp_angle(temp_angle<0)=temp_angle(temp_angle<0)+360;
                gaussian_angle{o,i-1}(:,:)=temp_angle;
                gaussian_gradient{o,i-1}(:,:)=sqrt(gradient_x.^2+gradient_y.^2);
                
            end
        end
    end
end
end

            
                
        


