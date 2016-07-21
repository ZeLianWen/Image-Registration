function [key_point_array]=find_scale_space_extream....
    (...
    dog_pyramid,...%DOG Pyramid
    nOctaves,...%The number of groups in Pyramid
    nOctaveLayers,....%%Default is 3
    contrast_threshold,....%%Contrast threshold£¬Default is 0.04
    sigma,...%Default is 1.6
    edge_threshold,...%Threshold response threshold,Default is 10
    gaussian_gradient,...%Gradient magnitude
    gaussian_angle...%Gradient angle
    )

LOG_POLAR_HIST_BINS=36;
n=LOG_POLAR_HIST_BINS;
key_point_array=struct('x',{},'y',{},'octaves',{},'layer',{},...
     'xi',{},'size',{},'angle',{},'gradient',{});
num=0;

threshold=contrast_threshold/nOctaveLayers;

SIFT_IMG_BORDER=2;%Boundary constant
SIFT_ORI_PEAK_RATIO=0.8;%Histogram peak ratio

for i=1:1:nOctaves
    for j=2:1:nOctaveLayers+1
        current_dog=dog_pyramid{i,j};%Dog Pyramid current level index
        prev_dog=dog_pyramid{i,j-1};%Dog Pyramid prev level index
        next_dog=dog_pyramid{i,j+1};%Dog Pyramid next level index
        [M,N]=size(current_dog);
        
        for r=SIFT_IMG_BORDER:1:M-SIFT_IMG_BORDER%%row
            for c=SIFT_IMG_BORDER:1:N-SIFT_IMG_BORDER%%col
                val=current_dog(r,c);
                if(abs(val)>threshold &&...
                    ((val>0 && val>current_dog(r,c-1) && val>current_dog(r,c+1)...
                     && val>current_dog(r-1,c-1) && val>current_dog(r-1,c)...
                     && val>current_dog(r-1,c+1) && val>current_dog(r+1,c-1)...
                     && val>current_dog(r+1,c) && val>current_dog(r+1,c+1)...
                     && val>prev_dog(r-1,c-1) && val>prev_dog(r-1,c)...
                     && val>prev_dog(r-1,c+1) && val>prev_dog(r,c-1)...
                     && val>prev_dog(r,c) && val>prev_dog(r,c+1)...
                     && val>prev_dog(r+1,c-1) && val>prev_dog(r+1,c)...
                     && val>prev_dog(r+1,c+1)...
                     && val>next_dog(r-1,c-1) && val>next_dog(r-1,c)...
                     && val>next_dog(r-1,c+1) && val>next_dog(r,c-1)...
                     && val>next_dog(r,c) && val>next_dog(r,c+1)...
                     && val>next_dog(r+1,c-1) && val>next_dog(r+1,c)...
                     && val>next_dog(r+1,c+1))...
                     || (val<0 && val<current_dog(r,c-1) && val<current_dog(r,c+1)...
                     && val<current_dog(r-1,c-1) && val<current_dog(r-1,c)...
                     && val<current_dog(r-1,c+1) && val<current_dog(r+1,c-1)...
                     && val<current_dog(r+1,c) && val<current_dog(r+1,c+1)...
                     && val<prev_dog(r-1,c-1) && val<prev_dog(r-1,c)...
                     && val<prev_dog(r-1,c+1) && val<prev_dog(r,c-1)...
                     && val<prev_dog(r,c) && val<prev_dog(r,c+1)...
                     && val<prev_dog(r+1,c-1) && val<prev_dog(r+1,c)...
                     && val<prev_dog(r+1,c+1)...
                     && val<next_dog(r-1,c-1) && val<next_dog(r-1,c)...
                     && val<next_dog(r-1,c+1) && val<next_dog(r,c-1)...
                     && val<next_dog(r,c) && val<next_dog(r,c+1)...
                     && val<next_dog(r+1,c-1) && val<next_dog(r+1,c)...
                     && val<next_dog(r+1,c+1))))
                 
                    r1=r;c1=c;layer=j;%2<=layer<=nOctaveLayers+1
                    %Precise location of feature points
                    [key_point,is_local_extream]=adjust_local_extream...
                        (dog_pyramid,i,layer,r1,c1,nOctaveLayers,sigma,...
                        contrast_threshold,edge_threshold);
                    if(is_local_extream==false)
                        continue;
                    end
                    
                    scl_octv=key_point.size/(2^(key_point.octaves-1));
                    y=key_point.y/(2^(key_point.octaves-1));
                    x=key_point.x/(2^(key_point.octaves-1));
                    y=round(y);x=round(x);
                    
                    %The direction histogram of characteristic points
                    [hist,max_value]=calculate_oritation_hist...
                        (x,...
                        y,...
                        scl_octv,...
                        gaussian_gradient{key_point.octaves,key_point.layer-1},...
                        gaussian_angle{key_point.octaves,key_point.layer-1},...
                        n);
                    
                     mag_thr=max_value*SIFT_ORI_PEAK_RATIO;  
                     for k=1:1:n
                         if(k==1)
                             k1=n;
                         else
                             k1=k-1;
                         end
                         
                         if(k==n)
                             k2=1;
                         else
                             k2=k+1;
                         end
                         if(hist(k)>hist(k1) && hist(k)>hist(k2)...
                                 && hist(k)>mag_thr)
                            bin=k-1+0.5*(hist(k1)-hist(k2))/(hist(k1)+hist(k2)-2*hist(k));
                            if(bin<0)
                                bin=n+bin;
                            elseif(bin>=n)
                                bin=bin-n;
                            end
                            key_point.angle=(360/n)*bin;
                            key_point.gradient=hist(k);
                            num=num+1;
                            key_point_array(num)=key_point;
                         end
                     end
                end
            end
        end
    end
end
end

























