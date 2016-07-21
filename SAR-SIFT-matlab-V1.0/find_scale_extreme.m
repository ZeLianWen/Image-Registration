function [key_point_array]=find_scale_extreme...
    (sar_harris_function,...
    threshold,...
    sigma,...
    ratio,...
    gradient,...
    angle)

[M,N,num]=size(sar_harris_function);
BORDER_WIDTH=2;
HIST_BIN=36;
SIFT_ORI_PEAK_RATIO=0.8;
key_number=0;
key_point_array=zeros(M,6);

for i=1:1:num
    temp_current=sar_harris_function(:,:,i);
    gradient_current=gradient(:,:,i);
    angle_current=angle(:,:,i);
    for j=BORDER_WIDTH:1:M-BORDER_WIDTH
        for k=BORDER_WIDTH:1:N-BORDER_WIDTH
            temp=temp_current(j,k);
            if(temp>threshold &&...
                temp>temp_current(j-1,k-1) && temp>temp_current(j-1,k) && temp>temp_current(j-1,k+1) &&...
                temp>temp_current(j,k-1) && temp>temp_current(j,k+1) &&...
                temp>temp_current(j+1,k-1) && temp>temp_current(j+1,k) && temp>temp_current(j+1,k+1)...
                )
                  
                scale=sigma*ratio^(i-1);
                [hist,max_value]=calculate_oritation_hist(k,j,scale,...
                        gradient_current,angle_current,HIST_BIN);
                
                mag_thr=max_value*SIFT_ORI_PEAK_RATIO;  
                for kk=1:1:HIST_BIN
                    if(kk==1)
                        k1=HIST_BIN;
                    else
                        k1=kk-1;
                    end 
                    if(kk==HIST_BIN)
                        k2=1;
                    else
                        k2=kk+1;
                    end
                     if(hist(kk)>hist(k1) && hist(kk)>hist(k2)...
                          && hist(kk)>mag_thr)
                  
                        bin=kk-1+0.5*(hist(k1)-hist(k2))/(hist(k1)+hist(k2)-2*hist(kk));
                        if(bin<0)
                            bin=HIST_BIN+bin;
                        elseif(bin>=HIST_BIN)
                            bin=bin-HIST_BIN;
                        end
                        key_number=key_number+1;
                        key_point_array(key_number,1)=k;
                        key_point_array(key_number,2)=j;
                        key_point_array(key_number,3)=sigma*ratio^(i-1);
                        key_point_array(key_number,4)=i;
                        key_point_array(key_number,5)=(360/HIST_BIN)*bin;
                        key_point_array(key_number,6)=hist(kk);
                    end
                end
                
            end
        end
    end
end
key_point_array=key_point_array(1:key_number,1:6);

end

