function [descriptors,locs]=calc_descriptors...
(   gradient,...
    angle,...
    key_point_array...
)

circle_bin=8;
LOG_DESC_HIST_BINS=8;

M=size(key_point_array,1);
d=circle_bin;
n=LOG_DESC_HIST_BINS;
descriptors=zeros(M,(2*d+1)*n);
locs=key_point_array;

for i=1:1:M
    x=key_point_array(i,1);
    y=key_point_array(i,2);
    scale=key_point_array(i,3);
    layer=key_point_array(i,4);
    main_angle=key_point_array(i,5);
    current_gradient=gradient(:,:,layer);
    current_angle=angle(:,:,layer);
    descriptors(i,:)=calc_log_polar_descriptor(current_gradient,current_angle,...
                                           x,y,scale,main_angle,d,n);
                                                                                                                                                                                                                
    
end


