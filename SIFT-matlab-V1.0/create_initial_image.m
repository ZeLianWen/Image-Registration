function [image]=create_initial_image(I,double_image_size,sigma)

if(size(I,3)==3)
    image_gray=rgb2gray(I);
else
    image_gray=I;
end

image_gray=im2double(image_gray);
SIFT_INIT_SIGMA=0.5;
WINDOW_GAUSSIAN=5;

if(double_image_size==true)
   sig_diff=sqrt(max(sigma*sigma-4*SIFT_INIT_SIGMA^2,0.01)); 
   image_gray=imresize(image_gray,2,'bilinear');
   w=fspecial('gaussian',[WINDOW_GAUSSIAN,WINDOW_GAUSSIAN],sig_diff);
   image=imfilter(image_gray,w,'replicate');
else
   sig_diff=sqrt(max(sigma*sigma-SIFT_INIT_SIGMA^2,0.01));
   w=fspecial('gaussian',[WINDOW_GAUSSIAN,WINDOW_GAUSSIAN],sig_diff);
   image=imfilter(image_gray,w,'replicate');
end

end

