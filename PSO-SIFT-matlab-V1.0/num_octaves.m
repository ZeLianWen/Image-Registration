function [nOctaves]=num_octaves(image,double_size_image)

temp=log2(min(size(image,1),size(image,2)))-2;
if(double_size_image==true)
    nOctaves=temp+1;
else
    nOctaves=temp;
end
    nOctaves=floor(nOctaves);
end
