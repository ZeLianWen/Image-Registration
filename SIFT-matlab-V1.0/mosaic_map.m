function [image1,image2,img3]=mosaic_map(img1,img2,d)

[m1,n1,p1] = size(img1);
m11 = ceil(m1/d);
n11 = ceil(n1/d);
for i=1:2:m11
    for j=2:2:n11
        img1((i-1)*d+1:i*d,(j-1)*d+1:j*d,:)=0;
    end
end
for i=2:2:m11
    for j=1:2:n11
        img1((i-1)*d+1:i*d,(j-1)*d+1:j*d,:)=0;
    end
end
image1=img1(1:m1,1:n1,:);

%%
[m2,n2,p2] = size(img2);
m22 = ceil(m2/d);
n22 = ceil(n2/d);
for i=1:2:m22
    for j=1:2:n22
       img2((i-1)*d+1:i*d,(j-1)*d+1:j*d,:)=0;
    end
end
for i=2:2:m22
    for j=2:2:n22
       img2((i-1)*d+1:i*d,(j-1)*d+1:j*d,:)=0;
    end
end
image2=img2(1:m2,1:n2,:);
%%
img3=image1+image2;

end
