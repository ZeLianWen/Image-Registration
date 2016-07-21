function display_product_image(gaussian_pyramid,dog_pyramid,gradient,angle,...
                               nOctaves,dog_center_layer,str)
                           
size_image=zeros(nOctaves,2);
for i=1:1:nOctaves
    size_image(i,1:2)=size(gaussian_pyramid{i,1});
end

%% display DOG Pyramid
ROW_size=sum(size_image);
ROW_size=ROW_size(1);
COL_size=size_image(1,1)*(dog_center_layer+2);
image_dog_pyramid=zeros(ROW_size,COL_size);
accumulate_ROW=0;
for i=1:1:nOctaves
    accumulate_ROW=accumulate_ROW+size_image(i,1);
    accumulate_COL=0;
    for j=1:1:dog_center_layer+2
        accumulate_COL=accumulate_COL+size_image(i,2);
        image_dog_pyramid(accumulate_ROW-size_image(i,1)+1:accumulate_ROW,...
           accumulate_COL-size_image(i,2)+1:accumulate_COL)=mat2gray(dog_pyramid{i,j});
    end
end

str1=['.\save_image\',str,'DOG Pyramid','.jpg'];
imwrite(image_dog_pyramid,str1,'jpg');
figure;%
imshow(image_dog_pyramid);
title([str,' DOG Pyramid-',num2str([ROW_size,COL_size])]);

%% display Gauss Pyramid
ROW_size=sum(size_image);
ROW_size=ROW_size(1);
COL_size=size_image(1,1)*(dog_center_layer+3);
image_gaussian_pyramid=zeros(ROW_size,COL_size);
accumulate_ROW=0;
for i=1:1:nOctaves
    accumulate_ROW=accumulate_ROW+size_image(i,1);
    accumulate_COL=0;
    for j=1:1:dog_center_layer+3
        accumulate_COL=accumulate_COL+size_image(i,2);
        image_gaussian_pyramid(accumulate_ROW-size_image(i,1)+1:accumulate_ROW,...
           accumulate_COL-size_image(i,2)+1:accumulate_COL)=mat2gray(gaussian_pyramid{i,j});
    end
end

str1=['.\save_image\',str,'OG Pyramid','.jpg'];
imwrite(image_gaussian_pyramid,str1,'jpg');
figure;
imshow(image_gaussian_pyramid);
title([str,' Gauss Pyramid--',num2str([ROW_size,COL_size])]);

%% display gradient image
ROW_size=sum(size_image);
ROW_size=ROW_size(1);
COL_size=size_image(1,1)*(dog_center_layer);
image_gaussian_gradient=zeros(ROW_size,COL_size);
accumulate_ROW=0;
for i=1:1:nOctaves
    accumulate_ROW=accumulate_ROW+size_image(i,1);
    accumulate_COL=0;
    for j=1:1:dog_center_layer
        accumulate_COL=accumulate_COL+size_image(i,2);
        image_gaussian_gradient(accumulate_ROW-size_image(i,1)+1:accumulate_ROW,...
           accumulate_COL-size_image(i,2)+1:accumulate_COL)=mat2gray(gradient{i,j});
    end
end
str1=['.\save_image\',str,'gradient','.jpg'];
imwrite(image_gaussian_gradient,str1,'jpg');
figure;
imshow(image_gaussian_gradient);
title([str,' gradient--',num2str([ROW_size,COL_size])]);

%% display orientation image
ROW_size=sum(size_image);
ROW_size=ROW_size(1);
COL_size=size_image(1,1)*(dog_center_layer);
image_gaussian_angle=zeros(ROW_size,COL_size);
accumulate_ROW=0;
for i=1:1:nOctaves
    accumulate_ROW=accumulate_ROW+size_image(i,1);
    accumulate_COL=0;
    for j=1:1:dog_center_layer
        accumulate_COL=accumulate_COL+size_image(i,2);
        image_gaussian_angle(accumulate_ROW-size_image(i,1)+1:accumulate_ROW,...
           accumulate_COL-size_image(i,2)+1:accumulate_COL)=mat2gray(angle{i,j});
    end
end
str1=['.\save_image\',str,'orientation','.jpg'];
imwrite(image_gaussian_angle,str1,'jpg');
figure;
imshow(image_gaussian_angle);
title([str,' orientation--',num2str([ROW_size,COL_size])]);










