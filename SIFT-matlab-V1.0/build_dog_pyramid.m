function [dog_pyramid]=build_dog_pyramid...
(gaussian_pyramid,...%高斯金字塔
 nOctaves,...%金字塔的组数
 dog_center_layer)%dog金字塔中心层数，默认是3

dog_pyramid=cell(nOctaves,dog_center_layer+2);%创建dog金字塔大小
for i=1:1:nOctaves
    for j=1:1:dog_center_layer+2
        dog_pyramid{i,j}(:,:)=gaussian_pyramid{i,j+1}(:,:)-gaussian_pyramid{i,j}(:,:);
    end
end
end


