function [dog_pyramid]=build_dog_pyramid(gaussian_pyramid,nOctaves,dog_center_layer)

dog_pyramid=cell(nOctaves,dog_center_layer+2);
for i=1:1:nOctaves
    for j=1:1:dog_center_layer+2
        dog_pyramid{i,j}(:,:)=gaussian_pyramid{i,j+1}(:,:)-gaussian_pyramid{i,j}(:,:);
    end
end
end


