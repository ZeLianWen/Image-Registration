function [cor_11,cor_22]=logic_filter(cor_1,cor_2,delat_x,delat_y,unit_x,unit_y,S,RMO)

delat_x=-delat_x;
delat_y=-delat_y;

cor1_xy=cor_1(:,[1,2]);
cor2_xy=cor_2(:,[1,2]);
temp_S=1/S;
temp_RMO=-RMO(1);
temp_T=[temp_S*cos(temp_RMO/180*pi),-temp_S*sin(temp_RMO/180*pi);...
    temp_S*sin(temp_RMO/180*pi),temp_S*cos(temp_RMO/180*pi)];
diff_xy=cor2_xy'-temp_T*cor1_xy';

diff_x=diff_xy(1,:);
diff_y=diff_xy(2,:);
abs_diff_delat_x=abs(diff_x-delat_x);
abs_diff_delat_y=abs(diff_y-delat_y);

index=find((abs_diff_delat_x<unit_x) & (abs_diff_delat_y<unit_y));
cor_11=cor_1(index,:);
cor_22=cor_2(index,:);

end
