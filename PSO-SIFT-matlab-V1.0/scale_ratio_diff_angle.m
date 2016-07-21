function scale_ratio_diff_angle(cor11,cor22)

[S,RMO,delat_x,delat_y,scale_hist,scale_x,angle_hist,...
    angle_x,x_hist,x_x,y_hist,y_x]=S_RMO_RtoS(cor11,cor22);

%Scale ratio histogram
f1=figure;
scale_hist=scale_hist*100/sum(scale_hist);
bar(scale_x,scale_hist,'b');
grid on
xlabel('S_r_e_f/S_s_e_n_s_e_d');ylabel('%');
str1=['.\save_image\','Scale ratio histogram','.jpg'];
saveas(f1,str1,'jpg');

%Histogram of angle difference
f2=figure;
angle_hist=angle_hist*100/sum(angle_hist);
bar(angle_x,angle_hist,'b');
grid on
xlabel('\Delta\theta(Degrees)');ylabel('%');
str1=['.\save_image\','Histogram of angle difference','.jpg'];
saveas(f2,str1,'jpg');

%Mapping X direction displacement difference histogram
f3=figure;
x_hist=x_hist*100/sum(x_hist);
bar(x_x,x_hist,'b');
grid on
xlabel('\DeltaX(Pixels)');ylabel('%');
str1=['.\save_image\','X direction displacement difference histogram','.jpg'];
saveas(f3,str1,'jpg');

%Mapping Y direction displacement difference histogram
f4=figure;
y_hist=y_hist*100/sum(y_hist);
bar(y_x,y_hist,'b');
grid on
%,'FontName','Time New Roman','FontSize',8
xlabel('\DeltaY(Pixels)');
ylabel('%');
str1=['.\save_image\','Y direction displacement difference histogram','.jpg'];
saveas(f4,str1,'jpg');

end

