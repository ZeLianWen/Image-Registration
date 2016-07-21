function [end_solution,point_1,point_2,rmse]=...
scale_orien_joint_restriction(solution,loc1,loc2,cor1,cor2,des1,des2,Td,change_form,distance)
                                          
loc1=loc1(:,[2,1,6,7,3,4]);
loc2=loc2(:,[2,1,6,7,3,4]);

cor1=cor1(:,[5,6,1,2]);
cor2=cor2(:,[5,6,1,2]);

unit_scale=0.075;
unit_angle=9;
unit_x=7.5;
unit_y=7.5;
[S,RMO,delat_x,delat_y,~,~,~,~,~,~,~,~]=S_RMO_StoR(cor1,cor2,unit_scale,unit_angle,unit_x,unit_y);

M_des1=size(des1,1);
M_des2=size(des2,1);
    
K1=size(S,2);
K2=size(RMO,2);
match=zeros(M_des1,2);
for i=1:1:K1
    for j=1:1:K2
        temp_S=S(i);
        temp_RMO=RMO(j);
        prev_match=match;
        for k=1:1:M_des1
            %%Scale ratio error
            temp_scale=loc1(k,5);
            temp_scale=repmat(temp_scale,M_des2,1);
            ratio_scale=loc2(:,5)./temp_scale;
            error_S=abs(1-temp_S*ratio_scale);
            error_S=error_S';
            
            %Relative principal direction error
            temp_angle=loc1(k,6);
            temp_angle=repmat(temp_angle,M_des2,1);
            diff_angle=temp_angle-loc2(:,6);
            error_RMO=abs(diff_angle-temp_RMO);
            error_RMO=error_RMO';
            
            %position error
            temp1_xy=loc1(k,[1,2]);
            temp1_xy=repmat(temp1_xy,M_des2,1);
            temp1_xy=temp1_xy';
            temp1_xy=[temp1_xy;ones(1,M_des2)];
            T_temp1_xy=solution*temp1_xy;
            
            temp2_xy=loc2(:,[1,2]);
            temp2_xy=temp2_xy';
            temp2_xy=[temp2_xy;ones(1,M_des2)];
            diff_xy=T_temp1_xy-temp2_xy;
            T_error=sqrt(sum(diff_xy.^2,1));

            JD=(1+T_error).*(1+error_S).*(1+error_RMO).*distance(k,:);
            
            [vals,index] = sort(JD); 
            if(vals(1)/vals(2)<Td)
                match(k,1)=index(1);
                Dk=vals(1);
                match(k,2)=Dk;
            else
                match(k,1)=0;
                match(k,2)=0;
            end    
             
            if(i==1 && j==1)
                match(k,:)=match(k,:);
            elseif(prev_match(k,1)==0)
                match(k,:)=match(k,:);
            elseif(prev_match(k,1)~=0)
                if(match(k,1)~=0 && match(k,2)<prev_match(k,2))
                    match(k,:)=match(k,:);
                else
                    match(k,:)=prev_match(k,:); 
                end
            end
        end            
    end
end

temp_match=match(:,1)';
num = sum(temp_match > 0);
%fprintf('尺度方向联合约束距离比阶段Found %d matches.\n', num);

[~,point1,point2]=find(temp_match);
loc11=loc1(point1,[1,2,3,4,5,6]);
loc22=loc2(point2,[1,2,3,4,5,6]);
loc11=[loc11,point2'];
loc22=[loc22,point2'];

%% logic filter
[loc11,loc22,]=logic_filter(loc11,loc22,delat_x,delat_y,unit_x,unit_y,S,RMO);
%fprintf('logic filter后Found %d matches.\n', size(loc11,1));

uni1=[loc11(:,[1 2]),loc22(:,[1 2])];
[~,i,~]=unique(uni1,'rows','first');
loc11=loc11(sort(i)',:);loc22=loc22(sort(i)',:);

[end_solution,rmse,loc11,loc22]=FSC(loc11,loc22,change_form,1);%%Sub-pixel accuracy
%fprintf('尺度，位置，方向联合约束Found %d matches.\n', size(loc11,1));

point_1=loc11;
point_2=loc22;

end













