function [solution,rmse,cor1_new,cor2_new]=FSC(cor1,cor2,change_form,error_t)

[M,N]=size(cor1);
if(strcmp(change_form,'similarity'))
    n=2;
    max_iteration=M*(M-1)/2;
elseif(strcmp(change_form,'affine'))
    n=3;
    max_iteration=M*(M-1)*(M-2)/(2*3);
elseif(strcmp(change_form,'perspective'))
    n=4;
    max_iteration=M*(M-1)*(M-2)/(2*3);
end
if(max_iteration>800)
    iterations=800;
else
    iterations=max_iteration;%Algorithm iteration num
end

most_consensus_number=0;
cor1_new=zeros(M,N);
cor2_new=zeros(M,N);

%%
%rand('seed',0);
for i=1:1:iterations
    while(1)
        a=floor(1+(M-1)*rand(1,n));
        cor11=cor1(a,1:2);
        cor22=cor2(a,1:2);
        
        if(n==2 && (a(1)~=a(2)) && sum(cor11(1,1:2)~=cor11(2,1:2),2) &&...
                sum(cor22(1,1:2)~=cor22(2,1:2)))
            break;
        end
        if(n==3 && (a(1)~=a(2) && a(1)~=a(3) && a(2)~=a(3)) && ...
        sum(cor11(1,1:2)~=cor11(2,1:2)) && sum(cor11(1,1:2)~=cor11(3,1:2)) && sum(cor11(2,1:2)~=cor11(3,1:2))...
        && sum(cor22(1,1:2)~=cor22(2,1:2)) && sum(cor22(1,1:2)~=cor22(3,1:2)) && sum(cor22(2,1:2)~=cor22(3,1:2)))
            break;
        end 
        if(n==4 && (a(1)~=a(2) && a(1)~=a(3) && a(1)~=a(4) && a(2)~=a(3) && a(2)~=a(4) && a(3)~=a(4)) && ...
        sum(cor11(1,1:2)~=cor11(2,1:2)) && sum(cor11(1,1:2)~=cor11(3,1:2)) && sum(cor11(1,1:2)~=cor11(4,1:2)) &&... 
        sum(cor11(2,1:2)~=cor11(3,1:2)) && sum(cor11(2,1:2)~=cor11(4,1:2)) && sum(cor11(3,1:2)~=cor11(4,1:2))...
        && sum(cor22(1,1:2)~=cor22(2,1:2)) && sum(cor22(1,1:2)~=cor22(3,1:2)) && sum(cor22(1,1:2)~=cor22(4,1:2))&&...
        sum(cor22(2,1:2)~=cor22(3,1:2)) && sum(cor22(2,1:2)~=cor22(4,1:2)) && sum(cor22(3,1:2)~=cor22(4,1:2)))
            break;
        end 
    end
      
    [parameters,~]=LSM(cor11,cor22,change_form);
    solution=[parameters(1),parameters(2),parameters(5);
        parameters(3),parameters(4),parameters(6);
        parameters(7),parameters(8),1];
    
     match1_xy=cor1(:,1:2)';
     match1_xy=[match1_xy;ones(1,M)];
    %for perspective
    if(strcmp(change_form,'perspective'))
        match1_test_trans=solution*match1_xy;
        match1_test_trans_12=match1_test_trans(1:2,:);
        match1_test_trans_3=match1_test_trans(3,:);
        match1_test_trans_3=repmat(match1_test_trans_3,2,1);
        match1_test_trans=match1_test_trans_12./match1_test_trans_3;
        match1_test_trans=match1_test_trans';
        match2_xy=cor2(:,1:2);
        match2_test=match2_xy;
        test=match1_test_trans-match2_test;
        diff_match2_xy=sqrt(sum(test.^2,2));
        index_in=find(diff_match2_xy<error_t);
        consensus_num=size(index_in,1);
    else%for similarity and affine
        t_match1_xy=solution*match1_xy;
        match2_xy=cor2(:,1:2)';
        match2_xy=[match2_xy;ones(1,M)];
        diff_match2_xy=t_match1_xy-match2_xy;
        diff_match2_xy=sqrt(sum(diff_match2_xy.^2));
        index_in=find(diff_match2_xy<error_t);
        consensus_num=size(index_in,2);
    end
     
    if(consensus_num>most_consensus_number)
        most_consensus_number=consensus_num;
        cor1_new=cor1(index_in,:);
        cor2_new=cor2(index_in,:);
    end
end

%Delete duplicate point pair
uni1=cor1_new(:,[1 2]);
[~,i,~]=unique(uni1,'rows','first');
cor1_new=cor1_new(sort(i)',:);cor2_new=cor2_new(sort(i)',:);
uni1=cor2_new(:,[1 2]);
[~,i,~]=unique(uni1,'rows','first');
cor1_new=cor1_new(sort(i)',:);cor2_new=cor2_new(sort(i)',:);

[parameters,rmse]=LSM(cor1_new(:,1:2),cor2_new(:,1:2),change_form);
solution=[parameters(1),parameters(2),parameters(5);
    parameters(3),parameters(4),parameters(6);
    parameters(7),parameters(8),1];

end
























