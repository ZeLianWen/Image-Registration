function [solution,rmse,cor1_new,cor2_new]=ransac(cor1,cor2,change_form,error_t)
%该函数是ransac算法的具体实现，该算法主要完成数据的拟合，这里的目的是拟合一个几何变换
%cor1是待配准图像中的坐标点，是一个M*2的矩阵，每一行分别存放一个点的x和y坐标
%cor2是参考图像中的坐标点
%change_form是拟合的几何变换类型，这里目前支持‘相似变换’和‘仿射变换’
%solution是计算的几何变换模型参数，rmse是计算的变换误差
%cor1_new是cor1中满足条件的点的集合
%cor2_new是cor2中满足条件的点的集合
%error_t;%误差阈值

%% 参数初始化
[M,N]=size(cor1);
if(strcmp(change_form,'相似变换'))
    n=2;%对于相似变换需要计算模型参数的点个数是2
    max_iteration=M*(M-1)/2;
elseif(strcmp(change_form,'仿射变换'))
    n=3;%对于仿射变换需要计算模型参数的个数是3
    max_iteration=M*(M-1)*(M-2)/(2*3);
end
if(max_iteration>500)
    iterations=500;
else
    iterations=max_iteration;%算法迭代次数
end
consensus_number=0.05*M;%一致集最小个数阈值是10
consensus_number=max(consensus_number,n);
best_solution=zeros(3,3);%初始化解为3*3的矩阵
most_consensus_number=0;%初始化开始一致集个数很小
rmse=10000;
cor1_new=zeros(M,N);
cor2_new=zeros(M,N);

%%
rand('seed',0);
for i=1:1:iterations
    while(1)%随机生成三个不相等的数据
        a=floor(1+(M-1)*rand(1,n));
        cor11=cor1(a,1:2);%随机选择的n个点对
        cor22=cor2(a,1:2);
        if(n==2 && (a(1)~=a(2)) && sum(cor11(1,1:2)~=cor11(2,1:2),2) &&...
                sum(cor22(1,1:2)~=cor22(2,1:2)))
            break;
        end
        if(n==3 && (a(1)~=a(2) && a(1)~=a(3) && a(2)~=a(3)) && ...
        sum(cor11(1,1:2)~=cor11(2,1:2)) && sum(cor11(1,1:2)~=cor11(3,1:2)) && sum(cor22(2,1:2)~=cor11(3,1:2))...
        && sum(cor22(1,1:2)~=cor22(2,1:2)) && sum(cor22(1,1:2)~=cor22(3,1:2)) && sum(cor22(2,1:2)~=cor22(3,1:2)))
            break;
        end       
    end
      
    [parameters,~]=LSM(cor11,cor22,change_form);
    solution=[parameters(1),parameters(2),parameters(5);
        parameters(3),parameters(4),parameters(6);
        parameters(7),parameters(8),1];
    match1_xy=cor1(:,1:2)';
    match1_xy=[match1_xy;ones(1,M)];
    t_match1_xy=solution*match1_xy;
    match2_xy=cor2(:,1:2)';
    match2_xy=[match2_xy;ones(1,M)];
    diff_match2_xy=t_match1_xy-match2_xy;
    diff_match2_xy=sqrt(sum(diff_match2_xy.^2));
    index_in=find(diff_match2_xy<error_t);%满足一致条件的点的索引
    consensus_num=size(index_in,2);%满足条件的一致集合点个数
    
    %if(consensus_num>consensus_number)%如果满足一致集个数要求   
        if(consensus_num>most_consensus_number)
            most_consensus_number=consensus_num;
            cor1_new=cor1(index_in,:);
            cor2_new=cor2(index_in,:);
        end
    %end
end

%删除重复点对后，再次计算变换矩阵关系
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
























