function [parameters,rmse]=LSM(match1,match2,change_form)

match1_xy=match1(:,[1 2]);
match2_xy=match2(:,[1 2]);

A=[];                                              % Ax=b;
%A=[x1,y1,x1,y1£»x1,y1,x1,y1;......xn,yn,xn,yn£»xn,yn,xn,yn]
for i=1:size(match1_xy,1)
    A=[A;repmat(match1_xy(i,:),2,2)];
end
%B=[1 1 0 0;0 0 1 1....1 1 0 0;0 0 1 1]
B=[1 1 0 0;0 0 1 1];B=repmat(B,size(match1_xy,1),1);
%A=[x1,y1,0,0£»0,0,x1,y1;.....xn,yn,0,0£»0,0,xn,yn]
A=A.*B;
%B=[1,0;0,1;....1,0;0,1]
B=[1 0;0 1];B=repmat(B,size(match1_xy,1),1);
%A=[x1,y1,0,0,1,0;0,0,x1,y1,0,1;.....xn,yn,0,0,1,0;0,0,xn,yn,0,1]
A=[A B];
t_match2_xy=match2_xy';b=t_match2_xy(:);
           
%affine
if(strcmp(change_form,'affine'))
    [Q,R]=qr(A);
    parameters=R\(Q'*b);
    parameters(7:8)=0;
    %parameters=A\b;
    N=size(match1,1);
    match1_test=match1(:,[1 2]);match2_test=match2(:,[1 2]);
    M=[parameters(1) parameters(2);parameters(3) parameters(4)];
    match1_test_trans=M*match1_test'+repmat([parameters(5);parameters(6)],1,N);
    match1_test_trans=match1_test_trans';
    test=match1_test_trans-match2_test;
    rmse=sqrt(sum(sum(test.^2))/N);
elseif(strcmp(change_form,'similarity'))
    %[x,y,1,0;
    % y,-x,0,1]*[a,b,c,d]'=[u,v]
    A=[];
    for i=1:1:size(match1_xy,1)
        A=[A;match1_xy(i,:),1,0;match1_xy(i,2),-match1_xy(i,1),0,1];
    end
    [Q,R]=qr(A);
    parameters=R\(Q'*b);
    %parameters=A\b;
    parameters(7:8)=0;
    parameters(5:6)=parameters(3:4);
    parameters(3)=-parameters(2);
    parameters(4)=parameters(1);
    
    N=size(match1,1);
    match1_test=match1(:,[1 2]);match2_test=match2(:,[1 2]);
    M=[parameters(1) parameters(2);parameters(3) parameters(4)];
    match1_test_trans=M*match1_test'+repmat([parameters(5);parameters(6)],1,N);
    match1_test_trans=match1_test_trans';
    test=match1_test_trans-match2_test;
    rmse=sqrt(sum(sum(test.^2))/N);
end                                      
end
