function [key_point,is_local_extream]=adjust_local_extream...
    (dog_pyramid,...
    octaves,...
    layer,...
    r,....
    c,....
    nOctaveLayers,...%%Default is3
    sigma,...%%Default is1.6
    contrast_threshold,....%Contrast threshold£¬Default is 0.04
    edge_threshold)%%Threshold response threshold£¬Default is 10

key_point=struct('x',{},'y',{},'octaves',{},...
'layer',{},'xi',{},'size',{},'angle',{},'gradient',{});
SIFT_MAX_INTERP_STEPS=5;%Maximum number of iterations
is_local_extream=false;%
xr=0;xc=0;xi=0;%Offset coordinate initialization

for i=1:1:SIFT_MAX_INTERP_STEPS
    current_dog=dog_pyramid{octaves,layer};%Index of the current layer image
    prev_dog=dog_pyramid{octaves,layer-1};%Previous layer image index
    next_dog=dog_pyramid{octaves,layer+1};%Next layer image index
    
    dx=(current_dog(r,c+1)-current_dog(r,c-1))/2;%Partial derivative of X direction
    dy=(current_dog(r+1,c)-current_dog(r-1,c))/2;%Partial derivative of Y direction
    dz=(next_dog(r,c)-prev_dog(r,c))/2;%Partial derivative of sacle(Z) direction
     
    %Calculate the two order partial derivative
    v2=current_dog(r,c)*2;
    dxx=(current_dog(r,c+1)+current_dog(r,c-1)-v2);%Two order partial derivatives of X direction
    dyy=(current_dog(r+1,c)+current_dog(r-1,c)-v2);%Two order partial derivatives of Y direction
    dzz=(next_dog(r,c)+prev_dog(r,c)-v2);%Two order partial derivatives of the scale(Z) direction

    %Mixed two order partial derivative
    dxy=(current_dog(r-1,c-1)+current_dog(r+1,c+1)-...
        current_dog(r-1,c+1)-current_dog(r+1,c-1))/4;
    dxz=(next_dog(r,c+1)-next_dog(r,c-1)-....
        prev_dog(r,c+1)+prev_dog(r,c-1))/4;
    dyz=(next_dog(r+1,c)-next_dog(r-1,c)-....
        prev_dog(r+1,c)+prev_dog(r-1,c))/4;
    
    %Hessian matrix
    H=[dxx,dxy,dxz;
        dxy,dyy,dyz;
        dxz,dyz,dzz];
    %% formula H*X=-[dx,dy,dz]'
    dX=H\([-dx,-dy,-dz]');
    xc=dX(1);%Column direction of the offset, that is, X direction
    xr=dX(2);%Line direction of the offset, that is, Y direction
    xi=dX(3);%Offset of scale,that is, Z direction
    
    if(abs(xc)<0.5 && abs(xr)<0.5 && abs(xi)<0.5)
        is_local_extream=true;%Is the extreme point, exit cycle
        break;
    end
    
    %If the offset is greater than a large amount of data, indicating that the extreme point is not stable, delete
    INT_MAX=max(size(current_dog));
    if(abs(xc)>INT_MAX/3 || abs(xr)>INT_MAX/3 || abs(xi)>INT_MAX/3 )
        is_local_extream=false;%Not extreme points
        break;
    end
    
    %According to the offset from the above, re define the location of the interpolation Center
    SIFT_IMG_BORDER=5;  
    c=c+round(xc);
    r=r+round(xr);
    layer=layer+round(xi);
        
    %If the coordinate range is exceeded, the extreme point is not a feature point
    [M,N]=size(current_dog);
    if(layer<2 || layer>nOctaveLayers+1 ||c<SIFT_IMG_BORDER....
            || c>=N-SIFT_IMG_BORDER ||....
        r<SIFT_IMG_BORDER....
            || r>=M-SIFT_IMG_BORDER)
        is_local_extream=false;%Not extreme points
        break;
    end
end

%If the above is the local extreme points, and then continue to judge
if(is_local_extream==true)
    current_dog=dog_pyramid{octaves,layer};
    prev_dog=dog_pyramid{octaves,layer-1};
    next_dog=dog_pyramid{octaves,layer+1};

    %1 order partial derivative
    dx=(current_dog(r,c+1)-current_dog(r,c-1))/2;
    dy=(current_dog(r+1,c)-current_dog(r-1,c))/2;
    dz=(next_dog(r,c)-prev_dog(r,c))/2;
    
    contr=[dx,dy,dz]*[xc,xr,xi]'*(1/2)+current_dog(r,c);
    if(abs(contr)<(contrast_threshold/nOctaveLayers))
        is_local_extream=false;
    end
end

%% Edge response
if(is_local_extream==true)
    %Calculate the two order partial derivative
    v2=current_dog(r,c)*2;
    dxx=(current_dog(r,c+1)+current_dog(r,c-1)-v2);
    dyy=(current_dog(r+1,c)+current_dog(r-1,c)-v2);
    %Mixed two order partial derivative
    dxy=(current_dog(r-1,c-1)+current_dog(r+1,c+1)-...
        current_dog(r-1,c+1)-current_dog(r+1,c-1))/4;
    tr=dxx+dyy;%Trace of a matrix
    det=dxx*dyy-dxy*dxy;%Determinant of a matrix
    if(det<=0 || (tr*tr*edge_threshold>=det*(edge_threshold+1)^2))
        is_local_extream=false;
    end
end
  
if(is_local_extream==true)
    key_point(1).x=(c+xc)*(2^(octaves-1));
    key_point(1).y=(r+xr)*(2^(octaves-1));
    key_point(1).octaves=octaves;
    key_point(1).layer=layer;
    key_point(1).xi=xi;
    key_point(1).angle=0;
    key_point(1).gradient=0;
    key_point(1).size=sigma*(2^((layer-1+xi)/nOctaveLayers))*(2^(octaves-1));
end
 
end



    
    
    
    
    















