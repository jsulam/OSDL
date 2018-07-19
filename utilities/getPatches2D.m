function [ Y ] = getPatches2D( I , p , Npatches)
%   [ Y ] = getPatches2D( I , p , Npatches)
% 
%   Gets Npatches 2D patches at random location from image I
%   
%   Inputs: I: Image
%           p: patch size
%           Npatches: # of patches to take. If empty, takes all possible
%           patches
% 
%   Output: Y: Patches in matrix of size (p x p x Npatches)
%
% J. Sulam - Technion
% Jan. 2016

    [N,M]=size(I);
    if isempty(Npatches) || (Npatches > (N-p+1)*(M-p+1))
        Npatches = (N-p+1)*(M-p+1);
    end
    Y=zeros(p,p,Npatches);
    
    i=1;
    j=1;
    cords = randperm((M-p+1)*(N-p+1),Npatches);
    inds_x = zeros(Npatches,1);
    inds_y = zeros(Npatches,1);
    for i=1:Npatches
        inds_x(i) = floor(cords(i)/(M-p+1)-.0001)+1;
        inds_y(i) = cords(i)-(inds_x(i)-1)*(M-p+1);
    end
    
    for n = 1:Npatches
        patch=I(inds_x(n):inds_x(n)+p-1,inds_y(n):inds_y(n)+p-1);
        Y(:,:,n)=patch;
        j=j+1;
        if (j>M-p+1)
            j=1;
            i=i+1;
        end
    end

end

