function X = SparseDictT(phi,A,Y)
% SparseDictT: Applies the Transpose of the sparse Dictionary A, with base 
% dictionary phi, to the data in Y. 
% 
%   [X] = SparseDictT(phi,A,Y): returns the results of the product 
%                               (PHI*A)^T*Y
% 
%   [X] = SparseDict(phi,[],Y): returns the results of the product PHI^T*Y
% 
%   J. Sulam - Technion IIT% 
%   Januray 2016


[m1] = size(phi,2)^2;
n = size(phi,1)^2;
N = size(Y,2);

if ~isempty(A)
    X = A'*reshape( mtimesx(mtimesx(phi',reshape(Y,[sqrt(n) sqrt(n) N]),'speed'),phi,'speed'),[m1,N]);
else
    X = reshape( mtimesx(mtimesx(phi',reshape(Y,[sqrt(n) sqrt(n) N]),'speed'),phi,'speed'),[m1,N]);
end

end