function Y = SparseDict(phi,A,X)
% SparseDict: Applies the Sparse Dictionary A, with base dictionary phi, to
% the data in X. 
% 
%   [Y] = SparseDict(phi,A,X): returns the results of the product PHI*A*X
% 
%   [Y] = SparseDict(phi,[],X): returns the results of the product PHI*X
% 
%   J. Sulam - Technion IIT% 
%   Januray 2016

[m1] = size(phi,2)^2;
n = size(phi,1)^2;
N = size(X,2);

if ~isempty(A)
    Y = reshape( mtimesx(mtimesx(phi,reshape(full(A*X),[sqrt(m1) sqrt(m1) N])),phi'),[n,N]);
else
    Y = reshape( mtimesx(mtimesx(phi,reshape(full(X),[sqrt(m1) sqrt(m1) N])),phi'),[n,N]);
end


end