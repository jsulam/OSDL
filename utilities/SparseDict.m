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

N = size(X,2);
[n, m] = size(phi);

if ~isempty(A)
    tmp = full(A * X);
else
    tmp = full(X);
end
tmp = phi * reshape(tmp, [m, m * N]);
tmp = phi * reshape(tmp', m, n * N);
tmp = reshape(tmp, n * N, n)';
Y = reshape(tmp, [n^2, N]);


end