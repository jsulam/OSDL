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


N = size(Y,2);
[n, m] = size(phi);

tmp = phi' * reshape(Y, [n, n * N]);
tmp = phi' * reshape(tmp', n, m * N);
tmp = reshape(tmp, m * N, m)';
if ~isempty(A)
    X = A' * reshape(tmp, [m^2, N]);
else
    X = reshape(tmp, [m^2, N]);
end

end