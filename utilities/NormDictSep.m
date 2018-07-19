function [Aout,atomnorms] = NormDictSep(phi,A)
%NORMDICTSEP Sparse-Separable Dictionary Normalization.
% 
%   Normalizes the sparse dictionary given by D = PHI*A, where PHI =
%   kron(phi,phi), to unit norm per atom.
% 
%   Input:  phi:    base dictionary (1-D)
%           A:      Sparse Matrix
% 
%   Output: Aout:   Normalized Sparse Matrix
%           atomnorms:  The norms of the input Dictionary
% 
%   J. Sulam - Technion IIT% 
%   Januray 2016

M = size(A,2);
atomnorms = sqrt(sum(SparseDict(phi,[],A).^2,1));
Aout = A*sparse(1:M,1:M,1./atomnorms,M,M);

end
