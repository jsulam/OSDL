function X = ApplySparseDict(phi,A,y,flag)

switch flag
    case 1
        X = SparseDict(phi,A,y);
    case 2
        X = SparseDictT(phi,A,y);
end

end