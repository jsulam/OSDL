function X = HardThres2(X,k,columnSort, colMin, colMax) 
    % X = HardThres(X,k)
    % 
    % HardThresholding function that keeps the k highest coefficients (in
    % absolute value) PER COLUMN in the matrix X
    % MegaFace_dataset.tar.gz.part
    % Jeremias Sulam
    % July 2015
    % CS - Technion
    k = ceil(k);
    [n,m] = size(X);

    if columnSort
        [~,inds] = sort(abs(X),1,'descend');

        for i = 1 : m
            X(inds(k+1:end,i),i) = 0;
        end
    
        X = sparse(X);
    
    else
        Ktotal = k*m;
        kMin = m*colMin;
        K = Ktotal - kMin;  % total number of nonzeros to take from the middle portion
        colMax = min(K,n);
        Signs = sign(X);
        [vals,indsX] = maxk(abs(X),colMax,1);
        indsMed = indsX(colMin+1:colMax,:);
        vals_med = vals(colMin+1:colMax,:);
        vals_med = vals_med(:);
        [~,inds] = maxk(vals_med,K,1,'sorting',false);
        
        [indsToKeep1, indsToKeep2] = ind2sub([colMax - colMin, m], inds);
        X = zeros(size(X));
        for i = 1:m
            X(indsX(1:colMin,i),i) = vals(1:colMin,i);
        end
        
        for i = 1:K
            X(indsMed(indsToKeep1(i), indsToKeep2(i)),indsToKeep2(i)) = vals_med(inds(i));
        end
        
        X = X.*Signs;
        
    end
    
    
end