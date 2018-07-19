function Rec = SP(K, phi, A, y, spIterations, cglsIterations)
% For algorithm description, explanation and analysis, please check
% Wei Dai and Olgica Milenkovic
% "Subspace Pursuit for Compressive Sensing: Closing the
% Gap Between Performance and Complexity"%

    if (nargin < 4 || isempty(spIterations))
        spIterations = 100;
    end
    
    if (nargin < 5 || isempty(spIterations))
        cglsIterations = [];
    end

	m = size(phi, 1)^2;
	N = size(A, 2);

    y_r = y;
    in = 1;

    %cv = abs(y_r' * Phi);
	cv = abs(SparseDictT(phi, A, y_r))';
    [~, cv_index] = maxk(cv, K, 2, 'sorting', false);
    cv_index = sort(cv_index);
    %Phi_x = Phi(:, cv_index);
	A_x = A(:, cv_index);
    Index_save(in, :) = cv_index;

    %x_p = cgls(Phi_x, y, [], [], cglsIterations);
	func_h = @(y, tflag) ApplySparseDict(phi, A_x, y, tflag);
	x_p = cgls(func_h, y, [], [], cglsIterations);
    %y_r = y - Phi_x * x_p;
	y_r = y - SparseDict(phi, A_x, x_p);
    norm_save(in) = norm(y_r);

    for j = 1 : spIterations
        in = in + 1;

        % find T^{\prime} and add it to \hat{T}
        %cv = abs(y_r' * Phi);
		cv = abs(SparseDictT(phi, A, y_r))';
        [~, cv_index] = maxk(cv, K, 2, 'sorting', false);
        cv_add = union(Index_save(in - 1, :), sort(cv_index));
        %Phi_x = Phi(:,cv_add);
		A_x = A(:, cv_add);

        % find the most significant K indices
        %x_p = cgls(Phi_x, y, [], [], cglsIterations);
		func_h = @(y, tflag) ApplySparseDict(phi, A_x, y, tflag);
		x_p = cgls(func_h, y, [], [], cglsIterations);
        [~, i_sort] = maxk(abs(x_p), K, 1, 'sorting', false);
        cv_index = cv_add(sort(i_sort));
        %Phi_x = Phi(:, cv_index);
		A_x = A(:, cv_index);
        Index_save(in, :) = cv_index;

        % calculate the residue
        %x_p = cgls(Phi_x, y, [], [], cglsIterations);
		func_h = @(y, tflag) ApplySparseDict(phi, A_x, y, tflag);
		x_p = cgls(func_h, y, [], [], cglsIterations);
        %y_r = y - Phi_x*x_p;
		y_r = y - SparseDict(phi, A_x, x_p);

        norm_save(in) = norm(y_r);

        if ( norm_save(in) <= eps ) | ...
            (norm_save(in)/norm_save(in-1) >= 1)
            break;
        end
    end

    x_hat = zeros(N, 1);
    x_hat(Index_save(in,:)) = reshape(x_p, K, 1);
    Rec.T = Index_save;
    Rec.x_hat = x_hat;
    Rec.PResidue = norm_save;
end