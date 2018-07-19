function Rec = CSRec_SP(K,Phi,y)
% For algorithm description, explanation and analysis, please check
% Wei Dai and Olgica Milenkovic
% "Subspace Pursuit for Compressive Sensing: Closing the
% Gap Between Performance and Complexity"%

[m,N]=size(Phi);

y_r = y;
in = 1;

cv = abs( y_r'*Phi );
[cv_sort, cv_index] = sort(cv,'descend');
cv_index = sort( cv_index(1:K) );
Phi_x = Phi(:,cv_index);
Index_save(in,:) = cv_index;

x_p = inv(Phi_x'*Phi_x)*Phi_x' * y;
y_r = y - Phi_x*x_p;
norm_save(in) = norm(y_r);

while 1
   in = in+1;

   % find T^{\prime} and add it to \hat{T}
   cv = abs( y_r'*Phi );
   [cv_sort, cv_index] = sort(cv,'descend');
   cv_index = sort( cv_index(1:K) );
   cv_add = union(Index_save(in-1,:), cv_index);
   Phi_x = Phi(:,cv_add);

   % find the most significant K indices
   x_p = inv(Phi_x'*Phi_x)*Phi_x' * y;
   [x_p_sort, i_sort] = sort( abs(x_p) , 'descend' );
   cv_index = cv_add( i_sort(1:K) );
   cv_index = sort( cv_index );
   Phi_x = Phi(:,cv_index);
   Index_save(in,:)=cv_index;

   % calculate the residue
   x_p = inv(Phi_x'*Phi_x)*Phi_x' * y;
   y_r = y - Phi_x*x_p;

   norm_save(in) = norm(y_r);

   if ( norm_save(in) == 0 ) | ...
           (norm_save(in)/norm_save(in-1) >= 1)
       break;
   end
end

x_hat = zeros(N,1);
x_hat( Index_save(in,:) ) = reshape(x_p,K,1);
Rec.T = Index_save;
Rec.x_hat = x_hat;
Rec.PResidue = norm_save;