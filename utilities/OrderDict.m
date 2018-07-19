function Dout = OrderDict(Din,opt,mode)

% function Dout = OrderDict(Din,opt)
%   Sorts dictionary atoms by decreasing variance (opt=1)
%   or decreasing entropy (opt2)
% 
% Jeremias Sulam - Technion

[n,m] = size(Din);
Dout = zeros(size(Din));


switch opt
    case 1 % variance
        vars = std(Din,0,1);
        [~,ind] = sort(vars,mode);
        Dout = Din(:,ind); 
       	return
    case 2 % entropy
        ent = zeros(m,1);
        for i = 1 : m
            ent(i) = entropy(reshape(Din(:,i),[sqrt(n),sqrt(n)]));
        end
        [~,ind] = sort(ent,mode);
        Dout = Din(:,ind); 
       	return
    case 3
        Tv = zeros(m,1);
        for i = 1 : m
            [Gx,Gy] = gradient(reshape(Din(:,i),[sqrt(n),sqrt(n)]));
            Tv(i) = sum(abs(Gx(:))+abs(Gy(:)));
        end
        [~,ind] = sort(Tv,mode);
        Dout = Din(:,ind); 
        
end


end