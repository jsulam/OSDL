function [] = ShowState(i,I)
% This function is intended to show an approximate progress of the training
% process.
%
%  [] = ShowState(i,I), i: current State. 
%                       I: Maximum State
% 
% J. Sulam - Technion
% Jan. 2016

C = 5; % every 5%

    if i==1
        fprintf('\n \n')
        fprintf('Training  <<--')
    end
    
    if i<I
        if floor((100*(i+1)/I)/ C) - floor(100*(i/I)/ C) == 1
            progress = C*floor((100*(i+1)/I)/ C);
            fprintf([num2str(progress),'%%-']);
        end
    end
    
    if i==I
        fprintf('->> Finished. \n \n ')
    end
end