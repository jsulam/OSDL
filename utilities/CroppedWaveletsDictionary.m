function [Wc] = CroppedWaveletsDictionary(n,wname,rd)
% function Wc = CroppedWaveletsDictionary(n,wname,rd)
% 
% Cropped Wavelets (Synthesis) Dictionary construction.
% 
% Input:    n:      signal dimension
%           wname:  Wavelet name
%           rd:     (optinal) Redundancy option (default 1). If rd==1, then 
%                   the dictionary is redundant by cropping. If rd==0, then 
%                   the output is a regular wavelet (1-D) synthesis 
%                   dictionary.
% 
% Output:   Wc:     Cropped Wavelet Synthesis Dictionary
% 
% References:
% J. Sulam et at, Trainlets: Dictionary Learning in High Dimensions, 2016

if nargin < 3
    rd = 1;
end

if rd
    Nw = 2^(ceil(log2(n))+1);
else 
    Nw = n;
end
dwtmode('per','nodisp');


% Inverse Transform / synthesis dictionary
[c,L]=wavedec(ones(1,Nw),floor(log2(Nw)),wname);
I = eye(length(c));
Wc = zeros(Nw,length(c));
for i=1:size(Wc,2);
    Wc(:,i)=waverec(I(i,:),L,wname);
end
% Cropping
Wc = Wc ( round(Nw/2)-n/2+1 : round(Nw/2)+n/2 ,  1:end  );  
% deleting zero atoms
norms = diag(Wc'*Wc);   
% Normalizing atoms
Wc = NormDict(Wc(:,norms~=0));

end