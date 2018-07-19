clc
clear

addpath('utilities\');

%% loading Image and data formation

p = 32; % patch size

Im = double(imread('lena.png'));
Y = getPatches2D( Im , p , []);
Y =  reshape(Y,[p^2,size(Y,3)]);
Y = bsxfun(@minus, Y, mean(Y));

Ytraining = Y;
Ytest = Y(:,end-2e3+1:end);%
clear Y

%% Cropped Wavelets Base Dictionary

phi = CroppedWaveletsDictionary(p,'sym8',2);
m = size(phi,2);
Aini = speye(m^2);
BaseDict{1} = phi;BaseDict{2} = phi;
Tdata = 15;
disp('First OMP')
tic,
Err0 = norm( SparseDict(phi,Aini,omps(BaseDict,Aini,Ytest,[],Tdata)) - Ytest,'fro')/sqrt(numel(Ytest)),
toc,

%% OSDL

params.BatchSize = 512;
params.Tdict = 64;
params.Iter = 2;
params.mode = 'sparse';
params.Tdata = Tdata;
params.dictsep = phi; 
params.Ytrain = Ytraining;

[A,err,errG,times] = OSDL(params);
A = NormDictSep(phi,A);

ErrOSDL = norm( SparseDict(phi,A,omps(BaseDict,A,Ytest,[],Tdata)) - Ytest,'fro')/sqrt(numel(Ytest)),
