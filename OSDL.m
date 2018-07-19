function [A,err,errG,times,times_test,V] = OSDL(params)
% Online Sparse Dictionary Learning algorithm (OSDL).
% 
% [A,err,errG,times] = OSDL (params)
% 
%   Main parameters:
%       - dictsep:          Separable dictionaries per dimension, in matrix
%                           form.
%       - Tdict :           Atom cardinality. Number of non zeros in each
%                           atom.
%       - Ytrain:           Training data (OSDL does not include the
%                           randomization of the data).
%       - datadir:          If training data is not provided, OSDL can also
%                           recieve a directory to data mini-batches. If 
%                           this is the case, dataname should be also 
%                           provided.
%       - dataname:         Name of the matlab variable in the mini-batches
%                           provided in DataDir. Refer to README for
%                           further details and an example on this usage.
%       - Ytest:            Testing data (optional)
%       - initA:            Initial Sparse Dictionary (optional).
%                           Otherwise, the dictionary is initialized with
%                           the identity matrix.
%       - BatchSize:        Batch size, if Ytrain is provided. Otherwise,
%                           the BatchSize is the determined as the size of 
%                           the training data in each file in DataDir.
%       - Iter:             Number of iterations/epochs.
%       - pursuit:          The pursuit algorithm to use: either 'omp' or
%                           'sp'. If 'sp', the provided mode parameter is
%                           ignored (treated as 'sparse'). Default: 'omp'
%       - useGram          Whether to save Gram matrix for omp  
%                           calculations in order to improve performance.
%       - pursuit_test      The pursuit algorithm to use while testing:
%                           either 'omp' or 'sp'. If pursuit is 'sp' or 
%                           'useGram'is false, then pursuit_test will 
%                           always be 'sp'.
%       - mode:             'sparse'/'error', constraint for the Sparse
%                           Coding Stage. 'sparse' provides Tdata number of
%                           nonzeros per training example. 'error' codes
%                           each example so that its energy is not larger
%                           than Tdata.
%       - Tdata:            Data Sparsity constraint (if mode = 'sparse')
%                           or Data fidelity contraint (if mode = 'error')
%       - columnSort        Boolean indicates wether to use column sorting
%                           in hardThres steps (true) or a full matrix 
%                           sort (false). default is true.
%       - colMin            integer represents minimum number of non zero 
%                           indices in each column for hardThres function. 
%                           default is Tdata*0.1
%       - colMax            integer represents maximum number of non zero 
%                           indices in each column for hardThres function. 
%                           default is Tdata*1.4
%		- atomNumberForMutualCheck
%							number of atoms to check when clearing the
%							dictionary from similar atoms	
%       - sp_iterations   number of iterations for sp when it is used as
%                           the pursuit method. Default: 15
%       - cgls_iterations   number of iterations for cgls (conjugate 
%                           gradient least squares) when sp is used as the
%                           pursuit method. Default: 100
% 
%   Auxiliary parameters:
%       - TestingInterval:  Number of dictionary updates before sparse
%                           coding testing data
%       - NCleanDict:       Times per data sweep to clean/prune dictionary
%                           (default: 1)
%       - muthresh:         Mutual Threshold for cleaning Dictionary (0.99
%                           default)
%       - gamma:            Momentum parameter. (0.9 default)
%       - Tmax:             Maximal Training Time. (default = []:
%                           deactivated)       
%       - num_cores:        option for parallel computing (Default 0). If
%                           1, it enables (or uses) matlab's parfor for
%                           distributing the sparse coding stage.
% 
%   Output:
%       - A:                Trained Sparse Dictionary
%       - err:              Residual: average representation error for
%                           'sparse' mode or the average NNZ coeffitients
%                           in 'error' mode
%       - errG:             Generalization error: representation error over
%                           the Testing Data, if provided
%       - times:            A times vector corresponding to each iteration.
% 
% 
% References:
% "Trainlets: Dictionary Learning in High Dimensions", J. Sulam, B. Ophir, 
% M. Zibulevsky and M. Elad, to appear in IEEE Transactions on Signal Processing,
% arXiv:1602.00212v3.



% AVAILABLE ONLINE: 
%  Version 1.0

% My Versions
%  Version: 1.1
%     - Added times_gen to register the time at testing
%     - Added parallelization of sparse coding: NOT WORKING WELL
%     - Added TestingInterval option
%     - Added: Time does not include testing time now.
%     - Added saving intermediate results functionality
% 
% Jeremias Sulam
% jsulam@cs.technion.ac.il
% Technion - IIT
% January 2016

% addpath('utilities\');

% ----- Checking for packages ----- 

if ~exist('omp')
    error('OMP Package missing!');
end
if ~exist('omps')
    error('Sparse OMP Package missing!');
end

% ----- Input Parameters -------

if isfield(params,'dictsep')
    phi = params.dictsep;
    dictsep{1} = phi;
    dictsep{2} = phi;
else
    error('Base dictionary missing!');
end


if isfield(params,'Tdict')
    Tdict = params.Tdict;
else
    error('Dictionary Sparsity missing!');
end


if isfield(params,'initA')
    A = params.initA;
    if ~issparse(A)
        error('Sparse Dictionary matrix A should be sparse!')
    end
else
    m_ = size(dictsep{1},2)^2;
    A = zeros(m_,m_);
    for i = 1 : m_
        A(randperm(m_,Tdict)',i) = randn(Tdict,1);
    end
    A = NormDictSep(phi,sparse(A));
end

if isfield(params,'Ytrain')
    Ytrain = params.Ytrain;
    params.Ytrain = [];
    EXTERNAL_DATA = 0;
else
    if isfield(params,'datadir')
        DATADIR = params.datadir;
        EXTERNAL_DATA = 1;
        if isfield(params,'dataname')
            DATANAME = params.dataname;
        else
            error('Data name not specified');
        end
    else
        error('Training data or data directory missing!');
    end
end

if isfield(params,'save_dir')
    save_dir = params.save_dir;
else
    save_dir = [];
end

if isfield(params,'Tmax')
    Tmax = params.Tmax;
else
    Tmax = [];
end
bandExit = 0;


if isfield(params,'BatchSize')
    K = params.BatchSize;
else
    K = 512;
end

if isfield(params,'Iter')
    Iter = params.Iter;
else
    Iter = 4;
end


if isfield(params,'Ytest')
    Ytest = params.Ytest;
else
    Ytest = [];
end

if isfield(params,'Tdata')
    Tdata = params.Tdata;
else
    error('Sparsity or Error constraint needed.')
end

if isfield(params,'pursuit')
    pursuit = params.pursuit;
else
    pursuit = 'omp';
end


% define omp_bool according to pursuit algorithm
if strcmp(pursuit, 'omp')
    omp_bool=1;
    if isfield(params,'mode')
        mode = params.mode;
    else
        error('MODE needs to be specified (sparse\error).')
    end
else
    omp_bool=0;
    mode = 'sparse';
end

if isfield(params, 'sp_iterations')
    spIter = params.sp_iterations;
else
    spIter = 15;
end
if isfield(params,'cgls_iterations')
    cglsIter = params.cgls_iterations;
else
    cglsIter = 100;
end

if isfield(params,'gamma')
    gamma = params.gamma;
else
    gamma = 0.92;
end

% Extra regularization: How often to perform Dictionary Cleaning or to use
% dictionary subset per minibatch (DropAtoms)

if isfield(params,'NCleanDict')
    Nclean = params.NCleanDict;
    DropAtoms = 0;
    if isfield(params,'DropAtoms')
        warning('Do not specify both NCleanDict and DropAtoms options.');
    end
else
    if isfield(params,'DropAtoms')
        Nclean = 0;
        DropAtoms = params.DropAtoms;
        if DropAtoms < 0 || DropAtoms > 1
            DropAtoms = .5;
        end
        if isfield(params,'StopDropOut')
            StopDropOut = params.StopDropOut;
        else
            StopDropOut = 1; % stop option after 1 epoch
        end
    else % default options
        Nclean = 1;
        DropAtoms = 0;
    end     
end


% mutual threshold for atoms cleaning
if isfield(params,'muthresh')
    muthresh = params.muthresh;
else
    muthresh = 0.98;
end

% hard threshold params
if isfield(params,'columnSort')
    columnSort = params.columnSort;
else
    columnSort = true;
end

if isfield(params,'colMin')
    colMin = params.colMin;
else
    colMin = ceil(Tdict*0.1);
end

if isfield(params,'colMax')
    colMax = params.colMax;
else
    colMax = ceil(Tdict*1.4);
end

if isfield(params,'atomNumberForMutualCheck')
    atomNumberForMutualCheck = params.atomNumberForMutualCheck;
else
    atomNumberForMutualCheck = ceil(size(A, 2));
end

% Momentum variable
if isfield(params,'V')
    if ~isempty(params.V)
        V = params.V;
        params.V = [];
        V = double(V);
    else
        V = zeros(size(A)); 
    end  
else
    V = zeros(size(A));    
end

% Parallel options
if isfield(params,'num_cores')
    parallel_opt = params.num_cores;
    if parallel_opt
        pool = gcp('nocreate');
        if isempty(pool) && params.num_cores > 1
            pool = parpool(params.num_cores);
        end
        Ncores = pool.NumWorkers;
    end
else
    parallel_opt = 0;
end

if isfield(params,'useGram') && ~parallel_opt
    useGram = params.useGram;
else
    useGram = 1;
end

if isfield(params,'pursuit_test')
    pursuit_test = params.pursuit_test;
    if strcmp(pursuit_test, 'omp') && omp_bool && useGram
        omp_test = 1;
    else
        omp_test = 0;
    end
else
    omp_test = 0;
end

% Init Variables for external data, if enabled
if EXTERNAL_DATA
    DATALIST = what(DATADIR);
    DATALIST = DATALIST.mat;
    NumBatches = length(DATALIST);
end

if ~EXTERNAL_DATA
    N = size(Ytrain,2);
    NumBatches = floor(N/K);
    means = mean(Ytrain);
    for i = 1 : length(means)
        Ytrain(:,i) = Ytrain(:,i) - means(i);
    end
    % Ytrain = Ytrain./max(Ytrain(:));
end

if ~isempty(Ytest)
    means = mean(Ytest);
    for i = 1 : length(means)
        Ytest(:,i) = Ytest(:,i) - means(i);
    end
end
TIME_TEST = 0;
TIME_SAVE = 0;

% How often to evaluate Test, if enabled
if isfield(params,'TestingInterval')
    TESTING_INTERVAL = params.TestingInterval;
else
    TESTING_INTERVAL = NumBatches;
end

% Saving intermediate results?
if isfield(params,'SavingInterval')
    SAVE_FLAG = 1;
    if params.SavingInterval > 0
        SAVING_INTERVAL = params.SavingInterval;
    else
        SAVING_INTERVAL = NumBatches;
    end
else
    SAVE_FLAG = 0;
end


% ---- Init and Variable Definition ---------

[m1,m2] = size(A);

Ik = speye(m1);

% control variables
err = zeros(Iter*NumBatches,1);
if ~isempty(Ytest)
    times_test = zeros(Iter*floor(NumBatches/TESTING_INTERVAL),1);
    errG = zeros(Iter*floor(NumBatches/TESTING_INTERVAL),1);
else
    times_test = [];
    errG = [];
end
times = zeros(Iter*NumBatches,1);

% Dictionary variables
PHI = SparseDict(phi,[],speye(size(phi,2)^2));
Dnorms = sqrt(sum(SparseDict(phi,[],A).^2,1));  
Inorms = sparse(1:length(Dnorms),1:length(Dnorms),Dnorms.^(-1),length(Dnorms),length(Dnorms),length(Dnorms));    
An = sparse(A*Inorms);
if omp_bool
	GPHI = SparseDictT(phi,[],PHI);
	PHI = [];
    G = A'*GPHI*A;
    if useGram
        Gn = Inorms*G*Inorms;
    else
        Gn = [];
    end
else
	GPHI=[];
    G=[];
    Gn=[];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             Main Loop                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


Gene = tic;

tic_test = tic;

atoms_dropout_histogram = zeros(m2,1);

for iter = 1 : Iter
   
    usecount = zeros(m2,1);  % counts how many times each atom has been used per datasweep.*
    TotErr = 0;
    TotPsnr = 0;
    samples_counter = 0;
    
    %%% MiniBatches Iterations %%%
    for k = 1 : NumBatches
        
        fprintf('Batch %i of %i. ',k,NumBatches)
        
        if ~EXTERNAL_DATA
            Ybatch = Ytrain(:,1+(k-1)*K:k*K);
        else
            load([DATADIR,'/',DATALIST{k}]);
            Ybatch = double(eval(DATANAME));
            meansBatch = mean(Ybatch);
            for i = 1 : length(meansBatch)
                Ybatch(:,i) = Ybatch(:,i) - meansBatch(i);
            end
            %Ybatch = bsxfun(@minus, Ybatch, mean(Ybatch));

            clear(DATANAME);
        end
        
        %%% Sparse Coding %%%
        
        % option for sub-dictionary
        if DropAtoms~=0 && iter <= StopDropOut
            mask = sort(randperm(size(An,2),round(size(An,2)*DropAtoms)));
            atoms_dropout_histogram(mask) = atoms_dropout_histogram(mask)+1;
            Asub = An(:,mask);
            if omp_bool
                Gn_sub = Gn(:,mask);
                Gn_sub = Gn_sub(mask,:);
            end
        else
            Asub = An;
            if omp_bool
                Gn_sub = Gn;
            end
            mask = 1:size(An,2);
        end
        
        Xt = zeros(size(A,2),size(Ybatch,2));
        if omp_bool
            if strcmp(mode,'sparse')
%                 if parallel_opt
%                     Xt = OMP_p_G(An,phi,Ybatch,Tdata,Ncores,Gn);
%                 else
                    if useGram
                        proj = SparseDictT(phi,Asub,Ybatch);
                        Xt(mask,:) = omp(proj,Gn_sub,Tdata);
                    else
                        Xt = zeros(size(A,2),size(Ybatch,2));
                        dict = SparseDict(phi, [], Asub);
                        Xt(mask,:) = omp(dict,Ybatch, [],Tdata);
                    end
%                 end
            else
%                  if parallel_opt
%                     Xt = omp2_par(proj,sum(Ybatch.*Ybatch),Gn,Tdata);
%                  else
                    if useGram
                        proj = SparseDictT(phi,Asub,Ybatch);
                        Xt(mask,:) = omp2(proj,sum(Ybatch.*Ybatch),Gn_sub,Tdata);
                    else
                        Xt = zeros(size(A,2),size(Ybatch,2));
                        dict = SparseDict(phi, [], Asub);
                        Xt(mask,:) = omp2(dict,Ybatch, [],Tdata);
                    end
%                  end
            end
            Xt = Inorms*Xt;
        else
            N = size(Ybatch, 2);
			Xt(mask,:) = ParallelSP(size(Asub,2), N, Tdata, phi, Asub, Ybatch, spIter, cglsIter);
            Xt = Inorms*Xt;
        end
        
        %%% Fidelity Error calc %%%
        X_rec = SparseDict(phi,A,Xt);
        Er =  X_rec - Ybatch;

        %%% gradient calc %%%
        aux = sum(abs(Xt),2)';
        [~,ChosenAtoms] = find(aux~=0);
        ALFA = (Er*Xt(ChosenAtoms,:)');
        gradA = zeros(size(A));
        gradA(:,ChosenAtoms) = SparseDictT(phi,[],ALFA);
        
        %%% Step Size %%%
        num = norm(gradA(:,ChosenAtoms),'fro');
        BETA = gradA(:,ChosenAtoms)*Xt(ChosenAtoms,:);
        AUX = SparseDict(phi,[],BETA);
        den = norm(AUX,'fro');
        mu = num / den;
        
        %%% update of Sparse Dictionary A %%%
        V = gamma*V + mu*gradA(:,:);
        update = A(:,ChosenAtoms) - V(:,ChosenAtoms);
        A(:,ChosenAtoms) = sparse(HardThres(update,Tdict,columnSort,colMin,colMax));  
        
        %%% Gram and Dictionary Update %%%
        if ~isempty(ChosenAtoms)
            %%% update of normalization %%%
            Dnorms(ChosenAtoms) = sqrt(sum(SparseDict(phi,[],A(:,ChosenAtoms)).^2,1));
            Inorms = sparse(1:m2,1:m2,Dnorms.^(-1),m2,m2,m2); 
            An(:,ChosenAtoms) = sparse(A(:,ChosenAtoms)*Inorms(ChosenAtoms,ChosenAtoms));
            if omp_bool
                new_atoms = SparseDict(phi,[],A(:,ChosenAtoms));
                gram_cols = SparseDictT(phi,A,new_atoms);
                G(:,ChosenAtoms) = gram_cols;
                G(ChosenAtoms,:) = gram_cols';
                if useGram
                    Gn = Inorms*G*Inorms;
                end
            end
        end
        
        usecount = usecount + sum(abs(Xt)>1e-7, 2);

        if (Nclean~=0) && (rem(k,floor(NumBatches/Nclean)) == 0) && (iter<Iter)
            [A,G,changed_atoms] = cleardict(dictsep,PHI,A,An,Ybatch,GPHI,G,Gn,Ik,Tdict,Tdata,useGram,muthresh,atomNumberForMutualCheck,usecount,omp_bool);
            % fprintf(['[',num2str(length(changed_atoms)),']'])
            %%% update of normalization %%%
            if ~isempty(changed_atoms) && omp_bool
                Dnorms(changed_atoms) = sqrt(sum(SparseDict(phi,[],A(:,changed_atoms)).^2,1));
                Inorms = sparse(1:m2,1:m2,Dnorms.^(-1),m2,m2,m2); 
                An(:,changed_atoms) = sparse(A(:,changed_atoms)*Inorms(changed_atoms,changed_atoms));
                if useGram
                    Gn = Inorms*G*Inorms;
                end
            end
        end 

        %%% Testing Data, if enabled
        if ~isempty(Ytest) && mod(k,TESTING_INTERVAL)==0
            ind = k/TESTING_INTERVAL+(iter-1)*floor(NumBatches/TESTING_INTERVAL);
            times_test(ind)=toc(tic_test);
            Ttesting = tic;
            
            if omp_test
                if strcmp(mode,'sparse')
                    Xtest = Inorms*omps(dictsep,An,Ytest,Gn,Tdata);
                    errG(ind) = norm( SparseDict(phi,A,Xtest) - Ytest,'fro')/sqrt(numel(Ytest)); clear Xtest
                else
                    Xtest = omps2(dictsep,An,Ytest,Gn,Tdata);
                    errG(ind) = nnz(Xtest)/(numel(Ytest)); clear Xtest;
                end
            else
                N = size(Ytest, 2);
                Xtest = ParallelSP(m2, N, Tdata, phi, A, Ytest, [], cglsIter);
                errG(ind) = norm( SparseDict(phi,A,Xtest) - Ytest,'fro')/sqrt(numel(Ytest)); clear Xtest
            end
            TIME_TEST = TIME_TEST + toc(Ttesting);
        end
               
        %%% control variables update
        if strcmp(mode,'sparse')
            err(k+(iter-1)*NumBatches) = norm(Er,'fro')/sqrt(numel(Ybatch));
        else
            err(k+(iter-1)*NumBatches) = nnz(Xt)/(numel(Ybatch));
        end
            
        if SAVE_FLAG && mod(k,SAVING_INTERVAL)==0
            Tsaving = tic;
            save([save_dir, 'Results_aux_', date, '_pursuit_', pursuit, '_Gram=', num2str(useGram), '_parallel=', num2str(parallel_opt) , '_colSort=', num2str(columnSort), '_colMin=', num2str(colMin),'_Tdict=', num2str(Tdict), '_randomness=', num2str(atomNumberForMutualCheck)],'A','V','err','times','errG', 'times_test', 'NumBatches')
            TIME_SAVE = TIME_SAVE + toc(Tsaving);
        end
        times(k+(iter-1)*NumBatches) = toc(Gene) - TIME_TEST - TIME_SAVE;

        
        if ~isempty(Tmax)
            if times(k+(iter-1)*NumBatches)>Tmax,
                bandExit = 1;
                break;
            end
        end
        
        % stats
        TotErr = TotErr+ (err(k+(iter-1)*NumBatches))*K;
        samples_counter = samples_counter + K;
        AvErr = TotErr/samples_counter;
        TotPsnr = TotPsnr + psnr(X_rec/255,Ybatch/255);
        AvPSNR = TotPsnr / samples_counter;
        fprintf('Error: %.2f. Current PSNR: %.2f. Av. Time: %.2f s\n',AvErr, psnr(X_rec/255,Ybatch/255),toc(Gene)/k)
                
    end
    
    if bandExit
        fprintf('\nFinishing for Maximal Time Condition\n\n');
        break;
    end
    
end

save([ 'Results_aux_', date, '_pursuit_', pursuit, '_Gram=', num2str(useGram), '_parallel=', num2str(parallel_opt) , '_colSort=', num2str(columnSort), '_colMin=', num2str(colMin),'_Tdict=', num2str(Tdict), '_randomness=', num2str(atomNumberForMutualCheck)],'A','err','times','errG', 'times_test', 'NumBatches')

if parallel_opt
    delete(gcp());
end

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           cleardict                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [A,G,cleared_atoms] = cleardict(basedict,PHI,A,An,X,baseG,G,Gn,Ik,Tdict,Tdata,useGram,muthresh,atomNumberForMutualCheck,usecount,omp_bool) 
    % check for similar atoms
    use_thresh = 1;  % at least this number of samples must use the atom to be kept
    nnz = 3;
    [m1,dictsize] = size(A);
    phi = basedict{1};
    s = size(X,2);
    K = min(1e3,s);
    inds = randperm(s,K); % random set of K signals for computation of representation error
    Y=X(:,inds);
    if omp_bool
        Gamma = omps(basedict,An,Y,Gn,nnz);  
    else
		dict = SparseDict(phi, [], A);
		Gamma = ParallelSP(dictsize, K, nnz, phi, A, Y, 5, 10);
    end
    err = sum((Y - dictsep(basedict,A,Gamma)).^2);

	cleared_atoms = [];
	inds2 = randperm(dictsize, min(dictsize, atomNumberForMutualCheck));

	if omp_bool && useGram
		for j = 1:dictsize
            if usecount(j)<use_thresh
                [atom, err] = makeAlternativeAtomOMP(A, err, basedict, Ik, X, baseG, Tdict, inds);
				A(:, j) = atom;
                cleared_atoms(length(cleared_atoms)+1) = j;
				inds2 = inds2(find(inds2 ~= j));
            end
        end
		inds3 = inds2;
		for j = 1:length(inds2)
            Gj = Gn(inds3,inds2(j));
            Gj(1) = 0;
			inds3(1) = [];

            % replace atom if its too similar to another one or if it has 
            % barely been used
            if max(abs(Gj))>muthresh
                [atom, err] = makeAlternativeAtomOMP(A, err, basedict, Ik, X, baseG, Tdict, inds);
				A(:, inds2(j)) = atom;
                cleared_atoms(length(cleared_atoms)+1) = inds2(j);
            end
        end
		
		cleared_atoms = sort(cleared_atoms);
		
	else
		for j = 1:dictsize
			if usecount(j) < use_thresh
                [atom, err] = makeAlternativeAtomSP(A, err, Tdict, phi, X, inds);
				A(:, j) = atom;
			end
		end
		for j1 = 1:length(inds2)
			inds3 = inds2;
			inds3(j1) = [];
			atom1 = dict(:, inds2(j1));
			norm1 = norm(atom1);
			for j2 = 1:length(inds3)
				atom2 = dict(:, inds3(j2));
				if abs(dot(atom1, atom2))/norm1/norm(atom2) > muthresh
					[atom, err] = makeAlternativeAtomSP(A, err, Tdict, phi, X, inds);
					A(:, inds2(j1)) = atom;
                    break;
				end
			end
		end
	end

    if ~isempty(cleared_atoms) && omp_bool
        % update D and G
		new_atoms = SparseDict(phi,[],A(:,cleared_atoms));
        gram_cols = A'*SparseDictT(phi,[],new_atoms);
		G(:,cleared_atoms) = gram_cols;
		G(cleared_atoms,:) = gram_cols';
    end
end

function [atom, err] = makeAlternativeAtomSP(A, err, Tdict, phi, X, inds)
    if sum(err~=0) == 0
        i = randi([1 length(inds)], 1, 1);
    else    
        [~,i] = max(err); 
        err(i) = 0;
    end
	rec = SP(Tdict, phi, speye(size(phi, 2)^2), X(:,inds(i)), [], 10);
	atom = rec.x_hat;
end

function [atom, err] = makeAlternativeAtomOMP(A, err, basedict, Ik, X, baseG, Tdict, inds)
    if sum(err~=0) == 0
        i = randi([1 length(inds)], 1, 1);
    else    
        [~,i] = max(err); 
        err(i) = 0;
    end
	atom = omps(basedict, Ik, X(:,inds(i)), baseG, Tdict, 'checkdict', 'off');
	d = dictsep(basedict, Ik, atom);
	atom = atom./norm(d);
end
