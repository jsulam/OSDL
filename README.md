# OSDL
Online Sparse Dictionary Learning Algorithm


This is the Matlab Package for the Online Sparse Dictionary Learning (OSDL)
algorithm, presented in:
J. Sulam, B. Ophir, M. Zibulevsky and M. Elad, "Trainlets: Dictionary
Learning in High Dimensions," in IEEE Transactions on Signal Processing, 
vol. 64, no. 12, pp. 3180-3193, June15, 2016.

Version 1.1

INSTALLATION:
- The OSDL requieres to have the OMP-BOX and the OMPS-BOX installed, which 
can be freely downloaded from
http://www.cs.technion.ac.il/~ronrubin/software.html
- Once these packages have their corresponding mex files compiled 
(check their instructions for further details), make sure they are added in
the matlab path.

USAGE:
The main function is OSDL.m, which performs dictionary learning on the 
indictaded training data and outputs a sparse dictionary (refer to the 
referenced paper for more details). OSDL has two basic modes of operation, 
in terms of the training data:
1) If all training data can be stored in memory, it can be provided through
the parameter Ytrain. Other details (such as mini-batch size, etc) can also
be specified.
2) If the training data cannot be stored in memory, OSDL can receive a 
directory to where there training data is stored. In this case, the data 
should be saved in a directory in separate .mat files. Each .mat file 
corresponds to a mini-batch. The name of the variable (with the training 
data in matrix form) in the files should be specified through the 
parameters dataname. 
For example, data could be stored in a 'Data/' directory in minibatches 
Data1.mat, Data2.mat, ... DataN.mat. Each file would contain a matrix 
variable called DataMiniBatch, containing the training samples for that 
mini-batch ordered columnwise.

A demo Script is provided to show the basic usage of OSDL.

-------------------------------------------------------

UPDATES FROM V.1.0:

- Subspace Pursuit (SP) included as an alternative pursuit option. This provides
 a faster and lighter alternative to OMP which which does not scale as well
for high dimensional signals (dimension of 10,000+). Unlike OMP, the
number of iterations of SP can be much smaller than the target sparsity, 
and this implementaion does not require to store in memory the Gram matrix, 
which can be prohibitely large in such cases. This enables efficient 
parallelization of the pursuit as well.

- This version, unlike the previous one, does not require the MatrixMult
package (thanks to Grisha Varksman).

- Another external regularization option was added. In the previous version,
one could select how many times per epoch the dictionary to be cleaned from
unused or repeated atoms. Now it is possible to use an alternative method:
selecting a random subset of atoms (given by the parameter DropAtoms 
between 0 and 1) for the first StopDropOut epochs.
 

-------------------------------------------------------


All comments are welcomed at jsulam@cs.technion.ac.il

Jeremias Sulam
Computer Science Department - Technion
March, 2016.
