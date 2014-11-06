% ------------------------------------------------------------------------
%  This script sets up all the parameters needed to run a simple MLP.
%  The code is currently constrained to a single hidden layer, but the
%  training scripts can be edited for further flexibility.
% ------------------------------------------------------------------------

% initialize the search path
clear;close;clc;
addpath([pwd,'/UTILITIES/']); 

% initialize network design and set parameters
model =  struct;
	model.numblocks = 200; % number of runs through the training set
	model.numinitials = 2; % number of randomized models to be averaged across
	model.weightrange = 1; % range of initial weight values
	model.numhiddenunits = 3; % # hidden units
	model.learningrate = 0.15; % learning rate for gradient descent
	model.outputactrule = 'sigmoid'; % options: 'linear', 'sigmoid'

model.inputs = [-1 -1
				 1  1
				-1  1
				 1 -1];

model.targets =[1 0
                1 0
                0 1
                0 1];
       
% ------------------------------------------------------------------------	
% MLP_block or _trial can be used to train an mlp, updating the weights at
% every trial or by accumulating updates across an blocks.
% ------------------------------------------------------------------------
result = MLP_block(model);
% result = MLP_trial(model);


plot(mean(result.training,2))
v=axis;
v(3:4)=[0 1];
axis(v);