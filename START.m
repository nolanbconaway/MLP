% ------------------------------------------------------------------------
%  This script sets up all the parameters needed to run a simple MLP.
%  The code is currently constrained to a single hidden layer, but the
%  training scripts can be edited for further flexibility.
% ------------------------------------------------------------------------

% initialize the search path
clear;close;clc;
addpath([pwd,'/UTILITIES/']); 

inputs = [-1 -1
		   1  1
		  -1  1
		   1 -1];

targets = [1 0
		   1 0
		   0 1
		   0 1];

% initialize network design and set parameters
model =  struct;
	model.numblocks = 400; % number of runs through the training set
	model.numinitials = 10; % number of randomized models to be averaged across
	model.weightrange = 0.5; % range of inital weight values
	model.numhiddenunits = 5; % # hidden units
	model.learningrate = 0.1; % learning rate for gradient descent
	
% ------------------------------------------------------------------------	
% MLP_block or _trial can be used to train an mlp, updating the weights at
% every trial or by accumulating updates across an blocks.
% ------------------------------------------------------------------------
% result = MLP_block(model,inputs,targets);
result = MLP_trial(model,inputs,targets);


plot(mean(result.training,2))
