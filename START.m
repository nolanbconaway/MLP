% ------------------------------------------------------------------------
%  This script sets up all the parameters needed to run a simple MLP.
%  The code is currently constrained to a single hidden layer, but the
%  training scripts can be edited for further flexibility.
% ------------------------------------------------------------------------

% initialize the search path
clear;close;clc;
addpath([pwd '/utils/']); 

% initialize network design and set parameters
model =  struct;
	model.numblocks = 200; % number of runs through the training set
	model.numinitials = 2; % number of randomized models to be averaged across
	model.weightrange = 1; % range of initial weight values
	model.numhiddenunits = 3; % # hidden units
	model.learningrate = 0.25; % learning rate for gradient descent
	model.outputrule = 'sigmoid'; % options: 'linear', 'sigmoid'

model.inputs = [-1 -1
				 1  1
				-1  1
				 1 -1];

model.targets =[1 0
                1 0
                0 1
                0 1];
       
result = MLP(model);

plot(result.training)
v=axis;
v(3:4)=[0 1];
axis(v);