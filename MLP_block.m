function result = MLP_block(model)

% ----------------------------------------------------------------------------
% DESCRIPTION
%	this script does most of the work for training an MLP. 
%	it creates a result struct containing accuracy over blocks

% INPUT ARGUMENTS:
% 	model is a struct that is assumed to contain:
% 		model.numblocks = 16; % number of passes through the training set
% 		model.numinitials = 50; % number of randomized initializations
% 		model.weightrange = .5; % range of initial weight values
% 		model.numhiddenunits = 2; % # hidden units
% 		model.learningrate = .15; % learning rate for gradient descent
%	   	model.outputactrule = 'sigmoid'; % options: 'linear', 'sigmoid'
% 
%		model.inputs is an [eg,attribute] matrix of training patterns
%		model.targets is [eg,category] matrix of category assignments
% ----------------------------------------------------------------------------

%   these are optional editables, currently set at default values
	weightcenter=0; % mean value of weights
% ----------------------------------------------------------------------------

result=struct; %initialize the results structure
v2struct(model) %unpack input params

% initializing some useful variables
numattributes = size(inputs,2);
numtargets = size(targets,2);

training=zeros(numblocks,numinitials);

%   Initializing the network and running the simulation
%   ------------------------------------------------------ % 
for modelnumber = 1:numinitials
	
	%  generating initial weights
	[inweights,outweights] = getweights(numattributes, numhiddenunits, ...
		numtargets, weightrange, weightcenter);
	
	%   iterate over each trial in the presentation order
	%   ------------------------------------------------------ % 
	for blocknumber = 1:numblocks
	   
% 		pass activations through model
		[outputactivations,hiddenactivation,hiddenactivation_raw,inputswithbias] = ...
			FORWARDPASS(inweights,outweights,inputs,outputactrule);

% 		determine classification accuracy
		[~,indecies] = max(targets,[],2);
		accuracy = diag(outputactivations(:,indecies)) ./ sum(outputactivations,2);
		training(blocknumber,modelnumber)=mean(accuracy);		

		%   Back-propagating the activations
		%   ------------------------------------------------------ % 
		[outweights, inweights] = BACKPROP(outweights,inweights,...
			outputactivations,targets,hiddenactivation,...  
			hiddenactivation_raw,inputswithbias,learningrate);
	end
end

% store performance in the result struct
result.training=training;
end
