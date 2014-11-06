function result= MLP_trial(model)
v2struct(model) %unpack input params

% ----------------------------------------------------------------------------
% DESCRIPTION
%	this script does most of the work for training an MLP. 
%	it creates a result struct containing accuracy over blocks

% INPUT ARGUMENTS:
% 	model is a struct that is assumed to contain:
% 		model.numblocks = 16; % number of passes through the training set
% 		model.numinitials = 50; % number of randomized initalizations
% 		model.weightrange = .5; % range of inital weight values
% 		model.numhiddenunits = 2; % # hidden units
% 		model.learningrate = .15; % learning rate for gradient descent
%	   	model.outputactrule = 'sigmoid'; % options: 'linear', 'sigmoid'
%	
%		model.inputs is an [eg,attribute] matrix of training patterns
%		model.targets is [eg,category] matrix of category assignments
% 
% ----------------------------------------------------------------------------

%   these are optional editables, currently set at default values
	weightcenter=0; % mean value of weights
% ----------------------------------------------------------------------------

result=struct; %initialize the results structure

% initializing some useful variables
numattributes = size(inputs,2);
numitems = size(inputs,1); 
numtargets = size(targets,2);
numupdates = numblocks*numitems;

training=zeros(numupdates,numinitials);

%   Initializing diva and running the simulation
%   ------------------------------------------------------ % 
for modelnumber = 1:numinitials
	
	%  generating initial weights
	[inweights,outweights] = getweights(numattributes, numhiddenunits, ...
		numtargets, weightrange, weightcenter);
	
	%  generating full presentation order
	presentationorder = getpresentationorder(numitems,numblocks);
	
	%   iterate over each trial in the presentation order
	%   ------------------------------------------------------ % 
	for trialnumber = 1:numupdates
		currentinput  =  inputs(presentationorder(trialnumber),:);
		currenttarget =  targets(presentationorder(trialnumber),:);
		
% 		pass activations through model
		[outputactivations,hiddenactivation,hiddenactivation_raw,inputswithbias] = ...
			FORWARDPASS(inweights,outweights,currentinput,outputactrule);
		
% 		determine classification accuracy
		accuracy = outputactivations(currenttarget==1) ./ ...
			sum(outputactivations);
		training(trialnumber,modelnumber)=accuracy;
		
		%   Back-propagating the activations
		%   ------------------------------------------------------ % 
		[outweights, inweights] = BACKPROP(outweights,inweights,...
			outputactivations,currenttarget,hiddenactivation,...  
			hiddenactivation_raw,inputswithbias,learningrate);	  
	end
end

% store perfomance in the result struct
result.training=blockrows(training,numitems);
end
