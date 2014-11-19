function...
	[outputactivations,hiddenactivation,hiddenactivation_raw,inputswithbias] = ...
		FORWARDPASS(inweights,outweights,...% weight matrices
			inputpatterns,...% activations to be passed through the model
			outputrule) % option for activation rule
				   
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% USAGE
% 	[outputactivations,hiddenactivation,hiddenactivation_raw,inputswithbias] = ...
%		forwardpass(inweights,outweights,inputpatterns,hiddenactrule,outactrule)
% 
% DESCRIPTION
% 	This completes a forward pass, and returns p(cat),as well as any info 
% 	needed for backprop for each activation. This script can be used for 
% 	trial by trial data, or for a vector of inputs.
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% 
% OUTPUT ARGUMENTS
% 	outputactivations: output layer activations
% 	hiddenactivation: hidden layer activations,including bias
% 	hiddenactivation_raw: dot product of inputs and in-hid weights
% 	inputswithbias: input activations, with bias
% 
% INPUT ARGUMENTS
% 	inweights,outweights: weight matrices
% 	inputpatterns (M x N matrix): activations to be passed through the model
% 	outactrule (string): option for activation rule
% 
%-------------------------------------------------------------------------

numitems=size(inputpatterns,1);

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% input and hidden unit propagation
inputswithbias = [ones(numitems,1),inputpatterns]; 
hiddenactivation_raw=inputswithbias*inweights;

% apply hidden node activation rule
hiddenactivation=logsig(hiddenactivation_raw);
hiddenactivation=[ones(numitems,1),hiddenactivation];

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% get output activation
outputactivations=hiddenactivation*outweights;

if strcmp(outputrule,'sigmoid') % applying sigmoid
	outputactivations=logsig(outputactivations);
end



