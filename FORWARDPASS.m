function...
	[outputactivations,hiddenactivation,hiddenactivation_raw,inputswithbias] = ...
		FORWARDPASS(inweights,outweights,...% weight matrices
			inputpatterns,...% activations to be passed through the model
			hiddenactrule,outactrule) % option for activation rules
                   
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
% 	hiddenactrule,outactrule (string): option for activation rule
% 
%-------------------------------------------------------------------------

numitems=size(inputpatterns,1);

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% input and hidden unit propgation
inputswithbias = [ones(numitems,1),inputpatterns]; 
hiddenactivation_raw=inputswithbias*inweights;

% apply hidden node activation rule
if strcmp(hiddenactrule,'sigmoid') % applying sigmoid;
    hiddenactivation=sigmoid(hiddenactivation_raw);
elseif strcmp(hiddenactrule,'tanh')  %applying tanh
    hiddenactivation=tanh(hiddenactivation_raw);
else hiddenactivation=hiddenactivation_raw;
end

% adding a value of 1 to represent the bias unit 
hiddenactivation=[ones(numitems,1),hiddenactivation];

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% get output activaton
outputactivations=hiddenactivation*outweights;

if strcmp(outactrule,'sigmoid') % applying sigmoid
	outputactivations=sigmoid(outputactivations);
elseif strcmp(outactrule,'tanh') %applying tanh
	outputactivations=tanh(outputactivations);
end



