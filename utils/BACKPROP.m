function [outweights, inweights] = BACKPROP(...
			outweights,inweights,... 	% weights to be updated
			outputactivations,...		% predictions produced by the network
			currenttarget,...			% target activations
			hiddenactivation,...	 	% hidden recodings passed to outputs
			hiddenactivation_raw,... 	% raw hidden layer recodings
			inputswithbias,...	   		% model inputs [with bias]
			learningrate)				% learning rate parameter
	
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% USAGE
% [outweights, inweights] = BACKPROP(outweights,inweights,...
%	 outputactivations,currenttarget,hiddenactivation,...
%	 hiddenactivation_raw,inputswithbias,learningrate)
% 
% DESCRIPTION
% 	This completes a backward pass of prediction error, and returns
% 	updated weight matrices. The code is general and can handle trial-based
% 	as well as block updates.
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% 
% INPUT ARGUMENTS
%   outweights,inweights		weights to be updated
%   outputactivations	   		predictions produced by the network
%   currenttarget		   		target activations
%   hiddenactivation			hidden recodings passed to outputs
%   hiddenactivation_raw		raw hidden layer recodings
%   inputswithbias		  		model inputs [with bias]
%   learningrate				learning rate parameter	
% 
%-------------------------------------------------------------------------

%  obtain error on the output units
outputdelta = 2*(outputactivations - currenttarget);

%  obtain error on the hidden units
hiddendelta=outputdelta*outweights';
hiddendelta=hiddendelta(:,2:end).*sigmoidgrad(hiddenactivation_raw);

%  compute weight changes
outputdelta = learningrate * hiddenactivation' * outputdelta;
hiddendelta = learningrate * inputswithbias'   * hiddendelta;

%  adjust weights
outweights = outweights - outputdelta;
inweights  = inweights - hiddendelta;

% set min and max value to prevent NaNs
outweights(outweights >  1e+100) =  1e+100;
inweights(inweights   >  1e+100) =  1e+100;
outweights(outweights < -1e+100) = -1e-100;
inweights(inweights   < -1e+100) = -1e-100;

end