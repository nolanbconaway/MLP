function [inweights,outweights] = getweights(...
	numattributes, numhiddens, numtargets, weightrange, weightcenter)


bias=1;

% GENERATE WEIGHTS BETWEEN INPUT AND HIDDEN LAYER
% -----------------------------------------------
inweights = 2 * (rand(numattributes + bias, numhiddens) - 0.5);
inweights = weightcenter + (weightrange * inweights); 


% GENERATE WEIGHTS BETWEEN HIDDEN AND OUTPUT LAYER
% ------------------------------------------------
outweights = 2 * (rand(numhiddens + bias, numtargets) - 0.5); 
outweights = weightcenter + (weightrange * outweights); 
