function [stimnumbers] = getpresentationorder(numStim,numBlocks)
                            
stimnumbers = [];
for i=1:numBlocks
    stimnumbers = cat(2,stimnumbers,randperm(numStim));
end

