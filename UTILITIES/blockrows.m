function result = blockrows(data,n)
% -------------------------------------------------------------------------
% this script aggregates over rows in a matrix. vectors are also supported,
% but 3d functionality has not been tested and is presumed not to work.
% aggregation occurs over rows, thus the result will retain the original
% number of columns.
% 
% INPUT ARGUMENTS:
%	data: a m * n array to aggregate over. aggregation occurs over m.
%	n: blocksize, or the number of rows to be included in each block.
% 
% USAGE:
% a = [1,2,3;4,5,6;7,8,9;10,11,12]
% a =
%      1     2     3
%      4     5     6
%      7     8     9
%     10    11    12
% 
% blockrows(a,2)
% ans =
%           2.5          3.5          4.5
%           8.5          9.5         10.5
% -------------------------------------------------------------------------

numrows = size(data,1);
numcols = size(data,2);

if numrows==1
	data=data';
	numrows = numcols;
	numcols = 1;
end

result = reshape(data,[n,numrows/n*numcols]);
result = mean(result);
result = reshape(result, [numrows/n, numcols]);

end
