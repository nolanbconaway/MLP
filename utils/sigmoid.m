function g = sigmoid(x)

% returns sigmoid evaluated elementwize in X
    g = 1 ./ (1+exp(-x)); 

end