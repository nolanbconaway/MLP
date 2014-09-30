function g = sigmoid(z)

% Compute sigmoid function on scalar or array values z
g = 1.0 ./ (1.0 + exp(-z));

end
