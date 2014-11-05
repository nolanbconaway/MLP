function g = tanhgrad(z)

%this script returns the gradient of the tanh function evaluated at z
g = 1-tanh(z).^2;

end
