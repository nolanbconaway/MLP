function g = sigmoidgrad(z)

%this script returns the gradient of the sigmoid function evaluated at z
g=(sigmoid(z)).*(1-sigmoid(z));

end
