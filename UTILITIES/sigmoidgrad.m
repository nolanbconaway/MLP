function g = sigmoidgrad(z)

%this script returns the gradient of the sigmoid function evaluated at z
g=(logsig(z)).*(1-logsig(z));

end
