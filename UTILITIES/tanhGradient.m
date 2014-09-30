function g = tanhGradient(z)

%this script returns the gradient of the tanh function evaluated at z
g = zeros(size(z)); g=1-(hyperbolic_tangent(z).*hyperbolic_tangent(z));

end
