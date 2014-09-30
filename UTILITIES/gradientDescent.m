function [updated_wts] = gradientDescent(learningrate,input,delta,wts)

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % this script conducts gradient descent on the parameters provided above
% % 
% % learningrate = obvious
% % input = incoming activations (i.e, from layer n-1, or, x)
% % delta =  error on target values (may be backpropagated or not).
% % wts = obvious

%     compute partial derivative
    bigDelta = input' * delta;

%     compute the value of the weight update
    wtUpdate = learningrate*bigDelta;

%     execute the update
    updated_wts = wts - (wtUpdate);


