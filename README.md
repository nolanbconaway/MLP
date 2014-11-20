This project contains code MATLAB code for a simple MLP. It was written for experimentation purposes and is not meant to be a robust tool for PDP modeling.

Run the code using **START.m**. This script will pass a model architecture to **MLP.m**, which trains up a network and return training performance information. 

Model error is evaluated using either SSE for linear output units or Cross-Entropy Error for sigmoid output units. The output unit type can be edited in the START.m script. Note that targets must be within range [0 1] if sigmoid outputs are used.

As-is, **START.M** trains an MLP classifier on a simple XOR dataset, and plots error over training blocks. However, the model can be easily converted to an autoencoder by setting the targets to the inputs. Continuous targets can be be handled by setting the output unit rule to 'linear'.

*Nolan Conaway*

*November 18, 2014*