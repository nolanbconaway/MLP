This project contains code MATLAB code for a simple MLP classifier. It was written for experimentation purposes and is not meant to be a robust tool for PDP modeling.

Run the code using **START.m**. This script will pass a model architecture to either **MLP_block.m** or **MLP_trial.m**, which train up a network and return training accuracy information, calculated using Luce's (1959) choice rule. These scripts can also be edited to allow regression -- you would only need to convert the error metric to SSE/Cross Entropy rather than p(correct).

*Nolan Conaway*

*November 16, 2014*
