function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% Set of C and sigma to try out
C = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

% dummy values for the kernel function
x1 = [1,2,1];
x2 = [0,4,-1];

% Initializations
minError = 0;
optC = 0;
optSigma = 0;

% Loop over all the C and sigma combinations
for i = 1:length(C)
    for j = 1:length(sigma)
        
        % Train the SVM
        model= svmTrain(X, y, C(i), @(x1, x2) gaussianKernel(x1, x2, sigma(j)));
        
        % Prediction error computation
        predictions = svmPredict(model, Xval);        
        predError = mean(double(predictions ~= yval));
        
        % Finding the minimum Error given by C and sigma
        if i*j == 1
            minError = predError;
        end
        
        if (i*j > 1)
            if predError <= minError
            minError = predError;
            optC = C(i);
            optSigma = sigma(j);
            end
        end
        
    end
end

% Returning the optimum C and sigma values
C = optC;
sigma = optSigma;

% =========================================================================

end
