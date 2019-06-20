function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


% Calculation of h_theta
h_theta = sigmoid(X*theta);

% Gradient calculation
gradSum = h_theta - y;
grad0 = (X.'*gradSum)/m;

theta_temp = theta; 
theta_temp(1) = 0;

grad = grad0 + (lambda/m).*theta_temp;

% Cost calculation
cost_temp = (-y.*log(h_theta) - (1 - y).*log(1 - h_theta));
J_temp = sum(cost_temp)/m;
reg_param = (lambda/(2*m))*(theta(2)^2 + theta(3)^2);
J = J_temp + reg_param;


% =============================================================

end
