%% Clear and Close Figures
clear all; close all;

fprintf('Loading data ...\n');

%% Load Data
data = load('houses.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Normalize the features
fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);

% Add constant term to X
X = [ones(m, 1) X];

fprintf('Running gradient descent ...\n');

% Choose some values for learning rate and max number of iterations
eta = 0.01;
num_iters = 1000;

% Init Theta and Run Gradient Descent 
thetaGD = zeros(3, 1);
[thetaGD, J_history] = gradientDescent(X, y, thetaGD, eta, num_iters);

% Compute theta with normal equation
thetaNE = pinv(X)*y;

% Display results
fprintf('Theta computed with gradient descent: \n');
fprintf(' %f \n', thetaGD);

fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', thetaNE);


% Plot Gradient Descent convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');


% Estimate the price of new houses

xnew1 = [ 1650 3 ];
xnew2 = [ 2100 4 ];
xnew3 = [ 2650 5 ];
xnorm1 = [ 1 (xnew1(1)-mu(1))/sigma(1) (xnew1(2)-mu(2))/sigma(2) ];
xnorm2 = [ 1 (xnew2(1)-mu(1))/sigma(1) (xnew2(2)-mu(2))/sigma(2) ];
xnorm3 = [ 1 (xnew3(1)-mu(1))/sigma(1) (xnew3(2)-mu(2))/sigma(2) ];
price1GD = xnorm1 * thetaGD;
price1NE = xnorm1 * thetaNE;
price2GD = xnorm2 * thetaGD;
price2NE = xnorm2 * thetaNE;
price3GD = xnorm3 * thetaGD;
price3NE = xnorm3 * thetaNE;

% ============================================================

fprintf(['Predicted prices of new houses:\n']);
fprintf(['%d %d : [GD] $%.2f  [NE] $%.2f \n'], xnew1, price1GD, price1NE);
fprintf(['%d %d : [GD] $%.2f  [NE] $%.2f \n'], xnew2, price2GD, price2NE);
fprintf(['%d %d : [GD] $%.2f  [NE] $%.2f \n'], xnew3, price3GD, price3NE);

