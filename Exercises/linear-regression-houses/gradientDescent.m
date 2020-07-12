function [theta, J_history] = gradientDescent(X, y, theta, eta, num_iters)

m = length(y); % number of training examples

J_history = zeros(num_iters, 1); % just for display

for iter = 1:num_iters

	n = size(X,2);
	delta = zeros(n, 1);
	for k=1:n,
		delta(k) = 1/m * sum ( (X * theta - y) .* X(:,k) );
	end;

	theta = theta - eta*delta; 

    % Save the cost J in every iteration (just for display)
    J_history(iter) = 1/(2*m) * ( (X*theta-y)'*(X*theta-y) );

end

end
