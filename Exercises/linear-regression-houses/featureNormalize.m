% Normalizes the features in X 
function [X_norm, mu, sigma] = featureNormalize(X)

X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

n = size(X,2);
for k=1:n,
	mu(k) = mean(X(:,k));
	sigma(k) = std(X(:,k));
	X_norm(:,k) = (X(:,k) - mu(k))/sigma(k);
end;


% ============================================================

end
