function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

m = length(y);
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    cost = X * theta - y;
    n = length(theta);
    r = zeros(n,1);
    for i = 1:n
        r(i) = sum(cost .* X(:, i));
    end
    theta = theta - alpha/m * r;

    J_history(iter) = computeCost(X, y, theta);

end
end
