import cvxpy as cp
import numpy as np

# Problem data.
np.random.seed(1)

n_features = 10
x_t = np.random.rand(n_features)
V = np.random.rand(n_features, n_features)
V = np.dot(V, V.T)

# Construct the problem.
theta = cp.Variable(n_features)
theta_hat = np.random.rand(n_features)
prob = cp.Problem(
    cp.Minimize(theta.T @ x_t), [cp.quad_form(theta - theta_hat, V) <= 0.25]
)

# The optimal objective value is returned by `prob.solve()`.
print(prob.solve())

theta2 = (
    theta_hat
    - np.sqrt(0.25) / np.sqrt(x_t @ (np.linalg.inv(V) @ x_t)) * np.linalg.inv(V) @ x_t
)
print(x_t.T @ theta2)
