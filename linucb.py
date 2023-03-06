import numpy as np
import matplotlib.pyplot as plt


def linear_ucb(x_train, y_train, alpha=2):
    N_ARMS = 3

    chosen_arms = []
    num_samples, n_features = x_train.shape
    A_arms = [np.eye(n_features), np.eye(n_features), np.eye(n_features)]
    b_arms = [np.zeros(n_features), np.zeros(n_features), np.zeros(n_features)]
    for i in range(num_samples):
        pred_arms = np.zeros(N_ARMS)
        x_t = x_train[i]
        for a in range(N_ARMS):
            A_inv = np.linalg.inv(A_arms[a])
            theta_a = A_inv @ b_arms[a]
            pred_arms[a] = theta_a @ x_t + alpha * np.sqrt(x_t@(A_inv@x_t))
        a_t = np.argmax(pred_arms)
        A_arms[a_t] += np.outer(x_t, x_t)
        b_arms[a_t] += y_train[i][a_t] * x_t
        chosen_arms.append(a_t)
    return chosen_arms


if __name__ == "__main__":
    np.random.seed(42)
    true_arms = 5*np.random.rand(5, 3)

    x_train = np.random.rand(2000, 5)
    y_train = x_train @ true_arms
    y_best = y_train.max(axis=1)

    chosen_arms = linear_ucb(x_train, y_train, alpha=10)

    # Update stats
    num_samples = x_train.shape[0]
    regrets = [0]
    for i in range(num_samples):
        regrets.append(regrets[-1] + (y_best[i] - x_train[i]@true_arms[:, chosen_arms[i]]))

    timesteps = np.linspace(0, len(regrets)-1, len(regrets))
    plt.plot(timesteps, regrets)
    plt.show()