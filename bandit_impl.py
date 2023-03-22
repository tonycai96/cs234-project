import numpy as np
from matplotlib import pyplot as plt

N_ARMS = 3


def stack_arm_features(x_train) -> list[np.ndarray]:
    n_samples, n_features = x_train.shape
    arm1_features = np.hstack(
        [x_train, np.zeros((n_samples, n_features)), np.zeros((n_samples, n_features))]
    )
    arm2_features = np.hstack(
        [np.zeros((n_samples, n_features)), x_train, np.zeros((n_samples, n_features))]
    )
    arm3_features = np.hstack(
        [np.zeros((n_samples, n_features)), np.zeros((n_samples, n_features)), x_train]
    )
    return [arm1_features, arm2_features, arm3_features]


# beta = 1 + sqrt(ln(2/delta)/2)
def linear_ucb(x_train: np.ndarray, y_train: np.ndarray, beta: float) -> np.ndarray:
    arms_features = stack_arm_features(x_train)
    n_samples, n_features = arms_features[0].shape
    A = np.eye(n_features)
    b = np.zeros(n_features)
    chosen_arms = []
    for t in range(n_samples):
        pred_arms = np.zeros(N_ARMS)
        for a in range(N_ARMS):
            x_at = arms_features[a][t]
            A_inv = np.linalg.inv(A)
            theta_a_hat = A_inv @ b
            theta_a = theta_a_hat + beta / np.sqrt(x_at @ (A_inv @ x_at)) * A_inv @ x_at
            pred_arms[a] = x_at.T @ theta_a
        a_t = np.argmax(pred_arms)
        A += np.outer(arms_features[a_t][t], arms_features[a_t][t])
        b += y_train[t][a_t] * arms_features[a_t][t]
        chosen_arms.append(a_t)
    return chosen_arms


def oful(
    x_train: np.ndarray, y_train: np.ndarray, delta: float, var: float, S: float
) -> np.ndarray:
    arms_features = stack_arm_features(x_train)
    n_samples, n_features = arms_features[0].shape
    A = np.eye(n_features)
    b = np.zeros(n_features)
    chosen_arms = []
    L = np.max(np.linalg.norm(x_train, axis=1))
    # S = 1 # np.max(y_train)
    for t in range(n_samples):
        pred_arms = np.zeros(N_ARMS)
        for a in range(N_ARMS):
            x_at = arms_features[a][t]
            A_inv = np.linalg.inv(A)
            theta_a_hat = A_inv @ b
            beta = var * np.sqrt(n_features * np.log((1 + t * L**2) / delta)) + S
            theta_a = theta_a_hat + beta / np.sqrt(x_at @ (A_inv @ x_at)) * A_inv @ x_at
            pred_arms[a] = x_at.T @ theta_a
        a_t = np.argmax(pred_arms)
        A += np.outer(arms_features[a_t][t], arms_features[a_t][t])
        b += y_train[t][a_t] * arms_features[a_t][t]
        chosen_arms.append(a_t)
    return chosen_arms


# alpha = reward at most (1 - alpha) worse than baseline
# beta = confidence interval bound
def safe_linear_ucb(
    x_train: np.ndarray, y_train: np.ndarray, alpha: float, beta: float
) -> np.ndarray[int]:
    arms_features = stack_arm_features(x_train)
    n_samples, n_features = arms_features[0].shape
    A = np.eye(n_features)
    b = np.zeros(n_features)
    z = np.zeros(n_features)  # cumulative features of non-optimal actions
    r_b_taken = 0
    r_b_total = 0

    # Debugging
    baseline_action_count = 0
    explore_action_count = 0

    chosen_arms = []
    for t in range(n_samples):
        pred_arms = np.zeros(N_ARMS)
        A_inv = np.linalg.inv(A)
        theta_hat = A_inv @ b
        for a in range(N_ARMS):
            x_at = arms_features[a][t]
            theta_a = theta_hat + beta / np.sqrt(x_at @ (A_inv @ x_at)) * A_inv @ x_at
            pred_arms[a] = x_at.T @ theta_a
        a_t = np.argmax(pred_arms)

        r_b_total += y_train[t][1]
        z_t = z + arms_features[a_t][t]
        # lower_bound = estimated worst-case reward from taking non-baseline action + observed reward from taking baseline action
        if z.sum() < 1e-6:
            lower_bound = r_b_taken
        else:
            theta = theta_hat - beta / np.sqrt(z @ (A_inv @ z)) * A_inv @ z
            lower_bound = theta @ z_t + r_b_taken
        # Check if it's safe to take a_t
        if lower_bound + r_b_taken >= (1 - alpha) * r_b_total:
            A += np.outer(arms_features[a_t][t], arms_features[a_t][t])
            b += y_train[t][a_t] * arms_features[a_t][t]
            z = z_t
            chosen_arms.append(a_t)
            explore_action_count += 1
        else:
            r_b_taken += y_train[t][1]
            chosen_arms.append(1)  # baseline action
            baseline_action_count += 1

    return np.array(chosen_arms)


def gaussian_thompson_sampling(
    x_train: np.ndarray, y_train: np.ndarray, v: float
) -> np.ndarray[int]:
    x_train = x_train.astype(np.float64)
    arms_features = stack_arm_features(x_train)
    n_samples, n_features = arms_features[0].shape
    A = np.eye(n_features)
    mu = np.zeros(n_features)
    f = np.zeros(n_features)
    chosen_arms = []
    for t in range(n_samples):
        theta = np.random.multivariate_normal(mu, v**2 * np.linalg.inv(A))
        pred_arms = np.zeros(N_ARMS)
        for a in range(N_ARMS):
            pred_arms[a] = np.dot(arms_features[a][t], theta)
        a = np.argmax(pred_arms)
        chosen_arms.append(a)
        A += np.outer(arms_features[a][t], arms_features[a][t])
        f += y_train[t][a] * arms_features[a][t]
        mu = np.linalg.inv(A) @ f
    return np.array(chosen_arms)


if __name__ == "__main__":
    np.random.seed(42)
    true_arms = np.random.rand(5, 3)

    N_SAMPLES = 4000
    x_train = np.random.rand(N_SAMPLES, 5)
    y_train = x_train @ true_arms
    y_train[:, 0] += np.random.normal(0, 0.1, N_SAMPLES)
    y_train[:, 1] += np.random.normal(0, 0.1, N_SAMPLES)
    y_train[:, 2] += np.random.normal(0, 0.1, N_SAMPLES)
    y_best = y_train.max(axis=1)

    # chosen_arms = linear_ucb(x_train, y_train, beta=0.5)
    # chosen_arms = oful(x_train, y_train, delta=0.05, var=0.1, S=1)
    # chosen_arms = linear_ucb_v2(x_train, y_train, sigma=0.1, reg=1, delta=0.01)
    # chosen_arms = safe_linear_ucb(x_train, y_train, alpha=0.1, beta=10)
    chosen_arms = gaussian_thompson_sampling(x_train, y_train, v=1)

    # Update stats
    num_samples = x_train.shape[0]
    regrets = [0]
    for i in range(num_samples):
        regrets.append(
            regrets[-1] + (y_best[i] - x_train[i] @ true_arms[:, chosen_arms[i]])
        )

    timesteps = np.linspace(0, len(regrets) - 1, len(regrets))
    plt.plot(timesteps, regrets)
    plt.show()
