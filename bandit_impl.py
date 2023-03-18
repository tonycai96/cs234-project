import numpy as np
from matplotlib import pyplot as plt


N_ARMS = 3


# Bound on quad_form(p-p_hat, V) increases as a function of time
def linear_ucb_v2(
    x_train: np.ndarray,
    y_train: np.ndarray,
    sigma: float,
    reg: float,
    delta: float = 0.05,
) -> np.ndarray:
    chosen_arms = []
    D = np.max(np.linalg.norm(x_train, axis=1))
    B = np.max(y_train)
    n_samples, n_features = x_train.shape
    A_arms = [
        reg * np.eye(n_features),
        reg * np.eye(n_features),
        reg * np.eye(n_features),
    ]
    b_arms = [np.zeros(n_features), np.zeros(n_features), np.zeros(n_features)]
    t_arms = [0, 0, 0]
    for t in range(n_samples):
        pred_arms = np.zeros(N_ARMS)
        x_t = x_train[t]
        for a in range(N_ARMS):
            A_inv = np.linalg.inv(A_arms[a])
            theta_a_hat = A_inv @ b_arms[a]
            # beta = (
            #     sigma * np.sqrt(n_features * np.log((1 + t_arms[a] * D**2 / reg) / delta))
            #     + np.sqrt(reg) * B
            # )
            beta = 10
            # theta_a = cp.Variable(n_features)
            # prob = cp.Problem(
            #     cp.Maximize(theta_a.T @ x_train[t]),
            #     [cp.quad_form(theta_a - theta_a_hat, A_arms[a]) <= beta**2],
            # )
            # pred_arms[a] = prob.solve()
            theta_a = theta_a_hat + beta / np.sqrt(x_t @ (A_inv @ x_t)) * A_inv @ x_t
            pred_arms[a] = x_t.T @ theta_a
        a_t = np.argmax(pred_arms)
        A_arms[a_t] += np.outer(x_t, x_t)
        b_arms[a_t] += y_train[t][a_t] * x_t
        t_arms[a_t] += 1
        chosen_arms.append(a_t)
    return chosen_arms


# Bound on quad_form(p-p_hat, V) doesn't increases as a function of time
def linear_ucb(x_train: np.ndarray, y_train: np.ndarray, beta: int = 2) -> np.ndarray:
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
            pred_arms[a] = theta_a @ x_t + beta * np.sqrt(x_t @ (A_inv @ x_t))
        a_t = np.argmax(pred_arms)
        A_arms[a_t] += np.outer(x_t, x_t)
        b_arms[a_t] += y_train[i][a_t] * x_t
        chosen_arms.append(a_t)
    return np.array(chosen_arms)


def linear_ucb_stacked(
    x_train: np.ndarray, y_train: np.ndarray, alpha: int = 2
) -> np.ndarray:
    chosen_arms = []
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
    arms_features = [arm1_features, arm2_features, arm3_features]
    n_features *= 3
    A = np.eye(n_features)
    b = np.zeros(n_features)
    for t in range(n_samples):
        pred_arms = np.zeros(N_ARMS)
        for a in range(N_ARMS):
            x_at = arms_features[a][t]
            beta = 10
            A_inv = np.linalg.inv(A)
            theta_a_hat = A_inv @ b
            theta_a = theta_a_hat + beta / np.sqrt(x_at @ (A_inv @ x_at)) * A_inv @ x_at
            pred_arms[a] = x_at.T @ theta_a
        a_t = np.argmax(pred_arms)
        A += np.outer(arms_features[a_t][t], arms_features[a_t][t])
        b += y_train[t][a_t] * arms_features[a_t][t]
        chosen_arms.append(a_t)
    return chosen_arms


def stack_arm_features(x_train):
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
    arms_features = [arm1_features, arm2_features, arm3_features]
    return arms_features


# alpha = reward at most (1 - alpha) worse than baseline
# beta = confidence interval bound
def safe_linear_ucb(
    x_train: np.ndarray, y_train: np.ndarray, alpha: float, beta: float
) -> np.ndarray:
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

    print("baseline count = ", baseline_action_count)
    print("explore count = ", explore_action_count)
    return chosen_arms


def thompson_sampling(x_train: np.ndarray, y_train: np.ndarray, v: float):
    arms_features = stack_arm_features(x_train)
    n_samples, n_features = arms_features[0].shape
    A = np.eye(n_features)
    mu = np.zeros(n_features)
    f = np.zeros(n_features)
    chosen_arms = []
    for t in range(n_samples):
        theta = np.random.multivariate_normal(mu, v**2*np.linalg.inv(A))
        r_arms = x_train[t] @ theta
        a = np.argmax(r_arms)
        chosen_arms.append(a)
        A += np.outer(arms_features[a][t], arms_features[a][t])
        f += y_train[t] * arms_features[a][t]
        mu = np.linalg.inv(A) @ f
    return chosen_arms


if __name__ == "__main__":
    np.random.seed(42)
    true_arms = 5 * np.random.rand(5, 3)

    x_train = np.random.rand(4000, 5)
    y_train = x_train @ true_arms
    y_train[:, 0] += np.random.normal(0, 0.1, 4000)
    y_train[:, 1] += np.random.normal(0, 0.1, 4000)
    y_train[:, 2] += np.random.normal(0, 0.1, 4000)
    y_best = y_train.max(axis=1)

    # chosen_arms = linear_ucb(x_train, y_train, alpha=10)
    # chosen_arms = linear_ucb_v2(x_train, y_train, sigma=0.1, reg=1, delta=0.01)
    # chosen_arms = linear_ucb_stacked(x_train, y_train, alpha=10)
    chosen_arms = safe_linear_ucb(x_train, y_train, alpha=0.1, beta=10)

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
