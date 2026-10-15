import numpy as np

''' Code produced with help of ChatGPT '''

def bootstrap_mean_std(X, loglX, n_bootstrap=1000, confidence=0.95):
    """
    Perform bootstrap resampling to estimate confidence intervals for the mean and std deviation.

    Parameters:
        X (np.ndarray): Array of sample locations.
        loglX (np.ndarray): Array of log-likelihoods corresponding to the samples.
        n_bootstrap (int): Number of bootstrap resampling iterations.
        confidence (float): Confidence level for the intervals (e.g., 0.95 for 95% CI).

    Returns:
        mean (float): Weighted mean of the original data.
        std (float): Weighted standard deviation of the original data.
        ci_mean (tuple): Confidence interval for the mean (low, high).
        ci_std (tuple): Confidence interval for the std deviation (low, high).
    """
    # Ensure inputs are numpy arrays
    X = np.asarray(X)
    loglX = np.asarray(loglX)

    # Convert log-likelihoods to weights
    weights = np.exp(loglX - np.max(loglX))  # Subtract max to prevent overflow
    weights /= np.sum(weights)  # Normalize weights

    # Compute weighted mean and std of original data
    mean = np.sum(weights * X)
    variance = np.sum(weights * (X - mean)**2)
    std = np.sqrt(variance)

    # Bootstrap resampling
    bootstrap_means = []
    bootstrap_stds = []

    for _ in range(n_bootstrap):
        # Resample indices with replacement according to weights
        resample_indices = np.random.choice(len(X), size=len(X), p=weights)
        resampled_X = X[resample_indices]
        resampled_weights = weights[resample_indices]

        # Normalize resampled weights
        resampled_weights /= np.sum(resampled_weights)

        # Compute weighted mean and std for resampled data
        resampled_mean = np.sum(resampled_weights * resampled_X)
        resampled_variance = np.sum(resampled_weights * (resampled_X - resampled_mean)**2)
        resampled_std = np.sqrt(resampled_variance)

        bootstrap_means.append(resampled_mean)
        bootstrap_stds.append(resampled_std)

    # Convert results to numpy arrays
    bootstrap_means = np.array(bootstrap_means)
    bootstrap_stds = np.array(bootstrap_stds)

    # Calculate confidence intervals
    alpha = 1 - confidence
    ci_mean = (np.percentile(bootstrap_means, 100 * alpha / 2), np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))
    ci_std = (np.percentile(bootstrap_stds, 100 * alpha / 2), np.percentile(bootstrap_stds, 100 * (1 - alpha / 2)))

    return mean, std, ci_mean, ci_std