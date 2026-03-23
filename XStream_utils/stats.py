import numpy as np


def h_mean_quantile_1d(values, weights, quantiles, n_bins=50, log_space=False):
    """
    H_mean-based quantile for a 1-D array.
    Bins the values, computes mean weight per bin (decoupled from sampling
    density), builds the CDF from those mean weights, then interpolates.
    Consistent with plot_corner's H_mean 2-D heatmap and hist1d_on_mean=True.
    """
    if log_space:
        v = np.log(np.clip(values, 1e-300, None))
    else:
        v = np.asarray(values, dtype=float)
    H_sum,   edges = np.histogram(v, bins=n_bins, weights=weights)
    H_count, _     = np.histogram(v, bins=n_bins)
    with np.errstate(invalid='ignore'):
        H_mean_bins = np.where(H_count > 0, H_sum / H_count, 0.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    cdf = np.cumsum(H_mean_bins)
    if cdf[-1] == 0:
        return np.full(len(quantiles), np.nan)
    cdf /= cdf[-1]
    result = np.interp(quantiles, cdf, centers)
    return np.exp(result) if log_space else result


def h_mean_quantile(values, weights, quantiles, n_bins=50, log_space=False):
    """H_mean-based quantile along axis=0 of a 2-D array (or 1-D).

    Parameters
    ----------
    log_space : bool
        If True, bin in log space (useful when values span orders of magnitude).
        Default is False.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        return h_mean_quantile_1d(values, weights, quantiles, n_bins, log_space)
    out = np.zeros((len(quantiles), values.shape[1]))
    for j in range(values.shape[1]):
        out[:, j] = h_mean_quantile_1d(values[:, j], weights, quantiles, n_bins, log_space)
    return out


def h_mean_stats(values, weights, n_bins=50, log_space=False):
    """
    H_mean-based mean and std along axis=0 of a 2-D array.

    Parameters
    ----------
    log_space : bool
        If True, bin and compute statistics in log space — returns mean and std
        of log(values). Default is False.
    """
    values = np.asarray(values, dtype=float)
    means = np.zeros(values.shape[1])
    stds  = np.zeros(values.shape[1])
    for j in range(values.shape[1]):
        v = np.log(np.clip(values[:, j], 1e-300, None)) if log_space else values[:, j]
        H_sum,   edges = np.histogram(v, bins=n_bins, weights=weights)
        H_count, _     = np.histogram(v, bins=n_bins)
        with np.errstate(invalid='ignore'):
            H_mean_bins = np.where(H_count > 0, H_sum / H_count, 0.0)
        centers = 0.5 * (edges[:-1] + edges[1:])
        total = H_mean_bins.sum()
        if total == 0:
            means[j] = np.nan; stds[j] = np.nan; continue
        mu       = np.sum(H_mean_bins * centers) / total
        sigma    = np.sqrt(np.sum(H_mean_bins * (centers - mu)**2) / total)
        means[j] = mu
        stds[j]  = sigma
    return means, stds
