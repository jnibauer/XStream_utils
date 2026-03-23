# XStream_utils

Utilities for extragalactic stream analysis: weighted corner plots and H-mean statistics.

## Installation

```bash
pip install git+https://github.com/jnibauer/XStream_utils.git
```

## Contents

- **`plotcontours`** : custom weighted corner plots 
- **`stats`** : H-mean-based quantiles and statistics, consistent with the corner plot's heatmap
- What is H-mean? Rather than taking a histogram (counts), we take the mean of `values` in bins (set by user). Essentially, `scipy.stats.binned_statistic`.
- Why do we need this? We sample a tempered likelihood (avg log-likelihood). The density of points $\neq$ density of posterior. By dividing values by the local density ($\propto N$), we build up a more accurate picture of the target distribution. 

---

## Example: 1D quantiles from weighted samples

`h_mean_quantile` computes quantiles by binning values, taking the mean weight per bin (decoupled from sampling density), and integrating the resulting CDF. This is consistent with the H-mean heatmap in `plot_corner`.

```python
import numpy as np
from XStream_utils import h_mean_quantile, h_mean_stats

# Suppose you have 1D density samples and their significance weights
# values: shape (n_particles,) or (n_particles, n_radii)
# weights: shape (n_particles,) — normalised significance weights

# --- 1D array ---
quantiles = h_mean_quantile(values, weights, quantiles=[0.16, 0.5, 0.84], n_bins=50)
# quantiles[0] = 16th percentile, quantiles[1] = median, quantiles[2] = 84th percentile

# For density-like quantities spanning orders of magnitude, use log-space binning:
quantiles = h_mean_quantile(values, weights, quantiles=[0.16, 0.5, 0.84], n_bins=50, log_space=True)

# --- 2D array (e.g. density profile samples over a radial grid) ---
# values: shape (n_particles, n_radii)
quantiles = h_mean_quantile(values, weights, quantiles=[0.16, 0.84], n_bins=50, log_space=True)
# quantiles: shape (2, n_radii)
lower, upper = quantiles[0], quantiles[1]

# --- Mean and std ---
# Returns mean and std of the H-mean distribution along axis=0
# With log_space=True, returns mean and std of log(values)
means, stds = h_mean_stats(values, weights, n_bins=50, log_space=True)
```

---

## Example: weighted corner plot

`plot_corner` takes parameter samples and their significance weights and renders a lower-triangle corner plot. The heatmap colour in each 2D panel shows the **mean weight per bin** (H-mean), which correctly reflects significance independent of sampling density.

```python
import numpy as np
import matplotlib.pyplot as plt
from XStream_utils import plot_corner

# Load your sampler output
dict_load   = np.load('my_KL_dict.npy', allow_pickle=True).item()
points      = dict_load['points']   # shape (n_particles, n_params)
log_w       = dict_load['log_w']
log_l       = dict_load['log_l']

# --- Compute significance weights ---
# (Lambda-based weighting from the KL divergence sampler)
def Lambda_func(Delta_sigma, N):
    sigma = Delta_sigma + 1
    return N * (-np.log(sigma) - 0.5 / sigma**2 + 0.5)

N = 289  # number of stream track points
Q_model  = -log_l
Q_best   = Q_model[np.argmax(log_l)]
Lambda_measured = N * (Q_best - Q_model)

Delta_sigma_measured = ...  # invert Lambda_func to get Delta_sigma per particle
prob_values = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * Delta_sigma_measured**2)

# Filter out NaN particles
mask        = np.isfinite(points).all(axis=1) & np.isfinite(prob_values)
points_good = points[mask]
prob_values_of_samples = prob_values[mask]
prob_values_of_samples = prob_values_of_samples / prob_values_of_samples.sum()

# --- True parameter values (shown as crosshairs) ---
true_params = np.array([y_prog, speed, vel_theta, vel_phi,
                         log10_M_prog, log10_m_halo, r_s, gamma, beta, t_age])

prior_keys_latex = [
    r'$y_{\rm prog}$ [kpc]',
    r'$v$ [kpc/Myr]',
    r'$\theta$',
    r'$\phi$',
    r'$\log_{10}(M_{\rm prog})$',
    r'$\log_{10}(M_{\rm halo})$',
    r'$r_s$ [kpc]',
    r'$\gamma$',
    r'$\beta$',
    r'$t_{\rm age}$ [Myr]',
]

# --- Plot ---
fig, axes_dict = plot_corner(
    points_good,
    prob_values_of_samples,
    true_params,
    prior_keys_latex,
    n_bins          = 24,
    show_contours   = True,
    contour_levels  = (0.68, 0.95),
    smooth_contours = True,
    smooth_sigma    = 1.0,
    contour_on_mean = True,   # contours drawn on H_mean, consistent with heatmap
    hist1d_on_mean  = True,   # 1D histograms also show H_mean
    global_norm     = False,  # each 2D panel normalised independently
    cmap            = 'Purples',
    contour_colors  = 'k',
    figsize         = (22, 22),
    savefig         = 'corner.pdf',  # omit to not save
)

plt.show()
```

### Overplotting points

`plot_corner` returns `axes_dict`, a dict mapping `(row, col) → axes`, covering the lower triangle and diagonal. Use it to overplot reference points:

```python
scatter_pts  = [pt1, pt2, pt3]
scatter_cols = ['r', 'blue', 'g']

for pt, color in zip(scatter_pts, scatter_cols):
    for (row, col), ax in axes_dict.items():
        if row == col:
            continue
        ax.scatter([pt[col]], [pt[row]], color=color, s=10, zorder=10)

plt.show()
```

### Embedding in a larger figure

Pass a `subplot_spec` to nest the corner plot inside an existing `GridSpec`:

```python
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(30, 22))
gs  = gridspec.GridSpec(1, 2, figure=fig)

fig, axes_dict = plot_corner(
    points_good, prob_values_of_samples, true_params, prior_keys_latex,
    fig=fig, subplot_spec=gs[0, 0],
)
# add other panels in gs[0, 1], etc.
```

---

## Main functions

### `h_mean_quantile(values, weights, quantiles, n_bins=50, log_space=False)`
Returns quantiles of the H-mean distribution. For 1D input returns an array of shape `(len(quantiles),)`; for 2D input returns `(len(quantiles), n_cols)`.

### `h_mean_stats(values, weights, n_bins=50, log_space=False)`
Returns `(means, stds)` of the H-mean distribution along axis 0. Shape `(n_cols,)` each.

### `plot_corner(points_good, prob_values_of_samples, true_params, prior_keys_latex, **kwargs)`
Returns `(fig, axes_dict)`. See docstring for full keyword argument list.

### `plot_2d_panel(ax, x, y, weights=None, **kwargs)`
Draw a single weighted 2D heatmap + contours on an existing axes object.
