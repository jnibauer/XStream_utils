import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter


def _get_contour_levels(H, levels):
    """
    Return bin-value thresholds such that each threshold's contour encloses
    the corresponding fraction of the total mass in H, starting from the
    highest-density bins outward (HPD-style).

    Parameters
    ----------
    H      : 2D ndarray
    levels : sequence of float, e.g. [0.68, 0.95]

    Returns
    -------
    list of float in ascending order (ready for matplotlib's `contour`)
    """
    flat  = H.flatten()
    total = flat.sum()
    if total == 0:
        return []
    sorted_flat = np.sort(flat)[::-1]
    cumsum      = np.cumsum(sorted_flat) / total
    thresholds  = []
    for level in levels:
        idx = np.searchsorted(cumsum, level)
        idx = min(idx, len(sorted_flat) - 1)
        thresholds.append(sorted_flat[idx])
    return sorted(thresholds)


def plot_2d_panel(
    ax,
    x,
    y,
    weights=None,
    n_bins=20,
    show_contours=True,
    contour_levels=(0.68, 0.95),
    smooth_contours=True,
    smooth_sigma=1.0,
    contour_on_mean=True,
    cmap='Purples',
    contour_colors='k',
    true_x=None,
    true_y=None,
    heatmap_on_mean=True,
):
    """
    Draw a single weighted 2-D heatmap + contours on an existing axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    x, y : array-like
        Data columns to plot.
    weights : array-like or None
        Per-sample weights. If None, uniform weights are used.
    n_bins : int
    show_contours : bool
    contour_levels : sequence of float
    smooth_contours : bool
    smooth_sigma : float
    contour_on_mean : bool
    cmap : str
    contour_colors : str or sequence
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if weights is None:
        weights = np.ones(len(x)) / len(x)

    H_sum, xedges, yedges = np.histogram2d(x, y, bins=n_bins, weights=weights)
    H_count, _, _ = np.histogram2d(x, y, bins=n_bins)
    with np.errstate(invalid='ignore'):
        H_mean = np.where(H_count > 0, H_sum / H_count, np.nan)

    H_plot = np.where(H_count > 0, H_mean if heatmap_on_mean else H_sum, np.nan)
    vmin   = np.nanmin(H_plot)
    vmax   = np.nanmax(H_plot)

    ax.pcolormesh(xedges, yedges, H_plot.T, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)

    if show_contours:
        H_base    = np.nan_to_num(H_mean, nan=0.0) if contour_on_mean else H_sum
        H_contour = gaussian_filter(H_base, sigma=smooth_sigma) if smooth_contours else H_base
        lvls      = _get_contour_levels(H_contour, contour_levels)
        if lvls:
            xc = 0.5 * (xedges[:-1] + xedges[1:])
            yc = 0.5 * (yedges[:-1] + yedges[1:])
            ax.contour(
                xc, yc, H_contour.T,
                levels=lvls,
                colors=contour_colors,
                linewidths=[0.8, 1.2],
                linestyles=['--', '-'],
                rasterized=True,
            )

    if true_x is not None:
        ax.axvline(true_x, color='k', lw=1.0, zorder=5)
    if true_y is not None:
        ax.axhline(true_y, color='k', lw=1.0, zorder=5)

    ax.set_xlim(xedges[0], xedges[-1])
    ax.set_ylim(yedges[0], yedges[-1])


def plot_corner(
    points_good,
    prob_values_of_samples,
    true_params,
    prior_keys_latex,
    n_bins=15,
    show_contours=True,
    global_norm=False,
    contour_levels=(0.68, 0.95),
    smooth_contours=True,
    smooth_sigma=1.0,
    contour_on_mean=True,
    hist1d_on_mean=True,
    cmap='Purples',
    contour_colors='k',
    figsize=(22, 22),
    savefig=None,
    fig=None,
    subplot_spec=None,
):
    """
    Custom weighted corner plot — no resampling, no `corner` package.
    We take the parameter samples and their normalised significance weights (from Lambda(sigma) calculation)
    and plot the mean weight per bin in the 2-D panels, which visually corresponds to the heatmap colour. The contours can be drawn
    from either the mean weight per bin (H_mean) or the total weight per bin (H_sum).
    H_mean is more principled since our sampler is not sampling the true posteiror, but the negative KL divergence.
    We assign signicance weights in post, so taking the mean in each cell gives a more accurate visual representation
    of the relative significance across the parameter space. H_sum, on the other hand, is more standard for Bayesian HPD contours,
     but can be misleading if some bins have very few samples (e.g. in the tails) since it doesn't account for sampling density.

    Parameters
    ----------
    points_good : ndarray, shape (n_particles, n_params)
        Parameter samples after filtering NaNs.
    prob_values_of_samples : ndarray, shape (n_particles,)
        Normalised significance weights for each particle.
    true_params : array-like, shape (n_params,)
        Ground-truth parameter values (shown as crosshairs / vertical lines).
    prior_keys_latex : list of str
        Axis labels in LaTeX, one per parameter.
    n_bins : int
        Number of bins per dimension for both 1-D and 2-D histograms.
    show_contours : bool
        Overlay confidence contours on 2-D panels.
    global_norm : bool
        If True, all 2-D panels share a single colorbar normalisation.
        If False, each panel is normalised independently.
    contour_levels : sequence of float
        Confidence levels for contours, e.g. (0.68, 0.95).
        Only used when show_contours=True.
    smooth_contours : bool
        Apply Gaussian smoothing to the histogram before drawing contours.
    smooth_sigma : float
        Smoothing kernel width in bins. Only used when smooth_contours=True.
    contour_on_mean : bool
        If True,  contours are drawn on H_mean (mean weight per bin) —
                  visually consistent with the heatmap colour.
        If False, contours are drawn on H_sum (total probability mass per bin) —
                  standard Bayesian HPD contours.
    hist1d_on_mean : bool
        If True,  1-D histograms show mean weight per bin (H_mean), consistent
                  with the 2-D heatmap and contours — useful for checking
                  alignment between 1-D and 2-D panels. Default is True.
        If False, 1-D histograms show total weight per bin (H_sum).
    cmap : str
        Matplotlib colormap name for the 2-D heatmap panels.
    contour_colors : str or sequence
        Colour(s) for the contour lines.
    figsize : tuple
        Figure size passed to plt.figure.
    savefig : str or None
        If provided, save the figure to this path before showing.
    fig : matplotlib.figure.Figure or None
        Existing figure to draw into. If None, a new figure is created.
        Ignored when subplot_spec is None and fig is None.
    subplot_spec : matplotlib.gridspec.SubplotSpec or None
        A GridSpec slot (e.g. ``gs[0, 1]``) to embed the corner plot into.
        When provided, the corner is drawn inside that slot using
        GridSpecFromSubplotSpec; ``fig`` must also be supplied.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes_dict : dict mapping (row, col) → matplotlib.axes.Axes
        Contains only the lower-triangle and diagonal panels.
    """
    n_params = points_good.shape[1]

    # ── Pre-compute histograms ─────────────────────────────────────────────────

    hist1d_data  = {}
    for i in range(n_params):
        counts, edges = np.histogram(
            points_good[:, i], bins=n_bins, weights=prob_values_of_samples
        )
        if hist1d_on_mean:
            count_n, _ = np.histogram(points_good[:, i], bins=edges)
            with np.errstate(invalid='ignore'):
                counts = np.where(count_n > 0, counts / count_n, 0.0)
        hist1d_data[i] = (counts, edges)

    hist2d_data    = {}
    all_H_mean_vals = []

    for row in range(n_params):
        for col in range(row):
            x = points_good[:, col]
            y = points_good[:, row]

            H_sum, xedges, yedges = np.histogram2d(
                x, y, bins=n_bins, weights=prob_values_of_samples
            )
            H_count, _, _ = np.histogram2d(x, y, bins=n_bins)
            with np.errstate(invalid='ignore'):
                H_mean = np.where(H_count > 0, H_sum / H_count, np.nan)

            hist2d_data[(row, col)] = (H_sum, H_mean, xedges, yedges)
            valid = H_mean[~np.isnan(H_mean)]
            if len(valid):
                all_H_mean_vals.extend(valid.tolist())

    global_vmin = np.min(all_H_mean_vals)
    global_vmax = np.max(all_H_mean_vals)

    # ── Build figure ───────────────────────────────────────────────────────────

    if fig is None:
        fig = plt.figure(figsize=figsize)

    if subplot_spec is not None:
        gs = gridspec.GridSpecFromSubplotSpec(
            n_params, n_params, subplot_spec=subplot_spec, hspace=0.05, wspace=0.05
        )
    else:
        gs = gridspec.GridSpec(n_params, n_params, figure=fig, hspace=0.05, wspace=0.05)

    last_im   = None
    axes_dict = {}

    for row in range(n_params):
        for col in range(n_params):

            if col > row:
                continue

            ax = fig.add_subplot(gs[row, col])
            axes_dict[(row, col)] = ax

            # ── Diagonal: weighted 1-D histogram ──────────────────────────────
            if row == col:
                counts, edges = hist1d_data[row]
                centers = 0.5 * (edges[:-1] + edges[1:])
                ax.bar(
                    centers, counts, width=np.diff(edges),
                    color='mediumpurple', alpha=0.85, linewidth=0, rasterized=True
                )
                ax.axvline(true_params[row], color='k', lw=1.5, zorder=5)
                ax.set_yticks([])
                ax.set_xlim(edges[0], edges[-1])

            # ── Off-diagonal: 2-D weighted heatmap ────────────────────────────
            else:
                H_sum, H_mean, xedges, yedges = hist2d_data[(row, col)]

                vmin = global_vmin if global_norm else np.nanmin(H_mean)
                vmax = global_vmax if global_norm else np.nanmax(H_mean)

                im = ax.pcolormesh(
                    xedges, yedges, H_mean.T,
                    cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True
                )
                last_im = im

                if show_contours:
                    H_base    = np.nan_to_num(H_mean, nan=0.0) if contour_on_mean else H_sum
                    H_contour = gaussian_filter(H_base, sigma=smooth_sigma) if smooth_contours else H_base
                    lvls      = _get_contour_levels(H_contour, contour_levels)
                    if lvls:
                        xc = 0.5 * (xedges[:-1] + xedges[1:])
                        yc = 0.5 * (yedges[:-1] + yedges[1:])
                        ax.contour(
                            xc, yc, H_contour.T,
                            levels=lvls,
                            colors=contour_colors,
                            linewidths=[0.8, 1.2],
                            linestyles=['--', '-'],
                            rasterized=True,
                        )

                ax.axvline(true_params[col], color='k', lw=1.0, zorder=5)
                ax.axhline(true_params[row], color='k', lw=1.0, zorder=5)
                ax.set_xlim(xedges[0], xedges[-1])
                ax.set_ylim(yedges[0], yedges[-1])

            # ── Axis labels and tick visibility ───────────────────────────────
            is_bottom_row = (row == n_params - 1)
            is_left_col   = (col == 0)

            if is_bottom_row:
                ax.set_xlabel(prior_keys_latex[col], fontsize=16)
                ax.tick_params(axis='x', which='major', labelsize=9, rotation=45)
            else:
                ax.tick_params(axis='x', labelbottom=False)

            if is_left_col and row > 0:
                ax.set_ylabel(prior_keys_latex[row], fontsize=16)
                ax.tick_params(axis='y', which='major', labelsize=9)
            elif row == col:
                pass
            else:
                ax.tick_params(axis='y', labelleft=False)

    # ── Global colorbar ────────────────────────────────────────────────────────
    if last_im is not None and global_norm:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.018, 0.70])
        norm    = mcolors.Normalize(vmin=global_vmin, vmax=global_vmax)
        sm      = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cb      = fig.colorbar(sm, cax=cbar_ax)
        cb.set_label('Mean weight per bin', fontsize=14)
        cb.ax.tick_params(labelsize=11)

    fig.align_xlabels([ax for (row, col), ax in axes_dict.items() if row == n_params - 1])
    fig.align_ylabels([ax for (row, col), ax in axes_dict.items() if col == 0 and row > 0])

    if savefig is not None:
        fig.savefig(savefig, bbox_inches='tight', dpi=150)

    return fig, axes_dict
