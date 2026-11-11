"""
Shared binning, input loading and plotting functions for the 3D EEC closure-test notebooks.

Used by:
  - plots_closure_tests_part1_refolding_statistical.ipynb
  - plots_closure_tests_part2_shape_prior.ipynb

Typical notebook setup:
    import sys
    sys.path.insert(0, '/global/cfs/cdirs/alice/blianggi/mypyjetty/pyjetty/pyjetty/'
                       'alice_analysis/analysis/user/jse/3Dunfolding/')
    import closure_tests_common as ctc
    from closure_tests_common import *
    inp = ctc.load_inputs()

Ratio-panel style: the +/-5% band is drawn as two dashed grey lines at 0.95 / 1.05
plus a grey box marking R_L >= RL_BOX_MIN.

Differential pT panels whose lower edge is >= PT_HIGH_MIN (100 GeV) are taken from
the RL-rebinned (24 -> 12 bins) files:
  - unfolded_tests_AA_AB_BB_rad_rebinRL2.root  (closure tests, all sections)
  - unfolded_rebinRL2.root                     (Htruth_{lab}_rebin2 references)
while the panels below 100 GeV (and the combined 20-200 GeV EEC panel) keep the
original 24-bin R_L binning from unfolded_tests.root / response_internals.root.
"""

import os
import sys

import numpy as np
import uproot
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, NullFormatter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jupyter_utils import (xslice_th3d, get_ratio_err, safe_divide,
                           get_lin_bincenters, get_log_bincenters, get_binwidths,
                           projectionX, projectionY, projectionZ)

BASE = "/global/cfs/cdirs/alice/blianggi/mypyjetty/storage/jse/rootfiles/unfolding/"

# ---------------------------------------------------------------------------
# Binnings
# ---------------------------------------------------------------------------
JETPT_BINS = np.array([10, 20, 40, 60, 80, 100, 120, 150, 200, 500], dtype=np.float64)

RL_NBINS = 24
RL_MIN, RL_MAX = 0.01, 1.0
RL_BINS = np.logspace(np.log10(RL_MIN), np.log10(RL_MAX), RL_NBINS + 1)

W_NBINS = 20
W_MIN, W_MAX = 0.0, 0.3
W_BINS = np.logspace(-5, -0.5, W_NBINS + 1)

binning    = [JETPT_BINS, RL_BINS, W_BINS]
bincenters = [get_lin_bincenters(binning[0]), get_log_bincenters(binning[1]), get_log_bincenters(binning[2])]
binwidths  = [get_binwidths(a) for a in binning]

# RL-rebinned counterparts (pairs of adjacent RL bins merged, 24 -> 12 bins)
RL_BINS_REBIN2    = RL_BINS[::2]
binning_rebin2    = [JETPT_BINS, RL_BINS_REBIN2, W_BINS]
bincenters_rebin2 = [bincenters[0], get_log_bincenters(RL_BINS_REBIN2), bincenters[2]]
binwidths_rebin2  = [get_binwidths(a) for a in binning_rebin2]

N_ITER = 5
LABELS = ["AA", "AB", "BB", "rad"]
VARIATIONS = ["pt015_w01", "ptm015_wm01"]
VAR_TEX = {"pt015_w01": r"$p_T^{+0.15}\,w^{+0.1}$", "ptm015_wm01": r"$p_T^{-0.15}\,w^{-0.1}$"}

# differential pT panels with lower edge >= PT_HIGH_MIN come from the rebinned files
PT_HIGH_MIN = 100.0

# ratio-panel styling: grey box above this R_L value
RL_BOX_MIN = 0.4

# Lauren's colors
L_colors = ['red', 'darkorange', 'limegreen', 'green', 'deepskyblue', 'blue', 'black']
L_linestyles = ['solid', 'dashed']
L_markers = ['s', '^', 'v', 'd', 'P']

proj = lambda a: a.values()
proj3D = lambda a: [projectionX(a.values()), projectionY(a.values()), projectionZ(a.values())]
proj3D_err = lambda a: [projectionX(a.errors()), projectionY(a.errors()), projectionZ(a.errors())]


class _AxisShim:
    """Minimal axis stand-in exposing .edges() (for RebinnedHist)."""
    def __init__(self, edges):
        self._edges = np.asarray(edges, dtype=np.float64)
    def edges(self):
        return self._edges


class RebinnedHist:
    """Minimal uproot-hist stand-in wrapping numpy arrays (values/errors/axis edges),
    used for RL-merged histograms built on the fly."""

    def __init__(self, values, errors, edges):
        self._values = np.asarray(values, dtype=np.float64)
        self._errors = np.asarray(errors, dtype=np.float64)
        self._edges = [np.asarray(a, dtype=np.float64) for a in edges]

    def values(self):
        return self._values

    def errors(self):
        return self._errors

    def axis(self, i):
        return _AxisShim(self._edges[i])


def merge_rl_pairs(h):
    """Merge adjacent R_L (axis-1) bin pairs of an uproot TH3: contents summed,
    errors propagated in quadrature (same convention as rebinY_TH3 in
    unfolding_tests.C). Returns a RebinnedHist with 12 R_L bins."""
    v = np.asarray(h.values(), dtype=np.float64)
    e = np.asarray(h.errors(), dtype=np.float64)
    npt, nrl, nw = v.shape
    v2 = v.reshape(npt, nrl // 2, 2, nw).sum(axis=2)
    e2 = np.sqrt((e.reshape(npt, nrl // 2, 2, nw) ** 2).sum(axis=2))
    edges = [h.axis(0).edges(), RL_BINS_REBIN2, h.axis(2).edges()]
    return RebinnedHist(v2, e2, edges)


# text drawn on the plots naming the closure test
CLOSURE_TAG_TEX = {
    "refolding": "Refolding test",
    "statistical": "Statistical closure",
    "shape": "Shape closure",
    "prior": "Prior closure",
}


def _njets_in_ptbins(h1, pt_edges):
    """Sum a 1D jet-pT histogram into the given pT bins (handles finer input binning).
    Also accepts a TH3 (values() is then the full 3D array; the pT mask selects whole slices).
    Returns (N_jets, N_jets_err) with one entry per pT bin."""
    vals = np.asarray(h1.values(), dtype=np.float64)
    errs = np.asarray(h1.errors(), dtype=np.float64)
    h_edges = np.asarray(h1.axis(0).edges(), dtype=np.float64)   # axis(0): needed for TH3, works for TH1
    centers = 0.5 * (h_edges[:-1] + h_edges[1:])

    N = np.zeros(len(pt_edges) - 1)
    N_err = np.zeros(len(pt_edges) - 1)
    for k in range(len(pt_edges) - 1):
        mask = (centers >= pt_edges[k]) & (centers < pt_edges[k + 1])
        N[k] = vals[mask].sum()
        N_err[k] = np.sqrt((errs[mask] ** 2).sum())
    return N, N_err


def _eec_from_pt_slice(vals_cut, errs_cut, w_centers, rl_widths, N, N_err=None):
    """Weighted, normalized EEC from a pT-sliced (RL x weight) 2D array.
    EEC(RL) = sum_w [ U(RL, w) * w ] / (N_jets * dRL)"""
    vals_cut = np.asarray(vals_cut, dtype=np.float64)
    errs_cut = np.asarray(errs_cut, dtype=np.float64)
    w_centers = np.asarray(w_centers, dtype=np.float64)

    eec = (vals_cut @ w_centers) / rl_widths
    eec_err = np.sqrt((errs_cut ** 2) @ (w_centers ** 2)) / rl_widths

    if N <= 0:
        return np.zeros_like(eec), np.zeros_like(eec_err)

    if N_err is not None:
        eec_err = get_ratio_err(eec, N, eec_err, N_err)   # propagate error on N_jets
    else:
        eec_err = eec_err / N
    eec = eec / N
    return eec, eec_err


def plot_closure_test_axes_singleplot(n_iter, bincenters, binning, binwidths, y_lines, y_points, y_points_err,
                                      L_colors, L_markers, label_lines, label_points,
                                      unfolding=True, extralabel="", closure_label=None):

    fig, axs = plt.subplots(2, 3, figsize=(18, 8), sharex='col',
                            gridspec_kw={'hspace': 0, 'height_ratios': [2, 1]})
    xlabels = [r"jet $p_{T}$", r"$R_{L}$", "energy weight"]

    for i in range(3):  # observable / axis index

        # ---------------- TOP PANELS - DISTRIBUTIONS ----------------
        ax = axs[0, i]
        ax.stairs(y_lines[i], binning[i], color='black', lw=1.8, label=label_lines)

        for it in range(n_iter):
            ax.errorbar(bincenters[i], y_points[it][i], yerr=y_points_err[it][i],
                        fmt=L_markers[it], color=L_colors[it], capsize=5, capthick=2,
                        linestyle="None", zorder=3, # markerfacecolor='none'
                        label=rf"{label_points}, $n_{{\mathrm{{iter}}}}$ = {it+1}")

        #if i in [1, 2]:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel("Counts")
        ax.legend(frameon=False, fontsize=9)
        ax.text(0.1, 0.1, extralabel, transform=ax.transAxes, fontsize=10,
                verticalalignment="top", horizontalalignment="left")
        if closure_label and i == 0:
            ax.text(0.05, 0.95, closure_label, transform=ax.transAxes, fontsize=12,
                    verticalalignment="top", horizontalalignment="left", fontweight='bold')

        # ---------------- BOTTOM PANELS - RATIO ----------------
        ax = axs[1, i]
        ax.axhline(1.05, color='gray', linestyle='--', lw=1.2, zorder=1, label=r"$\pm 5\%$")
        ax.axhline(0.95, color='gray', linestyle='--', lw=1.2, zorder=1)
        if i == 1:  # R_L panel: shade the region above RL_BOX_MIN
            ax.axvspan(RL_BOX_MIN, binning[1][-1], color='gray', alpha=0.2, lw=0,
                       zorder=0, label=rf"$R_L \geq {RL_BOX_MIN}$")
        ax.axhline(1.0, color='gray', linestyle='--', zorder=1)

        for it in range(n_iter):
            yvals_ratio = safe_divide(y_points[it][i], y_lines[i], filler=1)
            ax.scatter(bincenters[i], yvals_ratio, color=L_colors[it], marker=L_markers[it],
                       lw=1.5, zorder=2, label=rf"$n_{{\mathrm{{iter}}}}$ = {it+1}")

            yerrs_ratio = safe_divide(y_points_err[it][i], y_points[it][i])
            ax.fill_between(bincenters[i], (yvals_ratio-yerrs_ratio), (yvals_ratio+yerrs_ratio), alpha=0.4, edgecolor=L_colors[it], facecolor=L_colors[it], linewidth=0)


        ax.set_xlabel(xlabels[i])
        ax.set_ylabel("Unfolded / Truth" if unfolding else "Refolded / Raw")
        ax.legend(frameon=False, fontsize=9, ncol=2)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    plt.show()


def plot_closure_test_fullEEC(lab, h3_raw_lab, h3_refolded_lab, h1_raw, h1_refolded,
                              bincenters, binning, binwidths,
                              L_colors, L_markers,
                              pt_bin_indices=range(1, 8), nrows=2, ncols=4,
                              combine_pt=True,
                              propagate_refolded_Njet_err=True,
                              ratio_ylim=(0.8, 1.2), eec_ylim=None, logy=False,
                              unfolding=False, label_ref="Raw data", label_test="Refolded",
                              extralabel="", n_iter=None, closure_label=None):
    """
    Closure test on the full EEC: one panel (distribution + ratio) per jet pT bin,
    all panels connected with shared x and y axes. Legend goes in the first empty grid slot.

    unfolding=False -> "Refolded / Raw"   (refolding test)
    unfolding=True  -> "Unfolded / Truth" (statistical/shape/prior closure)
    label_ref / label_test set the legend names of the black curve and the markers.
    combine_pt=True (default) -> one EEC(R_L) plot with all selected pT bins
    summed together (default indices: 20-200 GeV), no per-pT differentiation.
    combine_pt=False -> one panel per pT bin instead.
    n_iter=None -> derived from len(h1_refolded).
    """
    n_iter = len(h1_refolded) if n_iter is None else n_iter

    pt_edges = binning[0]
    rl_edges = binning[1]
    rl_centers = bincenters[1]
    rl_widths = np.asarray(binwidths[1], dtype=np.float64)
    w_centers = bincenters[2]

    pt_bin_indices = list(pt_bin_indices)
    if combine_pt:
        # collapse all selected pT bins into a single summed panel
        panel_groups = [pt_bin_indices]
        nrows, ncols = 1, 1
    else:
        panel_groups = [[i] for i in pt_bin_indices]
    n_panels = len(panel_groups)
    if n_panels > nrows * ncols:
        raise ValueError(f"{n_panels} panels don't fit on a {nrows}x{ncols} grid")

    # jet counts per pT bin
    N_raw, N_raw_err = _njets_in_ptbins(h1_raw, pt_edges)
    N_ref, N_ref_err = zip(*[_njets_in_ptbins(h1_refolded[it], pt_edges) for it in range(n_iter)])

    # jet counts per panel (summed over the group's pT bins; errors in quadrature)
    N_raw_g = np.array([N_raw[bins].sum() if combine_pt else N_raw[bins[0]]
                        for bins in panel_groups])
    N_raw_err_g = np.array([np.sqrt((N_raw_err[bins] ** 2).sum()) for bins in panel_groups])
    N_ref_g = np.array([[N_ref[it][bins].sum() if combine_pt else N_ref[it][bins[0]]
                         for it in range(n_iter)] for bins in panel_groups])
    N_ref_err_g = np.array([[np.sqrt((N_ref_err[it][bins] ** 2).sum()) for it in range(n_iter)]
                            for bins in panel_groups])

    # pull values/errors once
    raw_vals, raw_errs = h3_raw_lab.values(), h3_raw_lab.errors()
    ref_vals = [h.values() for h in h3_refolded_lab[:n_iter]]
    ref_errs = [h.errors() for h in h3_refolded_lab[:n_iter]]

    # one gridspec: rows alternate (EEC, ratio), no spacing anywhere
    fig = plt.figure(figsize=(4.2 * ncols, 4.5 * nrows))
    gs = fig.add_gridspec(2 * nrows, ncols, height_ratios=[2, 1] * nrows, hspace=0, wspace=0)

    ax_top_ref, ax_bot_ref = None, None
    legend_handles, legend_labels = None, None

    for k, bins in enumerate(panel_groups):
        i = bins[0]
        row, col = divmod(k, ncols)

        ax_top = fig.add_subplot(gs[2 * row, col], sharex=ax_top_ref, sharey=ax_top_ref)
        ax_bot = fig.add_subplot(gs[2 * row + 1, col],
                                 sharex=ax_top_ref if ax_top_ref is not None else ax_top,
                                 sharey=ax_bot_ref)
        if ax_top_ref is None:
            ax_top_ref, ax_bot_ref = ax_top, ax_bot

        if combine_pt:
            # one slice spanning all selected pT bins (they are contiguous)
            slice_on_pt = lambda a: xslice_th3d(a, bins[0], bins[-1] + 1)
        else:
            slice_on_pt = lambda a: xslice_th3d(a, i, i + 1)

        # ---------------- raw EEC ----------------
        raw_EEC, raw_EEC_err = _eec_from_pt_slice(slice_on_pt(raw_vals), slice_on_pt(raw_errs),
                                                  w_centers, rl_widths, N_raw_g[k])

        # ---------------- refolded EEC per iteration ----------------
        ref_EEC, ref_EEC_err = [], []
        for it in range(n_iter):
            e, ee = _eec_from_pt_slice(slice_on_pt(ref_vals[it]), slice_on_pt(ref_errs[it]),
                                       w_centers, rl_widths, N_ref_g[k][it],
                                       N_ref_err_g[k][it] if propagate_refolded_Njet_err else None)
            ref_EEC.append(e)
            ref_EEC_err.append(ee)

        # ---------------- TOP: distributions ----------------
        ax_top.stairs(raw_EEC, rl_edges, color='black', lw=1.8, label=label_ref)
        ax_top.stairs(raw_EEC + raw_EEC_err, rl_edges, baseline=raw_EEC - raw_EEC_err,
                      fill=True, color='gray', alpha=0.3, lw=0, label="stat. unc.")
        for it in range(n_iter):
            ax_top.errorbar(rl_centers, ref_EEC[it], yerr=ref_EEC_err[it],
                            fmt=L_markers[it], color=L_colors[it], capsize=3, capthick=1.2,
                            markersize=5, linestyle="None", zorder=3,
                            label=rf"{label_test}, $n_{{\mathrm{{iter}}}}$ = {it+1}")
        ax_top.text(0.05, 0.95, rf"jet $p_{{T}}$ = {int(pt_edges[bins[0]])}-{int(pt_edges[bins[-1]+1])} GeV",
                    transform=ax_top.transAxes, fontsize=10,
                    verticalalignment="top", horizontalalignment="left")
        if closure_label:
            ax_top.text(0.05, 0.86, closure_label, transform=ax_top.transAxes, fontsize=11,
                        verticalalignment="top", horizontalalignment="left", fontweight='bold')

        # ---------------- BOTTOM: ratio test / ref ----------------
        ax_bot.axhline(1.05, color='gray', linestyle='--', lw=1.2, zorder=1, label=r"$\pm 5\%$")
        ax_bot.axhline(0.95, color='gray', linestyle='--', lw=1.2, zorder=1)
        ax_bot.axvspan(RL_BOX_MIN, rl_edges[-1], color='gray', alpha=0.2, lw=0,
                       zorder=0, label=rf"$R_L \geq {RL_BOX_MIN}$")
        ax_bot.axhline(1.0, color='gray', linestyle='--', zorder=1)
        for it in range(n_iter):
            ratio = safe_divide(ref_EEC[it], raw_EEC, filler=1)
            ax_bot.scatter(rl_centers, ratio, color=L_colors[it], marker=L_markers[it],
                           s=20, zorder=2)

            yerrs_ratio = safe_divide(ref_EEC_err[it], ref_EEC[it])
            ax_bot.fill_between(rl_centers, (ratio-yerrs_ratio), (ratio+yerrs_ratio), alpha=0.4, edgecolor=L_colors[it], facecolor=L_colors[it], linewidth=0)


        # ---------------- tick label / axis label visibility ----------------
        ax_top.tick_params(labelbottom=False, direction='in', which='both', top=True, right=True)
        ax_bot.tick_params(direction='in', which='both', top=True, right=True)

        if col == 0:
            ax_top.set_ylabel(r"$\frac{1}{N_{\mathrm{jet}}}\frac{d\Sigma_{\mathrm{EEC}}}{dR_{L}}$")
            ax_bot.set_ylabel("Unfolded / Truth" if unfolding else "Refolded / Raw")
        else:
            ax_top.tick_params(labelleft=False)
            ax_bot.tick_params(labelleft=False)

        has_panel_below = (k + ncols) < n_panels
        if has_panel_below:
            ax_bot.tick_params(labelbottom=False)
        else:
            ax_bot.set_xlabel(r"$R_{L}$")

        # grab legend entries from the first panel only
        if legend_handles is None:
            h_top, l_top = ax_top.get_legend_handles_labels()
            h_bot, l_bot = ax_bot.get_legend_handles_labels()
            legend_handles, legend_labels = h_top + h_bot, l_top + l_bot

    # ---------------- shared axis settings (apply to all panels) ----------------
    ax_top_ref.set_xscale('log')
    ax_top_ref.set_xlim(rl_edges[0], rl_edges[-1])
    # explicit ticks away from the panel edges so labels don't collide between columns
    xt = [0.02, 0.05, 0.1, 0.2, 0.5]
    ax_top_ref.set_xticks(xt, labels=[str(x) for x in xt])
    ax_top_ref.xaxis.set_minor_formatter(NullFormatter())

    if logy:
        ax_top_ref.set_yscale('log')
    else:
        # prune bottom tick so it doesn't collide with the ratio panel's top label
        ax_top_ref.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='lower'))
    if eec_ylim is not None:
        ax_top_ref.set_ylim(*eec_ylim)

    ax_bot_ref.set_ylim(*ratio_ylim)
    ax_bot_ref.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))

    # ---------------- shared legend in the first empty slot ----------------
    title = f"{lab} pairs" + (f"\n{extralabel}" if extralabel else "")
    if combine_pt:
        ax_top.legend(legend_handles, legend_labels, loc='upper right', frameon=False,
                      fontsize=10, title=title, title_fontsize=11)
    elif n_panels < nrows * ncols:
        row, col = divmod(n_panels, ncols)
        ax_leg = fig.add_subplot(gs[2 * row:2 * row + 2, col])
        ax_leg.axis('off')
        ax_leg.legend(legend_handles, legend_labels, loc='center', frameon=False,
                      fontsize=11, title=title, title_fontsize=12)
    else:
        fig.legend(legend_handles, legend_labels, loc='upper center',
                   ncol=len(legend_labels), frameon=False, fontsize=10, title=title)

    plt.show()
    return fig


def plot_closure_test_fullEEC_ptbins(lab, h3_raw_lab, h3_refolded_lab, h1_raw, h1_refolded,
                                     bincenters, binning, binwidths,
                                     L_colors, L_markers,
                                     pt_bin_indices=range(1, 8), nrows=2, ncols=4,
                                     propagate_refolded_Njet_err=True,
                                     ratio_ylim=(0.8, 1.2), eec_ylim=None, logy=False,
                                     unfolding=False, label_ref="Raw data", label_test="Refolded",
                                     extralabel="", high_pt=None,
                                     iters_shown=None, closure_label=None):
    """
    Closure test on the full EEC: one panel (distribution + ratio) per jet pT bin,
    all panels connected with shared x and y axes. Legend goes in the first empty grid slot.

    Per-pT-bin version of plot_closure_test_fullEEC (no combine_pt).
    unfolding=False -> "Refolded / Raw"   (refolding test)
    unfolding=True  -> "Unfolded / Truth" (statistical/shape/prior closure)
    label_ref / label_test set the legend names of the black curve and the markers.

    high_pt : dict or None
        If given, every differential panel whose lower pT edge is >= high_pt['pt_min']
        is drawn from an alternative (e.g. RL-rebinned) source instead of the default
        one. Required keys:
          'h3_raw_lab', 'h3_refolded_lab'  -- 3D values (single hist / list per iteration)
          'h1_raw', 'h1_refolded'          -- jet-count source (same population!)
          'binning', 'bincenters', 'binwidths' -- R_L binning of that source
          'pt_min'                         -- lower pT edge from which to use it
    iters_shown : list of 0-based iteration indices, or None (default: all).
        Only the listed iterations are drawn; legend labels keep the true
        iteration number (it+1).
    closure_label : str or None -- closure-test name drawn on each top panel.
    """
    n_iter = len(h1_refolded)
    iters = list(range(n_iter)) if iters_shown is None else list(iters_shown)

    pt_edges = binning[0]
    rl_edges = binning[1]
    rl_centers = bincenters[1]
    rl_widths = np.asarray(binwidths[1], dtype=np.float64)
    w_centers = bincenters[2]

    pt_bin_indices = list(pt_bin_indices)
    n_used = len(pt_bin_indices)
    if n_used > nrows * ncols:
        raise ValueError(f"{n_used} pT bins don't fit on a {nrows}x{ncols} grid")

    # jet counts per pT bin
    N_raw, N_raw_err = _njets_in_ptbins(h1_raw, pt_edges)
    N_ref, N_ref_err = zip(*[_njets_in_ptbins(h1_refolded[it], pt_edges) for it in range(n_iter)])

    # pull values/errors once
    raw_vals, raw_errs = h3_raw_lab.values(), h3_raw_lab.errors()
    ref_vals = [h.values() for h in h3_refolded_lab[:n_iter]]
    ref_errs = [h.errors() for h in h3_refolded_lab[:n_iter]]

    # optional high-pT source (e.g. RL-rebinned file) for the differential panels
    if high_pt is not None:
        hp_pt_edges = high_pt['binning'][0]
        hp_rl_edges = high_pt['binning'][1]
        hp_rl_centers = high_pt['bincenters'][1]
        hp_rl_widths = np.asarray(high_pt['binwidths'][1], dtype=np.float64)
        hp_w_centers = high_pt['bincenters'][2]
        hp_N_raw, hp_N_raw_err = _njets_in_ptbins(high_pt['h1_raw'], hp_pt_edges)
        hp_N_ref, hp_N_ref_err = zip(*[_njets_in_ptbins(high_pt['h1_refolded'][it], hp_pt_edges)
                                       for it in range(n_iter)])
        hp_raw_vals, hp_raw_errs = high_pt['h3_raw_lab'].values(), high_pt['h3_raw_lab'].errors()
        hp_ref_vals = [h.values() for h in high_pt['h3_refolded_lab'][:n_iter]]
        hp_ref_errs = [h.errors() for h in high_pt['h3_refolded_lab'][:n_iter]]

    # one gridspec: rows alternate (EEC, ratio), no spacing anywhere
    fig = plt.figure(figsize=(4.2 * ncols, 4.5 * nrows))
    gs = fig.add_gridspec(2 * nrows, ncols, height_ratios=[2, 1] * nrows, hspace=0, wspace=0)

    ax_top_ref, ax_bot_ref = None, None
    legend_handles, legend_labels = None, None

    for k, i in enumerate(pt_bin_indices):
        row, col = divmod(k, ncols)

        ax_top = fig.add_subplot(gs[2 * row, col], sharex=ax_top_ref, sharey=ax_top_ref)
        ax_bot = fig.add_subplot(gs[2 * row + 1, col],
                                 sharex=ax_top_ref if ax_top_ref is not None else ax_top,
                                 sharey=ax_bot_ref)
        if ax_top_ref is None:
            ax_top_ref, ax_bot_ref = ax_top, ax_bot

        # pick the source for this panel: RL-rebinned file for high-pT bins, fine-RL file otherwise
        if high_pt is not None and pt_edges[i] >= high_pt['pt_min']:
            rl_edges_k, rl_centers_k, rl_widths_k, w_centers_k = hp_rl_edges, hp_rl_centers, hp_rl_widths, hp_w_centers
            raw_vals_k, raw_errs_k = hp_raw_vals, hp_raw_errs
            ref_vals_k, ref_errs_k = hp_ref_vals, hp_ref_errs
            N_raw_i, N_raw_err_i = hp_N_raw[i], hp_N_raw_err[i]
            N_ref_i = [hp_N_ref[it][i] for it in range(n_iter)]
            N_ref_err_i = [hp_N_ref_err[it][i] for it in range(n_iter)]
        else:
            rl_edges_k, rl_centers_k, rl_widths_k, w_centers_k = rl_edges, rl_centers, rl_widths, w_centers
            raw_vals_k, raw_errs_k = raw_vals, raw_errs
            ref_vals_k, ref_errs_k = ref_vals, ref_errs
            N_raw_i, N_raw_err_i = N_raw[i], N_raw_err[i]
            N_ref_i = [N_ref[it][i] for it in range(n_iter)]
            N_ref_err_i = [N_ref_err[it][i] for it in range(n_iter)]

        slice_on_pt = lambda a: xslice_th3d(a, i, i + 1)

        # ---------------- raw EEC ----------------
        raw_EEC, raw_EEC_err = _eec_from_pt_slice(slice_on_pt(raw_vals_k), slice_on_pt(raw_errs_k),
                                                  w_centers_k, rl_widths_k, N_raw_i)

        # ---------------- refolded EEC per iteration ----------------
        ref_EEC, ref_EEC_err = {}, {}   # keyed by iteration index (iters_shown may skip iterations)
        for it in iters:
            e, ee = _eec_from_pt_slice(slice_on_pt(ref_vals_k[it]), slice_on_pt(ref_errs_k[it]),
                                       w_centers_k, rl_widths_k, N_ref_i[it],
                                       N_ref_err_i[it] if propagate_refolded_Njet_err else None)
            ref_EEC[it] = e
            ref_EEC_err[it] = ee

        # ---------------- TOP: distributions ----------------
        ax_top.stairs(raw_EEC, rl_edges_k, color='black', lw=1.8, label=label_ref)
        ax_top.stairs(raw_EEC + raw_EEC_err, rl_edges_k, baseline=raw_EEC - raw_EEC_err,
                      fill=True, color='gray', alpha=0.3, lw=0, label="stat. unc.")
        for it in iters:
            ax_top.errorbar(rl_centers_k, ref_EEC[it], yerr=ref_EEC_err[it],
                            fmt=L_markers[it], color=L_colors[it], capsize=3, capthick=1.2,
                            markersize=5, linestyle="None", zorder=3,
                            label=rf"{label_test}, $n_{{\mathrm{{iter}}}}$ = {it+1}")
        ax_top.text(0.05, 0.95, rf"jet $p_{{T}}$ = {int(pt_edges[i])}-{int(pt_edges[i+1])} GeV",
                    transform=ax_top.transAxes, fontsize=10,
                    verticalalignment="top", horizontalalignment="left")
        if closure_label and k == 0:   # only the first panel
            ax_top.text(0.05, 0.86, closure_label, transform=ax_top.transAxes, fontsize=11,
                        verticalalignment="top", horizontalalignment="left", fontweight='bold')

        # ---------------- BOTTOM: ratio test / ref ----------------
        ax_bot.axhline(1.05, color='gray', linestyle='--', lw=1.2, zorder=1, label=r"$\pm 5\%$")
        ax_bot.axhline(0.95, color='gray', linestyle='--', lw=1.2, zorder=1)
        ax_bot.axvspan(RL_BOX_MIN, rl_edges_k[-1], color='gray', alpha=0.2, lw=0,
                       zorder=0, label=rf"$R_L \geq {RL_BOX_MIN}$")
        ax_bot.axhline(1.0, color='gray', linestyle='--', zorder=1)
        for it in iters:
            ratio = safe_divide(ref_EEC[it], raw_EEC, filler=1)
            ax_bot.scatter(rl_centers_k, ratio, color=L_colors[it], marker=L_markers[it],
                           s=20, zorder=2)

            yerrs_ratio = safe_divide(ref_EEC_err[it], ref_EEC[it])
            ax_bot.fill_between(rl_centers_k, (ratio-yerrs_ratio), (ratio+yerrs_ratio), alpha=0.4, edgecolor=L_colors[it], facecolor=L_colors[it], linewidth=0)


        # ---------------- tick label / axis label visibility ----------------
        ax_top.tick_params(labelbottom=False, direction='in', which='both', top=True, right=True)
        ax_bot.tick_params(direction='in', which='both', top=True, right=True)

        if col == 0:
            ax_top.set_ylabel(r"$\frac{1}{N_{\mathrm{jet}}}\frac{d\Sigma_{\mathrm{EEC}}}{dR_{L}}$")
            ax_bot.set_ylabel("Unfolded / Truth" if unfolding else "Refolded / Raw")
        else:
            ax_top.tick_params(labelleft=False)
            ax_bot.tick_params(labelleft=False)

        has_panel_below = (k + ncols) < n_used
        if has_panel_below:
            ax_bot.tick_params(labelbottom=False)
        else:
            ax_bot.set_xlabel(r"$R_{L}$")

        # grab legend entries from the first panel only
        if legend_handles is None:
            h_top, l_top = ax_top.get_legend_handles_labels()
            h_bot, l_bot = ax_bot.get_legend_handles_labels()
            legend_handles, legend_labels = h_top + h_bot, l_top + l_bot

    # ---------------- shared axis settings (apply to all panels) ----------------
    ax_top_ref.set_xscale('log')
    ax_top_ref.set_xlim(rl_edges[0], rl_edges[-1])
    # explicit ticks away from the panel edges so labels don't collide between columns
    xt = [0.02, 0.05, 0.1, 0.2, 0.5]
    ax_top_ref.set_xticks(xt, labels=[str(x) for x in xt])
    ax_top_ref.xaxis.set_minor_formatter(NullFormatter())

    if logy:
        ax_top_ref.set_yscale('log')
    else:
        # prune bottom tick so it doesn't collide with the ratio panel's top label
        ax_top_ref.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='lower'))
    if eec_ylim is not None:
        ax_top_ref.set_ylim(*eec_ylim)

    ax_bot_ref.set_ylim(*ratio_ylim)
    ax_bot_ref.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))

    # ---------------- shared legend in the first empty slot ----------------
    title = f"{lab} pairs" + (f"\n{extralabel}" if extralabel else "")
    if n_used < nrows * ncols:
        row, col = divmod(n_used, ncols)
        ax_leg = fig.add_subplot(gs[2 * row:2 * row + 2, col])
        ax_leg.axis('off')
        ax_leg.legend(legend_handles, legend_labels, loc='center', frameon=False,
                      fontsize=11, title=title, title_fontsize=12)
    else:
        fig.legend(legend_handles, legend_labels, loc='upper center',
                   ncol=len(legend_labels), frameon=False, fontsize=10, title=title)

    plt.show()
    return fig


def plot_closure_test_fullEEC_ptbins_singleiter(lab, h3_raw_lab, h3_refolded_lab, h1_raw, h1_refolded,
                                                bincenters, binning, binwidths,
                                                L_colors, L_markers,
                                                iteration=3, **kwargs):
    """
    Same as plot_closure_test_fullEEC_ptbins, but only a single chosen iteration
    is drawn. `iteration` is 1-based, matching the n_iter labels in the legends
    (default: 3). All other keyword arguments are forwarded unchanged.
    """
    return plot_closure_test_fullEEC_ptbins(lab, h3_raw_lab, h3_refolded_lab, h1_raw, h1_refolded,
                                            bincenters, binning, binwidths, L_colors, L_markers,
                                            iters_shown=[iteration - 1], **kwargs)


def plot_refolding_iteration_deltas(lab, h3_refolded_lab, h1_refolded,
                                    bincenters, binning, binwidths,
                                    L_colors, L_markers,
                                    pt_bin_indices=range(1, 8), nrows=2, ncols=4,
                                    propagate_refolded_Njet_err=True,
                                    delta_ylim=(-0.05, 0.05), extralabel="", closure_label=None):
    """
    Plots the relative delta between consecutive refolding iterations for the full weighted,
    normalized EEC across jet pT bins:
    Delta_{iter k} = (Refolded_{iter k} - Refolded_{iter k-1}) / Refolded_{iter k-1}
    with error bars.
    """
    n_iter = len(h1_refolded)

    pt_edges = binning[0]
    rl_edges = binning[1]
    rl_centers = bincenters[1]
    rl_widths = np.asarray(binwidths[1], dtype=np.float64)
    w_centers = bincenters[2]

    pt_bin_indices = list(pt_bin_indices)
    n_used = len(pt_bin_indices)
    if n_used > nrows * ncols:
        raise ValueError(f"{n_used} pT bins don't fit on a {nrows}x{ncols} grid")

    # Extract jet counts per pT bin directly from h1_refolded
    N_ref, N_ref_err = zip(*[_njets_in_ptbins(h1_refolded[it], pt_edges) for it in range(n_iter)])

    # Extract 3D values and errors directly from h3_refolded_lab
    ref_vals = [h.values() for h in h3_refolded_lab[:n_iter]]
    ref_errs = [h.errors() for h in h3_refolded_lab[:n_iter]]

    fig = plt.figure(figsize=(4.2 * ncols, 3.0 * nrows))
    gs = fig.add_gridspec(nrows, ncols, hspace=0, wspace=0)

    ax_ref = None
    legend_handles, legend_labels = None, None

    for k, i in enumerate(pt_bin_indices):
        row, col = divmod(k, ncols)

        ax = fig.add_subplot(gs[row, col], sharex=ax_ref, sharey=ax_ref)
        if ax_ref is None:
            ax_ref = ax

        slice_on_pt = lambda a: xslice_th3d(a, i, i + 1)

        # ---------------- Compute refolded EEC per iteration ----------------
        ref_EEC, ref_EEC_err = [], []
        for it in range(n_iter):
            e, ee = _eec_from_pt_slice(slice_on_pt(ref_vals[it]), slice_on_pt(ref_errs[it]),
                                       w_centers, rl_widths, N_ref[it][i],
                                       N_ref_err[it][i] if propagate_refolded_Njet_err else None)
            ref_EEC.append(e)
            ref_EEC_err.append(ee)

        # ---------------- Reference bands & lines ----------------
        ax.axhspan(-0.01, 0.01, color='gray', alpha=0.15, lw=0, zorder=0, label=r"$\pm 1\%$")
        ax.axhline(0.0, color='gray', linestyle='--', zorder=1)

        # ---------------- Plot Iteration Deltas with Error Bars ----------------
        for it in range(1, n_iter):
            # Relative shift: (Iter_k - Iter_{k-1}) / Iter_{k-1}
            delta = safe_divide(ref_EEC[it] - ref_EEC[it - 1], ref_EEC[it - 1], filler=0)

            # Error propagation on ratio shift
            err_k = safe_divide(ref_EEC_err[it], ref_EEC[it - 1])
            err_k_prev = safe_divide(ref_EEC_err[it - 1] * ref_EEC[it], ref_EEC[it - 1]**2)
            delta_err = np.sqrt(err_k**2 + err_k_prev**2)

            ax.scatter(rl_centers, delta, marker=L_markers[it], color=L_colors[it], label=rf"Iter {it+1} - Iter {it}")

            # ax.errorbar(rl_centers, delta, yerr=delta_err,
            #             fmt=L_markers[it], color=L_colors[it], capsize=3, capthick=1.2,
            #             markersize=5, linestyle="None", zorder=3,
            #             label=rf"Iter {it+1} - Iter {it}")

        # ---------------- Panel Labeling & Ticks ----------------
        ax.text(0.05, 0.92, rf"jet $p_{{T}}$ = {int(pt_edges[i])}-{int(pt_edges[i+1])} GeV",
                transform=ax.transAxes, fontsize=10, verticalalignment="top")
        if closure_label and k == 0:
            ax.text(0.05, 0.80, closure_label, transform=ax.transAxes, fontsize=11,
                    verticalalignment="top", fontweight='bold')

        ax.tick_params(direction='in', which='both', top=True, right=True)

        if col == 0:
            ax.set_ylabel(r"$\frac{\mathrm{EEC}_{\mathrm{iter\,k}} - \mathrm{EEC}_{\mathrm{iter\,k-1}}}{\mathrm{EEC}_{\mathrm{iter\,k-1}}}$")
        else:
            ax.tick_params(labelleft=False)

        has_panel_below = (k + ncols) < n_used
        if has_panel_below:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel(r"$R_{L}$")

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    # ---------------- Shared Axis Settings ----------------
    ax_ref.set_xscale('log')
    ax_ref.set_xlim(rl_edges[0], rl_edges[-1])
    xt = [0.02, 0.05, 0.1, 0.2, 0.5]
    ax_ref.set_xticks(xt, labels=[str(x) for x in xt])
    ax_ref.xaxis.set_minor_formatter(NullFormatter())

    ax_ref.set_ylim(*delta_ylim)
    ax_ref.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))

    # ---------------- Legend Placement ----------------
    title = f"{lab} Iteration Convergence" + (f"\n{extralabel}" if extralabel else "")
    if n_used < nrows * ncols:
        row, col = divmod(n_used, ncols)
        ax_leg = fig.add_subplot(gs[row, col])
        ax_leg.axis('off')
        ax_leg.legend(legend_handles, legend_labels, loc='center', frameon=False,
                      fontsize=11, title=title, title_fontsize=12)
    else:
        fig.legend(legend_handles, legend_labels, loc='upper center',
                   ncol=len(legend_labels), frameon=False, fontsize=10, title=title)

    plt.show()
    return fig


def plot_closure_section(lab, tag, h3_ref, h3_unf, ref_label, inp,
                         point_label="Unfolded MC Det", extralabel="", var=None, it_shown=3):
    """Full closure plot set (marginals + differential-pT EEC) for one section/variation.
    The 3D histograms are passed in the h1 slots of the EEC plots so that N_jets
    always comes from the same population as the plotted values.

    Differential pT panels at/above PT_HIGH_MIN are automatically drawn from the
    RL-rebinned files (inp.HIGH_PT_REBIN), keyed by (lab, tag[, var]).
    it_shown : 1-based iteration number shown in the additional single-iteration
    differential-pT plot (default 3).
    """
    n_iter, bincenters, binning, binwidths = inp.n_iter, inp.bincenters, inp.binning, inp.binwidths
    L_colors, L_markers = inp.L_colors, inp.L_markers

    closure_label = CLOSURE_TAG_TEX.get(tag, tag)
    if var is not None:
        closure_label += rf", {VAR_TEX.get(var, var)}"

    y3_ref = proj3D(h3_ref)
    y3_unf = [proj3D(a) for a in h3_unf]
    y3_unf_err = [proj3D_err(a) for a in h3_unf]

    # 1D jet-count normalization for the EEC plots: the 3D hists are filled per
    # jet PAIR, so their pT-slice sums are N_pairs -- the EEC must be divided by
    # N_jets from 1D jet-pT histograms of the same population.
    h1_ref = inp.h1_gen_matched
    if tag == "statistical":
        h1_unf = inp.h1_stat_unfolded_mc
    else:   # shape / prior: same matched truth jets, only pair weights differ
        h1_unf = [inp.h1_gen_matched] * len(h3_unf)

    plot_closure_test_axes_singleplot(n_iter, bincenters, binning, binwidths,
                                      y3_ref, y3_unf, y3_unf_err,
                                      L_colors, L_markers, ref_label, point_label, unfolding=True,
                                      closure_label=closure_label)

    # fig = plot_closure_test_fullEEC(lab, h3_ref, h3_unf, h3_ref, h3_unf,
    #                                 bincenters, binning, binwidths, L_colors, L_markers,
    #                                 unfolding=True, label_ref=ref_label, label_test=point_label,
    #                                 extralabel=extralabel, n_iter=n_iter, closure_label=closure_label)
    fig_ptbins = plot_closure_test_fullEEC_ptbins(lab, h3_ref, h3_unf, h1_ref, h1_unf,
                                                  bincenters, binning, binwidths, L_colors, L_markers,
                                                  unfolding=True, label_ref=ref_label, label_test=point_label,
                                                  extralabel=extralabel, high_pt=inp.high_pt(lab, tag, var),
                                                  closure_label=closure_label)

    # same differential-pT plot, but only the chosen iteration
    fig_single = plot_closure_test_fullEEC_ptbins_singleiter(lab, h3_ref, h3_unf, h1_ref, h1_unf,
                                                             bincenters, binning, binwidths, L_colors, L_markers,
                                                             iteration=it_shown,
                                                             unfolding=True, label_ref=ref_label, label_test=point_label,
                                                             extralabel=extralabel + f" (iter {it_shown} only)",
                                                             high_pt=inp.high_pt(lab, tag, var),
                                                             closure_label=closure_label)

    # fig_delta = plot_refolding_iteration_deltas(lab, h3_unf, h3_unf,
    #                                             bincenters, binning, binwidths, L_colors, L_markers,
    #                                             extralabel=extralabel, closure_label=closure_label)
    return fig_ptbins, fig_single #fig, fig_delta


class ClosureInputs:
    """Container for all files / histograms / binnings used by the closure notebooks."""

    def high_pt(self, lab, tag, var=None):
        """RL-rebinned source dict for the differential pT panels of one section,
        or None if that section has no rebinned counterpart."""
        key = (lab, tag, var) if var is not None else (lab, tag)
        return self.HIGH_PT_REBIN.get(key)


def load_inputs():
    """Open all ROOT files and build every histogram dict the closure notebooks need.
    Returns a ClosureInputs namespace (attribute access, e.g. inp.h3_truth['AA'])."""
    import types

    inp = ClosureInputs()

    # ---- files ----
    inp.f_unfolded        = uproot.open(BASE + "unfolded.root")
    inp.f_unfolded1D      = uproot.open(BASE + "unfolded1D.root")
    inp.f_preunfold_rm    = uproot.open("/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/blianggi/jse/rms/60403841/response_groomed_forunfolding_merged.root")
    inp.f_preunfold_data  = uproot.open("/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/blianggi/jse/data/60320414/AnalysisResultsMerged_groomedbins.root")
    inp.f_closure_tests_1D = uproot.open(BASE + "unfolded1D_tests.root")
    inp.f_closure_tests   = uproot.open(BASE + "unfolded_tests.root")
    inp.f_resp_int        = uproot.open(BASE + "response_internals.root")
    # RL-rebinned (12-bin R_L) counterparts
    inp.f_closure_tests_rebin = uproot.open(BASE + "unfolded_tests_AA_AB_BB_rad_rebinRL2.root")
    inp.f_unfolded_rebin      = uproot.open(BASE + "unfolded_rebinRL2.root")

    # ---- binnings / labels ----
    inp.n_iter = N_ITER
    inp.labels = LABELS
    inp.VARIATIONS = VARIATIONS
    inp.VAR_TEX = VAR_TEX
    inp.binning, inp.bincenters, inp.binwidths = binning, bincenters, binwidths
    inp.binning_rebin2, inp.bincenters_rebin2, inp.binwidths_rebin2 = binning_rebin2, bincenters_rebin2, binwidths_rebin2
    inp.L_colors, inp.L_markers = L_colors, L_markers

    labels = LABELS

    # ---- MC / data inputs (response + data files) ----
    inp.h1_gen_all      = inp.f_preunfold_rm["jet_all_gen_eff_den_groomed"]
    inp.h1_reco_all     = inp.f_preunfold_rm["jet_all_rec_pur_den_groomed"]
    inp.h1_gen_matched  = inp.f_preunfold_rm["jet_match_gen_eff_num_groomed"]
    inp.h1_reco_matched = inp.f_preunfold_rm["jet_match_rec_pur_num_groomed"]
    inp.h1_raw          = inp.f_preunfold_data["groomed_sd0.1_jet_pt_raw1D"]

    inp.h3_gen_all, inp.h3_reco_all, inp.h3_gen_matched, inp.h3_reco_matched, inp.h3_raw = {}, {}, {}, {}, {}
    for lab in labels:
        inp.h3_gen_all[lab]      = inp.f_preunfold_rm[f"groomed_{lab}_gen_all"]
        inp.h3_reco_all[lab]     = inp.f_preunfold_rm[f"groomed_{lab}_reco_all"]
        inp.h3_gen_matched[lab]  = inp.f_preunfold_rm[f"groomed_{lab}_gen_matched"]
        inp.h3_reco_matched[lab] = inp.f_preunfold_rm[f"groomed_{lab}_reco_matched"]
        inp.h3_raw[lab]          = inp.f_preunfold_data[f"{lab}_sd0.1_raw"]

    # response truth (Htruth): all pairs of jet-matched truth jets = level the unfolding returns
    inp.h3_truth = {lab: inp.f_resp_int[f"Htruth_{lab}"] for lab in labels}

    # ---- refolding test ----
    inp.h1_unfolded  = [inp.f_closure_tests_1D[f"refolding_closure/Baysian_Unfolded1Diter{i}"] for i in range(1, N_ITER + 1)]
    inp.h1_refolded  = [inp.f_closure_tests_1D[f"refolding_closure/Baysian_Refolded1Diter{i}"] for i in range(1, N_ITER + 1)]
    inp.h3_unfolded, inp.h3_refolded = {}, {}
    for lab in labels:
        inp.h3_unfolded[lab] = [inp.f_closure_tests[f"refolding_closure/Bayesian_Unfolded_{lab}_iter{i}"] for i in range(1, N_ITER + 1)]
        inp.h3_refolded[lab] = [inp.f_closure_tests[f"refolding_closure/Bayesian_Refolded_{lab}_iter{i}"] for i in range(1, N_ITER + 1)]

    # ---- statistical closure ----
    inp.h3_stat_unfolded_mc = {}
    for lab in labels:
        inp.h3_stat_unfolded_mc[lab] = [inp.f_closure_tests[f"statistical_closure/Statistical_{lab}_Unfolded_iter{i}"]
                                        for i in range(1, N_ITER + 1)]
    inp.h1_stat_unfolded_mc = [inp.f_closure_tests_1D[f"statistical_closure/Statistical_Unfolded1Diter{i}"]
                               for i in range(1, N_ITER + 1)]

    # ---- shape closure ----
    inp.h3_shape_unf = {lab: {var: [inp.f_closure_tests[f"shape_closure/Shape_{lab}_{var}_Unfolded_iter{i}"]
                                    for i in range(1, N_ITER + 1)]
                              for var in VARIATIONS} for lab in labels}
    inp.h3_shape_truth = {lab: {var: inp.f_closure_tests[f"shape_closure/Shape_{lab}_{var}_truth"]
                                for var in VARIATIONS} for lab in labels}

    # ---- prior closure ----
    inp.h3_prior_unf = {lab: {var: [inp.f_closure_tests[f"prior_closure/Prior_{lab}_{var}_Unfolded_iter{i}"]
                                    for i in range(1, N_ITER + 1)]
                              for var in VARIATIONS} for lab in labels}

    # ------------------------------------------------------------------
    # RL-rebinned (12-bin R_L) counterparts, used for the differential pT
    # panels with lower edge >= PT_HIGH_MIN
    # ------------------------------------------------------------------
    # NOTE: do NOT use Htruth_{lab}_rebin2 from unfolded_rebinRL2.root as the
    # closure-test reference: that file comes from a different response production
    # (contents differ by ~300-370x per pT bin and the *normalized* EEC shape by
    # ~10%), so it is not the same population as the closure-test samples. The
    # correct rebinned truth reference is the fine Htruth from response_internals.root
    # merged in R_L pairs (same convention as rebinY_TH3 in unfolding_tests.C) --
    # verified consistent with all rebinned closure hists at the per-mille level.
    inp.h3_truth_rebin = {lab: merge_rl_pairs(inp.h3_truth[lab]) for lab in labels}
    inp.h3_raw_rebin   = {lab: inp.f_closure_tests_rebin[f"h3_raw_{lab}_rebin2"] for lab in labels}

    inp.h3_refolded_rebin = {lab: [inp.f_closure_tests_rebin[f"refolding_closure/Bayesian_Refolded_{lab}_iter{i}"]
                                   for i in range(1, N_ITER + 1)] for lab in labels}
    inp.h3_stat_unfolded_mc_rebin = {lab: [inp.f_closure_tests_rebin[f"statistical_closure/Statistical_{lab}_Unfolded_iter{i}"]
                                           for i in range(1, N_ITER + 1)] for lab in labels}
    inp.h3_shape_unf_rebin = {lab: {var: [inp.f_closure_tests_rebin[f"shape_closure/Shape_{lab}_{var}_Unfolded_iter{i}"]
                                          for i in range(1, N_ITER + 1)]
                                    for var in VARIATIONS} for lab in labels}
    inp.h3_shape_truth_rebin = {lab: {var: inp.f_closure_tests_rebin[f"shape_closure/Shape_{lab}_{var}_truth"]
                                      for var in VARIATIONS} for lab in labels}
    inp.h3_prior_unf_rebin = {lab: {var: [inp.f_closure_tests_rebin[f"prior_closure/Prior_{lab}_{var}_Unfolded_iter{i}"]
                                          for i in range(1, N_ITER + 1)]
                                    for var in VARIATIONS} for lab in labels}

    # per-section rebinned sources for the differential pT panels (used via inp.high_pt(...))
    # NOTE: the h1_* slots must be 1D jet-pT histograms (jet counts). The 3D hists
    # are filled per jet PAIR, so summing a 3D pT slice gives N_pairs, not N_jets --
    # normalizing with it deflates the EEC by the (pT-dependent) pairs-per-jet factor.
    HP = {}
    for lab in labels:
        HP[(lab, "refolding")] = dict(
            pt_min=PT_HIGH_MIN,
            h3_raw_lab=inp.h3_raw_rebin[lab], h1_raw=inp.h1_raw,
            h3_refolded_lab=inp.h3_refolded_rebin[lab], h1_refolded=inp.h1_refolded,
            binning=binning_rebin2, bincenters=bincenters_rebin2, binwidths=binwidths_rebin2)
        HP[(lab, "statistical")] = dict(
            pt_min=PT_HIGH_MIN,
            h3_raw_lab=inp.h3_truth_rebin[lab], h1_raw=inp.h1_gen_matched,
            h3_refolded_lab=inp.h3_stat_unfolded_mc_rebin[lab], h1_refolded=inp.h1_stat_unfolded_mc,
            binning=binning_rebin2, bincenters=bincenters_rebin2, binwidths=binwidths_rebin2)
        for var in VARIATIONS:
            HP[(lab, "shape", var)] = dict(
                pt_min=PT_HIGH_MIN,
                h3_raw_lab=inp.h3_shape_truth_rebin[lab][var], h1_raw=inp.h1_gen_matched,
                h3_refolded_lab=inp.h3_shape_unf_rebin[lab][var], h1_refolded=[inp.h1_gen_matched] * N_ITER,
                binning=binning_rebin2, bincenters=bincenters_rebin2, binwidths=binwidths_rebin2)
            HP[(lab, "prior", var)] = dict(
                pt_min=PT_HIGH_MIN,
                h3_raw_lab=inp.h3_truth_rebin[lab], h1_raw=inp.h1_gen_matched,
                h3_refolded_lab=inp.h3_prior_unf_rebin[lab][var], h1_refolded=[inp.h1_gen_matched] * N_ITER,
                binning=binning_rebin2, bincenters=bincenters_rebin2, binwidths=binwidths_rebin2)
    inp.HIGH_PT_REBIN = HP

    return inp
