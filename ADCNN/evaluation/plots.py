"""Notebook visualisation helpers for the detection evaluation.

Pure plotting / pretty-printing over the confusion-matrix and catalog outputs of
``detection.py`` (kept out of that module so it stays compute-only). Re-exported from
``detection`` for back-compat with the ``Evaluation/*.ipynb`` notebooks.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_detect_hist(cat, field, bins=12, title=None, density=False, xlim=None):
    nn_det = cat[cat["nn_detected"]]
    stk_det = cat[cat["stack_detection"]]
    cum_det = cat[cat["nn_detected"] | cat["stack_detection"]]
    vals = cat[field].to_numpy(); vals = vals[np.isfinite(vals)]
    if xlim is not None:
        vals = vals[(vals>=xlim[0]) & (vals<=xlim[1])]
    edges = np.histogram_bin_edges(vals, bins=bins)
    fig, ax = plt.subplots(figsize=(6.5,4.5))
    if not density:
        ax.hist(cat[field],     bins=edges, histtype="step", label="All injected", alpha=0.7)
        ax.hist(cum_det[field], bins=edges, histtype="step", label="Cumulative (NN ∪ LSST)")
        ax.hist(nn_det[field],  bins=edges, histtype="step", label="NN detected")
        ax.hist(stk_det[field], bins=edges, histtype="step", label="LSST stack detected")
        ax.set_xlabel(field.replace("_"," ")); ax.set_ylabel("Count")
    else:
        all_i = np.histogram(cat[field],     bins=edges)[0]
        cum_i = np.histogram(cum_det[field], bins=edges)[0]
        nn_i  = np.histogram(nn_det[field],  bins=edges)[0]
        stk_i = np.histogram(stk_det[field], bins=edges)[0]
        bin_widths = np.diff(edges)
        ax.stairs(cum_i/all_i, edges, label="Cumulative (NN ∪ LSST)")
        ax.stairs(nn_i/all_i,  edges, label="NN detected")
        ax.stairs(stk_i/all_i, edges, label="LSST stack detected")
        ax.set_xlabel(field.replace("_"," ")); ax.set_ylabel("Completeness")
    if title: ax.set_title(title)
    ax.legend(); ax.grid(True, alpha=0.3)
    return fig, ax

def print_confusion_matrix(cm, title="Confusion Matrix"):
    cm = np.array([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]])
    df = pd.DataFrame(cm, index=["Actual Negative", "Actual Positive"], columns=["Predicted Negative", "Predicted Positive"], dtype="Int64")
    print(title)
    print (f"F1 Score: {cm[1,1]*2/(cm[1,1]*2 + cm[0,1] + cm[1,0]):.4f}, F2 Score: {cm[1,1]*5/(cm[1,1]*5 + cm[0,1] + cm[1,0]*4):.4f}")
    print(df)
    print()

def plot_completeness_2d(catalog, snr_bins=10, trail_bins=10,
                        snr_range=None, trail_range=None, detected=None, title=None):

    snr = catalog["SNR"].values
    trail = catalog["trail_length"].values

    if detected is None:
        detected = catalog["nn_detected"].values.astype(bool)

    if trail_range is None:
        trail_range = (trail.min(), trail.max())

    if snr_range is None:
        snr_range = (snr.min(), snr.max())

    snr_edges = np.linspace(*snr_range, snr_bins + 1)
    trail_edges = np.linspace(*trail_range, trail_bins + 1)

    # total objects per bin
    H_total, _, _ = np.histogram2d(snr, trail,
                                   bins=[snr_edges, trail_edges])

    # detected objects per bin
    H_det, _, _ = np.histogram2d(snr[detected], trail[detected],
                                 bins=[snr_edges, trail_edges])

    completeness = np.divide(
        H_det, H_total,
        out=np.zeros_like(H_det),
        where=H_total > 0
    )

    fig, ax = plt.subplots(figsize=(6,5))

    im = ax.imshow(
        completeness.T,
        origin="lower",
        aspect="auto",
        extent=[snr_edges[0], snr_edges[-1], trail_edges[0], trail_edges[-1]],
        vmin=0, vmax=1,
        cmap="viridis"
    )

    plt.colorbar(im, ax=ax, label="Completeness")

    # annotate bins
    for i in range(snr_bins):
        for j in range(trail_bins):
            if H_total[i, j] > 1:
                x = 0.5 * (snr_edges[i] + snr_edges[i+1])
                y = 0.5 * (trail_edges[j] + trail_edges[j+1])
                ax.text(x, y, f"{completeness[i,j]:.2f}",
                        ha="center", va="center", color="white", fontsize=8)

    ax.set_xlabel("SNR")
    ax.set_ylabel("Trail length")
    if title is not None:
        ax.set_title(title)
    plt.tight_layout()
    return fig, ax
