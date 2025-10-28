import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

def robust_stats_mad(arr):
    med = np.median(arr); mad = np.median(np.abs(arr - med))
    sigma = 1.4826 * (mad + 1e-12)
    return np.float32(med), np.float32(sigma)

def _disk_mask(shape, yc, xc, r):
    H,W = shape; y,x = np.ogrid[:H,:W]
    return (y-yc)**2 + (x-xc)**2 <= r*r

def _line_mask(shape, yc, xc, beta_deg, length, width=1):
    H,W = shape; mask = np.zeros((H,W), bool)
    L = float(length)/2.0; th = np.deg2rad(float(beta_deg))
    dy,dx = np.sin(th), np.cos(th)
    y0,x0 = float(yc)-L*dy, float(xc)-L*dx
    y1,x1 = float(yc)+L*dy, float(xc)+L*dx
    steps = max(2, int(np.ceil(length*2)))
    ys = np.clip(np.rint(np.linspace(y0,y1,steps)).astype(int), 0, H-1)
    xs = np.clip(np.rint(np.linspace(x0,x1,steps)).astype(int), 0, W-1)
    mask[ys,xs] = True
    if width>1:
        rad = max(1,int(width//2))
        mask = ndi.binary_dilation(mask, structure=np.ones((2*rad+1,2*rad+1), bool))
    return mask

def _label_components_fds(mask_bool, pixel_gap=3):
    if pixel_gap>1:
        grown = ndi.binary_dilation(mask_bool, structure=np.ones((2*pixel_gap+1,2*pixel_gap+1), bool))
    else:
        grown = mask_bool
    labels, n = ndi.label(grown, structure=np.ones((3,3), bool))  # 8-connectivity
    return labels, int(n)

def mark_nn_and_stack_fds(csv_path, p_full, thr=0.5, pixel_gap=3, line_width=1):
    cat = pd.read_csv(csv_path).copy()
    need = {"image_id","x","y"}
    miss = need - set(cat.columns)
    if miss: raise ValueError(f"CSV missing columns: {miss}")
    if "stack_detection" in cat.columns: cat["stack_detected"] = cat["stack_detection"].astype(bool)
    elif "stack_mag" in cat.columns:    cat["stack_detected"] = ~cat["stack_mag"].isna()
    else:                               cat["stack_detected"] = False
    H,W = p_full.shape[1:]
    pred_bin = (p_full >= thr)
    labels_list = []
    for i in range(p_full.shape[0]):
        labels_list.append(_label_components_fds(pred_bin[i], pixel_gap=pixel_gap)[0])
    nn = np.zeros(len(cat), bool)
    has_beta = "beta" in cat.columns; has_len = "trail_length" in cat.columns
    for pid, grp in cat.groupby("image_id"):
        pid = int(pid)
        if pid<0 or pid>=len(labels_list): continue
        lab = labels_list[pid]
        for idx in grp.index:
            xc = int(np.clip(int(cat.at[idx,"x"]), 0, W-1))
            yc = int(np.clip(int(cat.at[idx,"y"]), 0, H-1))
            if has_beta and has_len and np.isfinite(cat.at[idx,"beta"]) and np.isfinite(cat.at[idx,"trail_length"]):
                m = _line_mask((H,W), yc, xc, beta_deg=float(cat.at[idx,"beta"]),
                               length=float(cat.at[idx,"trail_length"]), width=line_width)
            else:
                m = _disk_mask((H,W), yc, xc, r=max(1,pixel_gap))
            nn[idx] = (lab[m] > 0).any()
    cat["nn_detected"] = nn
    return cat

# ---- Pixelwise ROC (histogram) ----
def pixelwise_roc_from_hist(p_full, gt_full, nbins=1024):
    gt = (gt_full > 0).astype(np.uint8)
    edges = np.linspace(0.0, 1.0, nbins + 1, dtype=np.float32)
    counts_pos, _ = np.histogram(p_full[gt==1], bins=edges)
    counts_neg, _ = np.histogram(p_full[gt==0], bins=edges)
    tp_cum = np.cumsum(counts_pos[::-1])[::-1]
    fp_cum = np.cumsum(counts_neg[::-1])[::-1]
    P = max(int((gt==1).sum()), 1); N = max(int((gt==0).sum()), 1)
    tpr = tp_cum / P; fpr = fp_cum / N; thr = edges[:-1]
    order = np.argsort(fpr); auc = np.trapz(tpr[order], fpr[order])
    plt.figure(figsize=(6,5)); plt.plot(fpr, tpr); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("Pixelwise ROC"); plt.grid(True, alpha=0.3); plt.show()
    return fpr, tpr, thr, auc

# ---- Object candidate scores (FDS) ----
def object_candidates_and_scores(p_full, gt_full, base_thr=0.05, pixel_gap=3):
    scores, labels = [], []
    for i in range(p_full.shape[0]):
        lab, n = _label_components_fds(p_full[i] >= base_thr, pixel_gap=pixel_gap)
        if n==0: continue
        overlap = (gt_full[i] > 0)
        comp_ids = np.arange(1, n+1, dtype=np.int32)
        comp_max = ndi.maximum(p_full[i], labels=lab, index=comp_ids)
        comp_overlap = ndi.sum(overlap.astype(np.uint8), labels=lab, index=comp_ids)
        scores.extend(comp_max.tolist()); labels.extend((comp_overlap > 0).tolist())
    return np.asarray(scores, np.float32), np.asarray(labels, bool)

def roc_from_scores(pos_scores, neg_scores, num_thresh=512):
    if len(pos_scores)==0 or len(neg_scores)==0:
        return np.array([0.]), np.array([0.]), np.array([1.]), 0.0
    lo = float(min(pos_scores.min(), neg_scores.min()))
    hi = float(max(pos_scores.max(), neg_scores.max()))
    thresholds = np.linspace(hi, lo, num_thresh)  # descending
    P,N = len(pos_scores), len(neg_scores)
    pos_sorted = np.sort(pos_scores)[::-1]; neg_sorted = np.sort(neg_scores)[::-1]
    tpr = np.empty_like(thresholds); fpr = np.empty_like(thresholds)
    for k,t in enumerate(thresholds):
        tp = np.searchsorted(-pos_sorted, -t, side='right')
        fp = np.searchsorted(-neg_sorted, -t, side='right')
        tpr[k] = tp / P; fpr[k] = fp / N
    order = np.argsort(fpr); auc = np.trapz(tpr[order], fpr[order])
    return fpr, tpr, thresholds, auc

# ---- Confusions & F-scores ----
def pixelwise_confusion(p_full, gt_full, thr):
    gt = (gt_full > 0).astype(np.uint8); pred = (p_full >= thr).astype(np.uint8)
    tp = int(((pred==1)&(gt==1)).sum()); fp = int(((pred==1)&(gt==0)).sum())
    fn = int(((pred==0)&(gt==1)).sum()); tn = int(((pred==0)&(gt==0)).sum())
    prec = tp / max(tp+fp, 1); rec = tp / max(tp+fn, 1)
    f1 = 2*prec*rec / max(prec+rec, 1e-12)
    beta2 = 2.0; f2 = (1+beta2**2)*prec*rec / max(beta2**2*prec + rec, 1e-12)
    return (tp,fp,fn,tn), (prec,rec,f1,f2)

def objectwise_confusion_from_candidates(scores, labels, thr):
    pred_pos = scores >= thr
    tp = int(((pred_pos==True)&(labels==True)).sum())
    fp = int(((pred_pos==True)&(labels==False)).sum())
    fn = int(((pred_pos==False)&(labels==True)).sum())
    tn = int(((pred_pos==False)&(labels==False)).sum())
    prec = tp / max(tp+fp, 1); rec = tp / max(tp+fn, 1)
    f1 = 2*prec*rec / max(prec+rec, 1e-12)
    beta2 = 2.0; f2 = (1+beta2**2)*prec*rec / max(beta2**2*prec + rec, 1e-12)
    return (tp,fp,fn,tn), (prec,rec,f1,f2)

def print_confusion_matrix(cm, title="Confusion Matrix"):
    tp,fp,fn,tn = cm
    import pandas as _pd
    df = _pd.DataFrame(
        [[tp, fp],
         [fn, tn]],
        index=["Actual +", "Actual -"],
        columns=["Pred +", "Pred -"]
    )
    print(title)
    display(df.style.format(na_rep="-").set_properties(**{"text-align":"center"}))

def _choose_mag_field(df):
    for c in ["PSF_mag","integrated_mag","mag"]:
        if c in df.columns: return c
    return None

def plot_detect_hist(cat, field, bins=12, title=None):
    nn_det = cat[cat["nn_detected"]]
    stk_det = cat[cat["stack_detected"]]
    cum_det = cat[cat["nn_detected"] | cat["stack_detected"]]
    vals = cat[field].to_numpy(); vals = vals[np.isfinite(vals)]
    edges = np.histogram_bin_edges(vals, bins=bins)
    fig, ax = plt.subplots(figsize=(6.5,4.5))
    ax.hist(cat[field],     bins=edges, histtype="step", label="All injected", alpha=0.7)
    ax.hist(cum_det[field], bins=edges, histtype="step", label="Cumulative (NN ∪ LSST)")
    ax.hist(nn_det[field],  bins=edges, histtype="step", label="NN detected")
    ax.hist(stk_det[field], bins=edges, histtype="step", label="LSST stack detected")
    ax.set_xlabel(field.replace("_"," ")); ax.set_ylabel("Count")
    if title: ax.set_title(title)
    ax.legend(); ax.grid(True, alpha=0.3)
    return fig, ax
