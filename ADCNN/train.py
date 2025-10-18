import time, copy, torch, numpy as np, torch.nn.functional as F
from losses import blended_loss, make_loss_for_epoch, BCEIoUEdge
from thresholds import resize_masks_to, pick_thr_with_floor

@torch.no_grad()
def _pix_eval(model, loader, thr=0.2, max_batches=12):
    model.eval(); dev = next(model.parameters()).device
    tp=fp=fn=0.0; posm=[]; negm=[]
    t0=time.time()
    for i,(xb,yb) in enumerate(loader,1):
        xb,yb = xb.to(dev), yb.to(dev)
        logits = model(xb)
        yb_r   = resize_masks_to(logits, yb)
        p      = torch.sigmoid(logits)
        if (yb_r>0.5).any(): posm.append(float(p[yb_r>0.5].mean()))
        negm.append(float(p[yb_r<=0.5].mean()))
        pv,tv = p.view(-1), yb_r.view(-1)
        pred = (pv>=thr).float()
        tp += float((pred*tv).sum()); fp += float((pred*(1-tv)).sum()); fn += float(((1-pred)*tv).sum())
        if i>=max_batches: break
    P = tp/max(tp+fp,1); R = tp/max(tp+fn,1); F = 2*P*R/max(P+R,1e-8)
    return {"P":P,"R":R,"F":F,"pos_mean":float(sum(posm)/max(len(posm),1)), "neg_mean":float(sum(negm)/len(negm))}

class Trainer:
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train_full_probe(self, model, train_loader, val_loader, *,
                         seed=1337,
                         init_head_prior=0.70,
                         # warmup
                         warmup_epochs=1, warmup_batches=800, warmup_lr=2e-4, warmup_pos_weight=40.0,
                         # head
                         head_epochs=2, head_batches=2000, head_lr=3e-5, head_pos_weight=5.0,
                         # tail
                         tail_epochs=2, tail_batches=2500, tail_lr=1.5e-4, tail_pos_weight=2.0,
                         # long
                         max_epochs=60, val_every=3, base_lrs=(3e-4,2e-4,1e-4), weight_decay=1e-4,
                         # thr
                         thr_beta=1.0, thr_pos_rate_early=(0.03,0.10), thr_pos_rate_late=(0.08,0.12),
                         save_best_to="ckpt_best.pt",
                         quick_eval_train_batches=6, quick_eval_val_batches=12):

        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        model.to(self.device)

        # init head prior
        from phases import maybe_init_head_bias_to_prior, apply_phase, make_opt_sched, freeze_all
        maybe_init_head_bias_to_prior(model, init_head_prior)

        # -------- Warmup (BCE) --------
        print("Warmup…")
        freeze_all(model)
        for p in model.parameters(): p.requires_grad = True
        posw = torch.tensor(warmup_pos_weight, device=self.device)
        opt = torch.optim.Adam(model.parameters(), lr=warmup_lr, weight_decay=0.0)
        for ep in range(1, warmup_epochs+1):
            model.train(); seen=0; loss_sum=0.0
            for b,(xb,yb) in enumerate(train_loader, 1):
                xb,yb = xb.to(self.device), yb.to(self.device)
                logits = model(xb); yb_r = resize_masks_to(logits, yb)
                loss = F.binary_cross_entropy_with_logits(logits, yb_r, pos_weight=posw)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                loss_sum += float(loss.item())*xb.size(0); seen += xb.size(0)
                if b>=warmup_batches: break
            stats = _pix_eval(model, train_loader, thr=0.2, max_batches=quick_eval_train_batches)
            print(f"[WARMUP] ep{ep} loss {loss_sum/seen:.4f} | F1 {stats['F']:.3f} P {stats['P']:.3f} R {stats['R']:.3f}")

        thr0, *_ = pick_thr_with_floor(model, val_loader, max_batches=200, n_bins=256,
                                       beta=2.0, min_pos_rate=thr_pos_rate_early[0], max_pos_rate=thr_pos_rate_early[1])
        thr0 = float(max(0.05, min(0.20, thr0)))
        print(f"[thr0] ≈ {thr0:.3f}")

        # -------- Head-only (BCE) --------
        freeze_all(model)
        if hasattr(model, "head"):
            for p in model.head.parameters(): p.requires_grad = True
        head_posw = torch.tensor(head_pos_weight, device=self.device)
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=head_lr, weight_decay=0.0)
        for ep in range(1, head_epochs+1):
            model.train(); seen=0; loss_sum=0.0
            for b,(xb,yb) in enumerate(train_loader, 1):
                xb,yb = xb.to(self.device), yb.to(self.device)
                logits = model(xb); yb_r = resize_masks_to(logits, yb)
                loss = F.binary_cross_entropy_with_logits(logits, yb_r, pos_weight=head_posw)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                loss_sum += float(loss.item())*xb.size(0); seen += xb.size(0)
                if b>=head_batches: break
            stats = _pix_eval(model, train_loader, thr=thr0, max_batches=quick_eval_train_batches)
            print(f"[HEAD] ep{ep} loss {loss_sum/seen:.4f} | F1 {stats['F']:.3f} P {stats['P']:.3f} R {stats['R']:.3f}")

        # -------- Tail probe (gentle) --------
        freeze_all(model)
        if hasattr(model, "head"):
            for p in model.head.parameters(): p.requires_grad = True
        # unfreeze tails best-effort
        from phases import _unfreeze_if_exists
        for g in ["u4","u3","aspp"]:
            _unfreeze_if_exists(model, g)
        core_probe = BCEIoUEdge(lambda_bce=0.9, pos_weight=tail_pos_weight, lambda_edge=0.0).to(self.device)
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=tail_lr, weight_decay=1e-4)
        for ep in range(1, tail_epochs+1):
            model.train(); seen=0; loss_sum=0.0
            for b,(xb,yb) in enumerate(train_loader, 1):
                xb,yb = xb.to(self.device), yb.to(self.device)
                logits = model(xb); yb_r = resize_masks_to(logits, yb)
                loss = core_probe(logits, yb_r)
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                loss_sum += float(loss.item())*xb.size(0); seen += xb.size(0)
                if b>=tail_batches: break
        stats = _pix_eval(model, train_loader, thr=thr0, max_batches=quick_eval_train_batches)
        print(f"[tail-probe] loss≈{loss_sum/seen:.4f}")
        print("[quick_prob_stats] train @ thr0:", {k:round(v,3) for k,v in stats.items()})

        # -------- Long training --------
        best = {"F": -1.0, "state": None, "thr": thr0, "ep": 0}
        metric_thr = thr0

        for ep in range(1, max_epochs+1):
            _ = apply_phase(model, ep)
            core, aftl, w = make_loss_for_epoch(ep, self.device)
            opt, sched = make_opt_sched(model, ep, base_lrs, weight_decay)

            model.train(); seen=0; loss_sum=0.0; t0=time.time()
            for i,(xb,yb) in enumerate(train_loader, 1):
                xb,yb = xb.to(self.device), yb.to(self.device)
                logits = model(xb); yb_r = resize_masks_to(logits, yb)
                loss = blended_loss(core, aftl, w, logits, yb_r)
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(i)
                loss_sum += float(loss.item())*xb.size(0); seen += xb.size(0)
            train_loss = loss_sum/seen
            tr_stats = _pix_eval(model, train_loader, thr=metric_thr, max_batches=quick_eval_train_batches)
            print(f"[EP{ep:02d}] loss {train_loss:.4f} | train P {tr_stats['P']:.3f} R {tr_stats['R']:.3f} F {tr_stats['F']:.3f} "
                  f"| pos≈{tr_stats['pos_mean']:.3f} neg≈{tr_stats['neg_mean']:.3f} | {time.time()-t0:.1f}s")

            if ep % val_every == 0 or ep <= 3:
                pr_min, pr_max = (thr_pos_rate_early if ep < 26 else thr_pos_rate_late)
                thr, (VP,VR,VF), aux = pick_thr_with_floor(
                    model, val_loader, max_batches=120, n_bins=256, beta=thr_beta,
                    min_pos_rate=pr_min, max_pos_rate=pr_max
                )
                metric_thr = float(thr)
                print(f"[thr@ep{ep}] thr={metric_thr:.3f} | val P {VP:.3f} R {VR:.3f} F {VF:.3f} | pos_rate≈{aux['pos_rate']:.3f}")

                val_stats = _pix_eval(model, val_loader, thr=metric_thr, max_batches=quick_eval_val_batches)
                print(f"[VAL ep{ep}] P {val_stats['P']:.3f} R {val_stats['R']:.3f} F {val_stats['F']:.3f}")
                if val_stats['F'] > best["F"]:
                    best = {"F": val_stats['F'], "state": copy.deepcopy(model.state_dict()),
                            "thr": metric_thr, "ep": ep, "P": val_stats["P"], "R": val_stats["R"]}
                    if save_best_to:
                        torch.save({"state": best["state"], "thr": best["thr"], "ep": best["ep"],
                                    "P":best["P"], "R":best["R"], "F":best["F"]}, save_best_to)
                        print(f"  ↳ saved best → {save_best_to} (F={best['F']:.3f}, thr={best['thr']:.3f}, ep={best['ep']})")

        if best["state"] is not None:
            model.load_state_dict(best["state"], strict=True)
        summary = {"best_F": float(best["F"]), "best_P": float(best.get("P", 0.0)), "best_R": float(best.get("R", 0.0)),
                   "best_ep": int(best["ep"]), "final_thr": float(best["thr"])}
        print("=== DONE ===")
        print("Best summary:", summary)
        return model, best["thr"], summary
