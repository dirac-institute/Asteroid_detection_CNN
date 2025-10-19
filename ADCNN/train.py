import time, copy, torch, numpy as np, torch.nn.functional as F
import torch.distributed as dist
from torch import amp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from dist_utils import init_distributed, is_main_process
from losses import blended_loss, make_loss_for_epoch, BCEIoUEdge
from thresholds import resize_masks_to, pick_thr_with_floor
from metrics import roc_auc_ddp


@torch.no_grad()
def _pix_eval(model, loader, thr=0.2, max_batches=12):
    model.eval(); dev = next(model.parameters()).device
    tp=0.0; fp=0.0; fn=0.0; posm=[]; negm=[]
    for i,(xb,yb) in enumerate(loader,1):
        xb,yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
        logits = model(xb)
        yb_r   = resize_masks_to(logits, yb)
        p      = torch.sigmoid(logits)
        if (yb_r>0.5).any(): posm.append(float(p[yb_r>0.5].mean()))
        negm.append(float(p[yb_r<=0.5].mean()))
        pv,tv = p.reshape(-1), yb_r.reshape(-1)
        pred  = (pv>=thr).float()
        tp += float((pred*tv).sum()); fp += float((pred*(1-tv)).sum()); fn += float(((1-pred)*tv).sum())
        if i>=max_batches: break

    # --- DDP: sum tp/fp/fn across ranks ---
    if dist.is_available() and dist.is_initialized():
        vec = torch.tensor([tp, fp, fn], device=dev, dtype=torch.float32)
        dist.all_reduce(vec, op=dist.ReduceOp.SUM)
        tp, fp, fn = map(float, vec.tolist())

    P = tp/max(tp+fp,1.0); R = tp/max(tp+fn,1.0); F = 2*P*R/max(P+R,1e-8)
    return {
        "P":P, "R":R, "F":F,
        "pos_mean": float(sum(posm)/max(len(posm),1)),
        "neg_mean": float(sum(negm)/max(len(negm),1))
    }

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

        is_dist, rank, local_rank, world_size = init_distributed()
        self.amp = True
        scaler = amp.GradScaler('cuda', enabled=self.amp)

        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

        model.to(self.device)
        if is_dist:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                        find_unused_parameters=True, gradient_as_bucket_view=True)

        # Always refer to the underlying module for attribute access (head/u3/u4/aspp) & state_dict
        raw_model = model.module if isinstance(model, DDP) else model

        # init head prior
        from phases import maybe_init_head_bias_to_prior, apply_phase, make_opt_sched, freeze_all
        maybe_init_head_bias_to_prior(raw_model, init_head_prior)

        # -------- Warmup (BCE) --------
        freeze_all(raw_model)
        for p in raw_model.parameters():
            p.requires_grad = True
        posw = torch.tensor(warmup_pos_weight, device=self.device)
        opt = torch.optim.Adam(raw_model.parameters(), lr=warmup_lr, weight_decay=0.0)
        for ep in range(1, warmup_epochs+1):
            if hasattr(train_loader, "sampler") and isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(ep)
            model.train(); seen=0; loss_sum=0.0
            for b,(xb,yb) in enumerate(train_loader, 1):
                xb,yb = xb.to(self.device), yb.to(self.device)
                with amp.autocast('cuda', enabled=self.amp):
                    logits = model(xb)
                    yb_r = resize_masks_to(logits, yb)
                    loss = F.binary_cross_entropy_with_logits(logits, yb_r, pos_weight=posw)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
                loss_sum += float(loss.item())*xb.size(0); seen += xb.size(0)
                if b>=warmup_batches: break
            stats = _pix_eval(model, train_loader, thr=0.2, max_batches=quick_eval_train_batches)
            if is_main_process():
                print(f"[WARMUP] ep{ep} loss {loss_sum/seen:.4f} | F1 {stats['F']:.3f} P {stats['P']:.3f} R {stats['R']:.3f}")

        thr0, *_ = pick_thr_with_floor(
            model, val_loader, max_batches=200, n_bins=256,
            beta=2.0,
            min_pos_rate=thr_pos_rate_early[0],
            max_pos_rate=thr_pos_rate_early[1],
        )
        val_stats = _pix_eval(model, val_loader, thr=float(thr0), max_batches=quick_eval_val_batches)
        auc = roc_auc_ddp(model, val_loader, n_bins=256, max_batches=quick_eval_val_batches)
        if is_main_process():
            print(
                f"[WARMUP VALIDATION] AUC {auc:.3f} P {val_stats['P']:.3f} R {val_stats['R']:.3f} F {val_stats['F']:.3f} | thr={float(thr0):.3f}")

        # -------- Head-only (BCE) --------
        freeze_all(raw_model)
        if hasattr(raw_model, "head"):
            for p in raw_model.head.parameters():
                p.requires_grad = True

        head_posw = torch.tensor(head_pos_weight, device=self.device)
        opt = torch.optim.Adam(
            [p for p in raw_model.parameters() if p.requires_grad], lr=head_lr, weight_decay=0.0
        )
        for ep in range(1, head_epochs+1):
            if hasattr(train_loader, "sampler") and isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(10_000 + ep)
            model.train(); seen=0; loss_sum=0.0
            for b,(xb,yb) in enumerate(train_loader, 1):
                xb,yb = xb.to(self.device), yb.to(self.device)
                with amp.autocast('cuda', enabled=self.amp):
                    logits = model(xb)
                    yb_r = resize_masks_to(logits, yb)
                    loss = F.binary_cross_entropy_with_logits(logits, yb_r, pos_weight=head_posw)

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
                loss_sum += float(loss.item())*xb.size(0); seen += xb.size(0)
                if b>=head_batches: break
            stats = _pix_eval(model, train_loader, thr=thr0, max_batches=quick_eval_train_batches)
            if is_main_process():
                print(f"[HEAD] ep{ep} loss {loss_sum/seen:.4f} | F1 {stats['F']:.3f} P {stats['P']:.3f} R {stats['R']:.3f}")

        # -------- Tail probe (gentle) --------
        freeze_all(raw_model)  # ← raw_model
        if hasattr(raw_model, "head"):
            for p in raw_model.head.parameters():
                p.requires_grad = True
        from phases import _unfreeze_if_exists
        for g in ["u4", "u3", "aspp"]:
            _unfreeze_if_exists(raw_model, g)
        core_probe = BCEIoUEdge(lambda_bce=0.9, pos_weight=tail_pos_weight, lambda_edge=0.0).to(self.device)
        opt = torch.optim.Adam(
            [p for p in raw_model.parameters() if p.requires_grad], lr=tail_lr, weight_decay=1e-4
        )
        for ep in range(1, tail_epochs+1):
            if hasattr(train_loader, "sampler") and isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(20_000 + ep)
            model.train(); seen=0; loss_sum=0.0
            for b,(xb,yb) in enumerate(train_loader, 1):
                xb,yb = xb.to(self.device), yb.to(self.device)
                with amp.autocast('cuda', enabled=self.amp):
                    logits = model(xb)
                    yb_r = resize_masks_to(logits, yb)
                    loss = core_probe(logits, yb_r)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
                loss_sum += float(loss.item())*xb.size(0); seen += xb.size(0)
                if b>=tail_batches: break
        stats = _pix_eval(model, train_loader, thr=thr0, max_batches=quick_eval_train_batches)
        val_stats = _pix_eval(model, val_loader, thr=float(thr0), max_batches=quick_eval_val_batches)
        auc = roc_auc_ddp(model, val_loader, n_bins=256, max_batches=quick_eval_val_batches)
        if is_main_process():
            print(
                f"[TAIL PROBE VALIDATION] AUC {auc:.3f} P {val_stats['P']:.3f} R {val_stats['R']:.3f} F {val_stats['F']:.3f} | thr={float(thr0):.3f}",
                f"[tail-probe] loss≈{loss_sum/seen:.4f}", "[quick_prob_stats] train @ thr0:", {k:round(v,3) for k,v in stats.items()})

        # -------- Long training --------
        best = {"F": -1.0, "state": None, "thr": thr0, "ep": 0}
        metric_thr = thr0

        for ep in range(1, max_epochs+1):
            if hasattr(train_loader, "sampler") and isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(30_000 + ep)
            _ = apply_phase(raw_model, ep)
            core, aftl, w = make_loss_for_epoch(ep, self.device)
            opt, sched = make_opt_sched(raw_model, ep, base_lrs, weight_decay)
            model.train(); seen=0; loss_sum=0.0; t0=time.time()
            for i,(xb,yb) in enumerate(train_loader, 1):
                xb,yb = xb.to(self.device), yb.to(self.device)
                with amp.autocast('cuda', enabled=self.amp):
                    logits = model(xb)
                    yb_r = resize_masks_to(logits, yb)
                    loss = blended_loss(core, aftl, w, logits, yb_r)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
                sched.step(i)
                loss_sum += float(loss.item())*xb.size(0); seen += xb.size(0)
            train_loss = loss_sum/seen
            tr_stats = _pix_eval(model, train_loader, thr=metric_thr, max_batches=quick_eval_train_batches)
            if is_main_process():
                print(f"[EP{ep:02d}] loss {train_loss:.4f} | train P {tr_stats['P']:.3f} R {tr_stats['R']:.3f} F {tr_stats['F']:.3f} "
                    f"| pos≈{tr_stats['pos_mean']:.3f} neg≈{tr_stats['neg_mean']:.3f} | {time.time()-t0:.1f}s")

            if ep % val_every == 0 or ep <= 3:
                if hasattr(val_loader, "sampler") and isinstance(val_loader.sampler, DistributedSampler):
                    val_loader.sampler.set_epoch(40_000 + ep)
                pr_min, pr_max = (thr_pos_rate_early if ep < 26 else thr_pos_rate_late)
                thr, (VP,VR,VF), aux = pick_thr_with_floor(
                    model, val_loader, max_batches=120, n_bins=256, beta=thr_beta,
                    min_pos_rate=pr_min, max_pos_rate=pr_max
                )
                metric_thr = float(thr)
                val_stats = _pix_eval(model, val_loader, thr=metric_thr, max_batches=quick_eval_val_batches)
                auc = roc_auc_ddp(model, val_loader, n_bins=256, max_batches=quick_eval_val_batches)
                if is_main_process():
                    print(f"[VAL ep{ep}] AUC {auc:.3f} P {val_stats['P']:.3f} R {val_stats['R']:.3f} F {val_stats['F']:.3f} | thr={metric_thr:.3f} | pos_rate≈{aux['pos_rate']:.3f}")
                if val_stats['F'] > best["F"]:
                    best = {"F": val_stats['F'], "state": copy.deepcopy(raw_model.state_dict()),
                            "thr": metric_thr, "ep": ep, "P": val_stats["P"], "R": val_stats["R"]}
                    if is_main_process() and save_best_to:
                        torch.save({"state": best["state"], "thr": best["thr"], "ep": best["ep"],
                                    "P": best["P"], "R": best["R"], "F": best["F"]}, save_best_to)

        if best["state"] is not None:
            raw_model.load_state_dict(best["state"], strict=True)
        summary = {"best_F": float(best["F"]), "best_P": float(best.get("P", 0.0)), "best_R": float(best.get("R", 0.0)),
                   "best_ep": int(best["ep"]), "final_thr": float(best["thr"])}
        if is_main_process():
            print("=== DONE ===")
            print("Best summary:", summary)
        trained_model = raw_model
        return trained_model, best["thr"], summary
