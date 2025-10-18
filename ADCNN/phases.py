import math, torch
from typing import List

def set_requires_grad(mod, flag: bool):
    for p in mod.parameters(): p.requires_grad = flag

def freeze_all(model): set_requires_grad(model, False)

def _unfreeze_if_exists(model, path: str):
    mod = model
    for name in path.split('.'):
        if not hasattr(mod, name): return False
        mod = getattr(mod, name)
    for p in mod.parameters(): p.requires_grad = True
    return True

def maybe_init_head_bias_to_prior(model, p0=0.70):
    if p0 is None: return
    if hasattr(model, "head") and hasattr(model.head, "bias") and model.head.bias is not None:
        with torch.no_grad():
            b = math.log(p0/(1-p0))
            model.head.bias.data.fill_(b)

def apply_phase(model, ep: int) -> list[str]:
    """
    1–3: head only
    4–12: head + tail(u4,u3,aspp)
    13–25: head + u2,u3,u4,aspp
    26+: full
    """
    freeze_all(model)
    groups: List[str] = []
    if ep <= 3:
        if hasattr(model, "head"):
            for p in model.head.parameters(): p.requires_grad = True
        groups = ["head"]
    elif ep <= 12:
        if hasattr(model, "head"):
            for p in model.head.parameters(): p.requires_grad = True
        for g in ["u4","u3","aspp"]:
            if _unfreeze_if_exists(model, g): groups.append(g)
    elif ep <= 25:
        if hasattr(model, "head"):
            for p in model.head.parameters(): p.requires_grad = True
        for g in ["u4","u3","u2","aspp"]:
            if _unfreeze_if_exists(model, g): groups.append(g)
    else:
        for p in model.parameters(): p.requires_grad = True
        groups = ["<FULL>"]
    ntrain = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[phase] ep={ep} | trainable params={ntrain:,} | groups={groups}")
    return groups

def make_opt_sched(model, ep: int, base_lrs, weight_decay: float):
    if ep <= 12:  base_lr = base_lrs[0]
    elif ep <= 25: base_lr = base_lrs[1]
    else:          base_lr = base_lrs[2]
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=base_lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=6, T_mult=2, eta_min=base_lr/10)
    return opt, sched
