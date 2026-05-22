"""Task-metric eval of pilot_v7_huge (3980-panel) BEST checkpoint vs v7-big (1749):
SYNTH test_5sigma v7-only objectwise recall + REAL in-region stack-missed fire@truth.
Apples-to-apples (same test sets), unlike the val_auc which used a different val split.
"""
import sys
from pathlib import Path
REPO = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments/explore_simreal_gap"))
import torch
from eval_best_vs_last import real_fire, synth_recall

CK = REPO / "experiments/diffim_runs/pilot_v7_huge/ckpts/v7_huge_best_scripted.pt"

def main():
    dev = torch.device("cuda")
    m = torch.jit.load(str(CK), map_location=dev).eval()
    n, f5, f3, med = real_fire(m, dev)
    stp, sfn, srec = synth_recall(m, dev)
    print("\n===== pilot_v7_huge BEST (ep9, ~3980 panels) =====", flush=True)
    print(f"  SYNTH test_5sigma v7-only objectwise recall: {stp}/{stp+sfn} = {100*srec:.1f}%  (v7-big: 96.4%)", flush=True)
    print(f"  REAL in-region stack-missed v7-fire@truth: n={n}  >=0.5={f5} ({100*f5/n:.0f}%)  "
          f">=0.3={f3} ({100*f3/n:.0f}%)  med_pmax={med:.3f}   (v7-big: 77% @>=0.5)", flush=True)
    print("EVAL HUGE DONE", flush=True)

if __name__ == "__main__":
    main()
