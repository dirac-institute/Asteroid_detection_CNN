"""Task-metric eval of pilot_seg_huge (3980-panel) BEST checkpoint vs seg_model-big (1749):
SYNTH test_5sigma seg_model-only objectwise recall + REAL in-region stack-missed fire@truth.
Apples-to-apples (same test sets), unlike the val_auc which used a different val split.
"""
import sys
from pathlib import Path
REPO = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments/explore_simreal_gap"))
import torch
from eval_best_vs_last import real_fire, synth_recall

CK = REPO / "experiments/diffim_runs/pilot_seg_huge/ckpts/seg_huge_best_scripted.pt"

def main():
    ck = sys.argv[1] if len(sys.argv) > 1 else str(CK)
    label = sys.argv[2] if len(sys.argv) > 2 else "pilot_seg_huge"
    dev = torch.device("cuda")
    m = torch.jit.load(ck, map_location=dev).eval()
    n, f5, f3, med = real_fire(m, dev)
    stp, sfn, srec = synth_recall(m, dev)
    print(f"\n===== {label} BEST =====", flush=True)
    print(f"  SYNTH test_5sigma seg_model-only objectwise recall: {stp}/{stp+sfn} = {100*srec:.1f}%  (seg_model-big: 96.4%)", flush=True)
    print(f"  REAL in-region stack-missed seg_model-fire@truth: n={n}  >=0.5={f5} ({100*f5/n:.0f}%)  "
          f">=0.3={f3} ({100*f3/n:.0f}%)  med_pmax={med:.3f}   (seg_model-big: 77% @>=0.5)", flush=True)
    print("EVAL HUGE DONE", flush=True)

if __name__ == "__main__":
    main()
