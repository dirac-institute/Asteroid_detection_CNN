"""Build the NEO-distance HelioLinC hypothesis grid by filtering the full grid to r < RMAX AU.
NEOs sit near 1 AU; cutting the main-belt distances (r>~1.6) both speeds linking and removes the
main-belt FP chance-clusters that otherwise explode the cluster count. Usage:
  python make_neo_grid.py --src heliohypo_all.txt --rmax 1.6 --out run_neo_wide/heliohypo_neo.txt
"""
import argparse
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="full grid (#r rdot norm accel)")
ap.add_argument("--rmax", type=float, default=1.6)
ap.add_argument("--out", required=True)
a = ap.parse_args()

lines = Path(a.src).read_text().splitlines()
hdr, rows = lines[0], [l for l in lines[1:] if l.strip()]
keep = [l for l in rows if float(l.split()[0]) < a.rmax]
Path(a.out).write_text(hdr + "\n" + "\n".join(keep) + "\n")
print(f"[neo-grid] {len(keep)} of {len(rows)} hypotheses with r < {a.rmax} AU -> {a.out}")
