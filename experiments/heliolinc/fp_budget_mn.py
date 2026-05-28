"""Multi-node grid-sharded HelioLinC runner with streaming progress — shared by the FP-budget
experiments (fp_budget_mc.py, fp_budget_completeness.py).

The heliohypo grid points are independent, so one FP-budget run is split THREE ways:
  prep      build tracklets once into the run dir (pairdets.csv / pairs.txt / grid / meta.json);
  shard     each of NNODE nodes runs heliolinc over its slice grid[node::NNODE], split into many
            local chunks driven through a bounded pool of NCORES (oversubscribed so completions
            STREAM IN -> a real progress %/ETA), publishing cluster files to <rd>/clusters_mn/;
  finalize  link_refine over every node's published shards -> lr.csv / lr_rms.csv.

Each node's chunk files are prefixed with the node id so the three nodes never clash on shared disk.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path


def write_meta(rd: Path, **kw) -> None:
    (rd / "meta.json").write_text(json.dumps(kw))


def read_meta(rd: Path) -> dict:
    return json.loads((rd / "meta.json").read_text())


def _split_grid(grid_path: Path, node_idx: int, nnode: int, nchunks: int, scratch: Path) -> list[Path]:
    """This node's slice = grid lines where (line % nnode == node_idx); round-robin it into
    `nchunks` local chunk files under `scratch`. Returns the chunk file paths."""
    lines = grid_path.read_text().splitlines()
    hdr, body = lines[0], lines[1:]
    mine = [ln for i, ln in enumerate(body) if i % nnode == node_idx]
    scratch.mkdir(parents=True, exist_ok=True)
    chunks: list[Path] = []
    for s in range(nchunks):
        ch = mine[s::nchunks]
        if not ch:
            continue
        fp = scratch / f"grid_{s:04d}.txt"
        fp.write_text(hdr + "\n" + "\n".join(ch) + "\n")
        chunks.append(fp)
    return chunks


def run_grid_shards(rd: Path, grid_path: Path, mjdref: float, *, node_idx: int, nnode: int,
                    ncores: int, bin_dir: Path, clustrad: float = 100000.0, npt: int = 3,
                    minnights: int = 3, mintimespan: float = 0.5, shards_per_core: int = 4,
                    progress_s: float = 30.0, tag: str = "") -> int:
    """Run this node's grid slice through a bounded pool of `ncores` heliolinc processes, streaming
    a progress line (done/total + rate + ETA) every `progress_s` seconds. Publishes each completed
    chunk's clusters to ``<rd>/clusters_mn/hl_{clusters,summary}_<node>_<chunk>.csv``. Returns the
    number of failed chunks."""
    shared = rd / "clusters_mn"; shared.mkdir(exist_ok=True)
    scratch = Path(f"/lscratch/{os.environ.get('USER','u')}/fpbmn_{os.environ.get('SLURM_JOB_ID','x')}_{node_idx}")
    chunks = _split_grid(grid_path, node_idx, nnode, max(1, ncores * shards_per_core), scratch)
    total = len(chunks)
    print(f"[shard {tag} node {node_idx}/{nnode}] {total} local chunks over {ncores} cores "
          f"(grid slice, clustrad={clustrad:.0f})", flush=True)

    running: dict = {}            # Popen -> chunk_path
    done = fail = 0
    i = 0
    t0 = time.time(); last = t0

    def launch(ch: Path):
        s = ch.stem.split("_")[-1]
        out_c = scratch / f"hl_clusters_{s}.csv"; out_s = scratch / f"hl_summary_{s}.csv"
        p = subprocess.Popen([str(bin_dir / "heliolinc"), "-dets", "pairdets.csv", "-pairs", "pairs.txt",
                              "-mjd", str(mjdref), "-obspos", "Earth1day2020s_02a.txt",
                              "-heliodist", str(ch), "-clustrad", str(clustrad), "-npt", str(npt),
                              "-minobsnights", str(minnights), "-mintimespan", str(mintimespan),
                              "-out", str(out_c), "-outsum", str(out_s)],
                             cwd=rd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        running[p] = (ch, out_c, out_s, s)

    while i < total and len(running) < ncores:
        launch(chunks[i]); i += 1

    while running:
        for p in list(running):
            if p.poll() is None:
                continue
            ch, out_c, out_s, s = running.pop(p)
            if p.returncode == 0 and out_c.exists():
                if out_c.stat().st_size > 0:   # publish only non-empty cluster files
                    (shared / f"hl_clusters_{node_idx}_{s}.csv").write_bytes(out_c.read_bytes())
                    (shared / f"hl_summary_{node_idx}_{s}.csv").write_bytes(out_s.read_bytes())
            else:
                fail += 1
            done += 1
            if i < total:
                launch(chunks[i]); i += 1
        now = time.time()
        if now - last >= progress_s or (done == total and not running):
            rate = done / max(now - t0, 1e-9)
            eta = (total - done) / rate if rate > 0 else 0.0
            print(f"[shard {tag} node {node_idx}] {done}/{total} chunks | {fail} fail | "
                  f"{rate*60:.1f} chunk/min | elapsed {(now-t0)/60:.1f}m | ETA {eta/60:.1f}m", flush=True)
            last = now
        time.sleep(2)

    npub = len(list(shared.glob(f"hl_clusters_{node_idx}_*.csv")))
    # publish a per-node completion marker so finalize can verify the WHOLE grid was searched
    (shared / f"_node_{node_idx}.done").write_text(json.dumps(
        dict(node_idx=node_idx, nnode=nnode, total=total, fail=fail, npub=npub)))
    print(f"[shard {tag} node {node_idx}] DONE: {done} chunks, {fail} failed, {npub} published, "
          f"{(time.time()-t0)/60:.1f}m", flush=True)
    return fail


def finalize_link_refine(rd: Path, bin_dir: Path, *, maxrms: float = 100000.0):
    """link_refine over every published shard in <rd>/clusters_mn -> lr.csv / lr_rms.csv (in rd)."""
    shared = rd / "clusters_mn"
    # Integrity check: every node must have published a completion marker with zero failed chunks,
    # else part of the hypothesis grid was never searched and link_refine would silently report a
    # partial result as complete (losing any cluster whose grid point fell in a dead node/shard).
    markers = [json.loads(m.read_text()) for m in shared.glob("_node_*.done")]
    if markers:
        nnode = max(m["nnode"] for m in markers)
        missing = set(range(nnode)) - {m["node_idx"] for m in markers}
        tot_fail = sum(m["fail"] for m in markers)
        if missing:
            raise SystemExit(f"[finalize] ABORT: nodes {sorted(missing)}/{nnode} never finished -> "
                             "partial grid; refusing to report an incomplete result")
        if tot_fail:
            raise SystemExit(f"[finalize] ABORT: {tot_fail} heliolinc chunks failed across nodes -> "
                             "partial grid; refusing to report an incomplete result")
    lf = []
    for c in sorted(shared.glob("hl_clusters_*.csv")):
        s = c.name[len("hl_clusters_"):-len(".csv")]
        summ = shared / f"hl_summary_{s}.csv"
        if c.stat().st_size > 0 and summ.exists() and summ.stat().st_size > 0:
            lf.append(f"clusters_mn/{c.name} clusters_mn/{summ.name}")
    (rd / "lflist.txt").write_text("\n".join(lf) + "\n")
    print(f"[finalize] link_refine over {len(lf)} shard files", flush=True)
    subprocess.run([str(bin_dir / "link_refine"), "-pairdet", "pairdets.csv", "-lflist", "lflist.txt",
                    "-maxrms", str(maxrms), "-outfile", "lr.csv", "-outrms", "lr_rms.csv"],
                   cwd=rd, capture_output=True, text=True)
    return len(lf)
