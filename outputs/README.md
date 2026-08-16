# outputs/

Only the CURRENT delivered products and the truth sets that validate them live here. Everything
else is regenerable; the stale experiment tree (attic/, run_ring_ft, ringpipe_0706, ep6_gate,
merge_0706, 1k_cadence, threshold_sweep, timed_0705, pa_precision) was deleted 2026-08-16.

    runs/10k_cadence/run_night_<night>/        THE PRODUCT, nine embargo nights
        stream_1k/alerts.jsonl                 delivered ~1k alerts (record of truth, rank-ordered)
        stream_1k/alerts.csv                   the same, flat, one row per alert (ADCNN.qa.alerts_csv)
        stream_1k/pairs/alert_NNNNN_*.png      one pair image per delivered alert, rank in the name
        stream_1k/sheets/index.html            contact sheets for visual scanning
        stream/alerts.jsonl                    the full low-threshold stream behind the 1k cut
        alerts.jsonl + report/                 frozen science-alert product + QA overlays
        adcnn_dets*.csv, dets_merged.csv       detection + merged catalogues (GPU/Butler cost to rebuild)
        manifest.csv, known.csv, *.parquet     inputs of record

    runs/pa_validate/                          the truth sets the op point and gates rest on
        truth_v2/v3 + inj_dets_*                2-epoch injections (merge + op-point basis)
        truth_n20260713 + inj_dets_*            HELD-OUT night (never tunes anything)
        truth_tri_n2026070[6|13] + inj_dets_*   3-epoch injections (the 3+visit gate campaign)

    logs/            per-night campaign logs (the audit trail)
    deliverables/    exported reports
    query_snapshots/ Butler query provenance
    training_runs/   model training history (deployed models live in models/)

REGENERABLE, so deleted and not kept: cutout caches (stream*/cutouts.npz, ~25 min/night),
filter intermediates (surv/topk.jsonl), detection shards (_shard_*, redundant once
adcnn_dets.csv exists), pixel_vet backups (alerts_prevet.jsonl).

Verify a night before trusting it:  python -m ADCNN.pipelines.night_status outputs/runs/10k_cadence/run_night_<night>
