"""The cutout-cache identity hash: ONE implementation, four users.

`alert_cutouts` WRITES it, `select_clean` REWRITES it after its cut, and `alert_sheets` +
`night_status` VERIFY it. It existed as four hand-copied loops; they must agree exactly or every
cache reads as a mismatch and no night can render or certify. It has already diverged once (a
prefix-vs-whole-file disagreement made every over-limit night unrenderable).

The hash covers (visit, detector, ra, dec) per epoch, in file order. Position is in it because a
detector-only signature makes two alerts on the SAME panel pair interchangeable -- MEASURED,
8.8-25.2% of DELIVERED alerts share an epoch signature with at least one other -- so a swap between
them permuted the cache without changing the hash, which is the exact failure the fingerprint
exists to catch. Rounded to 1e-4 deg (~0.36") to keep float formatting out of the identity.

VERSIONED. Changing what is hashed invalidates every cache written by the previous version, and a
verifier that cannot tell "old format" from "wrong pixels" reports a false CACHE MISMATCH on a
perfectly good night. Metadata without a matching `fingerprint_version` is UNCHECKABLE, not wrong.
"""
from __future__ import annotations
import hashlib
import json

FINGERPRINT_VERSION = 2          # 1 = (visit, detector); 2 = (visit, detector, ra, dec)


def epoch_digest(alerts, cap=None):
    """sha256 over the (visit, detector, ra, dec) sequence of `alerts[:cap]`.

    `alerts` is an iterable of alert dicts OR an open path; `cap` bounds it to the prefix the cache
    was actually built over (renderers take --limit, so the cache covers a prefix, not the file).
    """
    if isinstance(alerts, (str, bytes)) or hasattr(alerts, "__fspath__"):
        with open(alerts) as f:
            return epoch_digest([json.loads(l) for l in f], cap)
    h = hashlib.sha256()
    for i, a in enumerate(alerts):
        if isinstance(cap, int) and i >= cap:
            break
        for e in (a.get("epochs") or []):
            h.update(f"{e.get('visit',-1)}:{e.get('detector',-1)}:"
                     f"{round(float(e.get('ra') or 0.0), 4)}:"
                     f"{round(float(e.get('dec') or 0.0), 4)};".encode())
        h.update(b"|")
    return h.hexdigest()


def verify(alerts_path, meta):
    """-> (checked, ok). `checked` is False when there is nothing to verify against: no metadata,
    no fingerprint, or one written by an older FINGERPRINT_VERSION. Callers must treat
    checked=False as "unknown", never as failure -- a cache is deleted after a successful night,
    so its absence is the normal state."""
    if not isinstance(meta, dict):
        return False, True
    fp = meta.get("alerts_fingerprint")
    if not fp or meta.get("fingerprint_version") != FINGERPRINT_VERSION:
        return False, True
    return True, epoch_digest(alerts_path, meta.get("n_alerts")) == fp
