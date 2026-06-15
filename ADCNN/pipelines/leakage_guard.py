"""Fail-loud blind/test leakage guard for the ADCNN training workflow.

The blind/test fields must NEVER appear in training inputs. tract-disjoint is NOT enough:
adjacent tracts observed on the same night share boundary-CCD exposures, so a tract-level
split can still leak individual ``(visit, detector)`` panels into the blind set (this happened
in rc1 — 12 panels leaked into 2 of 26 blind fields; non-inflating, but it must not recur).
The correct unit of disjointness is the EXPOSURE: the ``(visit, detector)`` pair.

Use :func:`assert_disjoint` in any data-build / training driver before consuming a manifest.
It raises :class:`LeakageError` (loud, non-silent) on any overlap.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Set, Tuple, Union

import pandas as pd

PathLike = Union[str, Path]
Pair = Tuple[int, int]


class LeakageError(RuntimeError):
    """Raised when blind/test exposures appear in training inputs."""


def visit_detector_pairs(csvs: Union[PathLike, Iterable[PathLike]]) -> Set[Pair]:
    """Collect the set of ``(visit, detector)`` exposures across one or more CSV manifests.

    Each CSV must carry ``visit`` and ``detector`` columns (the manifest schema). Fails loud
    on a missing column or unreadable file rather than silently returning an empty set.
    """
    if isinstance(csvs, (str, Path)):
        csvs = [csvs]
    pairs: Set[Pair] = set()
    for c in csvs:
        p = Path(c)
        if not p.exists():
            raise FileNotFoundError(f"leakage guard: manifest not found: {p}")
        df = pd.read_csv(p)
        missing = {"visit", "detector"} - set(df.columns)
        if missing:
            raise ValueError(f"leakage guard: {p} missing column(s) {sorted(missing)} "
                             f"(need visit+detector to check exposure-level disjointness)")
        pairs.update(zip(df["visit"].astype(int), df["detector"].astype(int)))
    return pairs


def assert_disjoint(train_csvs: Union[PathLike, Sequence[PathLike]],
                    blind_csvs: Union[PathLike, Sequence[PathLike]],
                    *, label: str = "train vs blind/test") -> Tuple[int, int]:
    """Raise :class:`LeakageError` if any ``(visit, detector)`` is shared between the two sets.

    Returns ``(n_train_exposures, n_blind_exposures)`` on success. This is the gate the data /
    training stages call so a leaked exposure stops the run instead of contaminating a model.
    """
    train = visit_detector_pairs(train_csvs)
    blind = visit_detector_pairs(blind_csvs)
    overlap = sorted(train & blind)
    if overlap:
        ex = ", ".join(f"(visit={v},det={d})" for v, d in overlap[:5])
        more = "" if len(overlap) <= 5 else f" (+{len(overlap) - 5} more)"
        raise LeakageError(
            f"BLIND LEAKAGE [{label}]: {len(overlap)} (visit,detector) exposure(s) appear in BOTH "
            f"training and blind/test inputs. Enforce exposure-level (not tract-level) exclusion. "
            f"Examples: {ex}{more}")
    return len(train), len(blind)
