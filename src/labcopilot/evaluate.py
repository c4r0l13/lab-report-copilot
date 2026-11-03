"""Compare extracted records with the truth, field by field."""

from __future__ import annotations

from .extract import FE_KEYS

META = ["sample_id", "replicate", "synthesis_route", "batch_date"]
STEP_KEYS = ["current_density_mA_cm2", "cell_voltage_V", *FE_KEYS]


def _num_eq(a, b, tol: float) -> bool:
    if a is None and b is None:
        return True
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def field_diffs(got: dict, want: dict) -> list[str]:
    """Names of fields where `got` differs from `want` (numbers to transcription precision)."""
    bad = [k for k in META if str(got.get(k)) != str(want.get(k))]
    gc, wc = got.get("composition") or {}, want.get("composition") or {}
    if set(gc) != set(wc) or any(not _num_eq(gc[e], wc[e], 0.0015) for e in wc):
        bad.append("composition")
    gs = sorted(got.get("steps") or [], key=lambda s: s.get("current_density_mA_cm2") or 0)
    ws = sorted(want.get("steps") or [], key=lambda s: s.get("current_density_mA_cm2") or 0)
    if len(gs) != len(ws):
        bad.append("steps.count")
    for i, (g, w) in enumerate(zip(gs, ws, strict=False)):
        for k in STEP_KEYS:
            tol = 0.0015 if k == "cell_voltage_V" else 0.011
            if not _num_eq(g.get(k), w[k], tol):
                bad.append(f"steps[{i}].{k}")
    return bad


def n_fields(rec: dict) -> int:
    return len(META) + 1 + 1 + len(rec["steps"]) * len(STEP_KEYS)
