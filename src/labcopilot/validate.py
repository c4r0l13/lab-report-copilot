"""LIMS-style validation rules for a CO2 reduction test record.

Every rule is a check a lab supervisor would apply by eye. Each returns zero or more issues; a record with no
issues can be accepted automatically ("straight-through"), anything else goes to a person.
"""

from __future__ import annotations

from dataclasses import dataclass

from .data import CO2R_STEPS

GAS = ["fe_h2", "fe_co", "fe_ch4", "fe_c2h4"]


@dataclass(frozen=True)
class Issue:
    rule: str
    detail: str


def _close(a: dict[str, float], b: dict[str, float], tol: float = 0.02) -> bool:
    return set(a) == set(b) and all(abs(a[k] - b[k]) <= tol for k in a)


def validate(rec: dict, master: dict[str, dict]) -> list[Issue]:
    issues: list[Issue] = []
    sid = rec.get("sample_id")
    reg = master.get(sid)
    if reg is None:
        issues.append(Issue("unknown_sample", f"{sid!r} is not registered in the LIMS"))
    comp = rec.get("composition") or {}
    if comp and abs(sum(comp.values()) - 1) > 0.02:
        issues.append(Issue("composition_sum", f"fractions sum to {sum(comp.values()):.3f}"))
    if reg is not None and comp and not _close(comp, reg["composition"]):
        issues.append(
            Issue(
                "composition_mismatch",
                "reported composition differs from the registered sample",
            )
        )

    steps = rec.get("steps") or []
    densities = sorted(s.get("current_density_mA_cm2") for s in steps if s.get("current_density_mA_cm2") is not None)
    if densities != CO2R_STEPS:
        issues.append(
            Issue(
                "incomplete_protocol",
                f"current densities {densities}, expected {CO2R_STEPS}",
            )
        )
    for s in steps:
        j = s.get("current_density_mA_cm2")
        v = s.get("cell_voltage_V")
        if v is None or not 1.5 <= v <= 5.0:
            issues.append(Issue("voltage_range", f"{v} V at {j} mA/cm2 is outside 1.5-5.0 V"))
        fes = [s.get(c) for c in [*GAS, "fe_gas_total", "fe_liquid"]]
        if any(x is None for x in fes):
            issues.append(Issue("missing_value", f"missing Faradaic efficiency at {j} mA/cm2"))
            continue
        gas_sum = sum(s[c] for c in GAS)
        if abs(gas_sum - s["fe_gas_total"]) > 1.0:
            issues.append(
                Issue(
                    "gas_balance",
                    f"gas products sum to {gas_sum:.1f}% but total is {s['fe_gas_total']}%",
                )
            )
        total = s["fe_gas_total"] + s["fe_liquid"]
        if total > 100.5 or s["fe_liquid"] < 0:
            issues.append(
                Issue(
                    "fe_over_unity",
                    f"total Faradaic efficiency {total:.1f}% at {j} mA/cm2",
                )
            )
    ordered = [s for s in sorted(steps, key=lambda s: s.get("current_density_mA_cm2") or 0)]
    volts = [s.get("cell_voltage_V") for s in ordered]
    if all(v is not None for v in volts) and any(b < a - 0.05 for a, b in zip(volts, volts[1:], strict=False)):
        issues.append(
            Issue(
                "voltage_not_monotonic",
                "cell voltage drops as current density increases",
            )
        )
    return issues
