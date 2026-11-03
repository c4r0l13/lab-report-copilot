"""Real electrochemical test results (OCx24) turned into one record per sample replicate.

The raw export mixes three kinds of rows: individual replicates ("1", "2", "3") and two summary rows per
sample ("avg" and "std"). Only replicate rows are measurements.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
FE_COLS = ["fe_h2", "fe_co", "fe_ch4", "fe_c2h4", "fe_gas_total", "fe_liquid"]
CO2R_STEPS = [50.0, 100.0, 150.0, 200.0, 300.0]  # mA/cm2, the test protocol


def load_raw(path: Path = RAW / "ExpDataDump_241113.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def measurements(raw: pd.DataFrame) -> pd.DataFrame:
    """Replicate rows only (drops the avg/std summary rows that share the table)."""
    m = raw[raw["rep"].astype(str).str.fullmatch(r"\d+")].copy()
    m["rep"] = m["rep"].astype(int)
    return m


def parse_composition(text: str) -> dict[str, float]:
    tokens = text.split("-")
    return {el: float(x) for el, x in zip(tokens[::2], tokens[1::2], strict=True)}


def format_date(yymmdd: int | float) -> str:
    s = f"{int(yymmdd):06d}"
    return f"20{s[:2]}-{s[2:4]}-{s[4:]}"


def base_sample_id(sample_id: str) -> str:
    """'vsp13_240823_Ag-0.5-Au-0.5_pp0_rep1' -> 'vsp13_240823_Ag-0.5-Au-0.5_pp0'."""
    return sample_id.split("_rep")[0]


def records(raw: pd.DataFrame, reaction: str = "CO2R") -> list[dict]:
    """One dict per (sample, replicate) with the measurements sorted by current density."""
    m = measurements(raw)
    m = m[m["reaction"] == reaction]
    out = []
    for (sid, rep), g in m.groupby(["sample id", "rep"], sort=True):
        g = g.sort_values("current density")
        out.append(
            {
                "sample_id": base_sample_id(sid),
                "replicate": int(rep),
                "composition": parse_composition(g["composition"].iloc[0]),
                "synthesis_route": g["source"].iloc[0],
                "batch_date": format_date(g["batch date"].iloc[0]),
                "reaction": reaction,
                "steps": [
                    {
                        "current_density_mA_cm2": float(r["current density"]),
                        "cell_voltage_V": None if pd.isna(r["voltage"]) else round(float(r["voltage"]), 3),
                        **{c: round(float(r[c]), 2) for c in FE_COLS},
                    }
                    for _, r in g.iterrows()
                ],
            }
        )
    return out


def master_data(recs: list[dict]) -> dict[str, dict]:
    """What a LIMS knows before any result arrives: registered samples and their nominal composition."""
    return {
        r["sample_id"]: {
            "composition": r["composition"],
            "synthesis_route": r["synthesis_route"],
        }
        for r in recs
    }


def table_audit(raw: pd.DataFrame) -> dict:
    """Data-quality facts about the raw export itself."""
    rep = raw["rep"].astype(str)
    m = measurements(raw)
    co2r = m[m["reaction"] == "CO2R"]
    total = co2r["fe_gas_total"] + co2r["fe_liquid"]
    naive = raw.loc[raw["reaction"] == "CO2R", "voltage"].mean()
    return {
        "rows": len(raw),
        "summary_rows_avg": int((rep == "avg").sum()),
        "summary_rows_std": int((rep == "std").sum()),
        "measurement_rows": len(m),
        "co2r_measurements": len(co2r),
        "fe_total_above_100_5": int((total > 100.5).sum()),
        "fe_liquid_negative": int((co2r["fe_liquid"] < 0).sum()),
        "fe_h2_above_100": int((co2r["fe_h2"] > 100).sum()),
        "mean_co2r_voltage_naive": round(float(naive), 3),
        "mean_co2r_voltage_measurements": round(float(co2r["voltage"].mean()), 3),
    }
