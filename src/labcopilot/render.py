"""Render records as PDF test reports in three formats, optionally with a realistic transcription error.

The numbers come from real OCx24 measurements; the report formats are generated here to mimic what a LIMS
receives from different labs: an English table, a Spanish table in SI units, and a raw instrument export.
"""

from __future__ import annotations

import copy
import random
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

LAYOUTS = ["table_en", "table_es", "instrument_export"]
ERRORS = [
    "decimal_shift",
    "digit_swap",
    "id_typo",
    "missing_step",
    "composition_swap",
    "voltage_typo",
    "date_typo",
]
ROUTES_EN = {"vsp": "spark ablation", "uoft": "chemical reduction"}
ROUTES_ES = {"vsp": "ablación por chispa", "uoft": "reducción química"}


def comp_text(comp: dict[str, float]) -> str:
    return ", ".join(f"{el} {x:g}" for el, x in comp.items())


def inject_error(rec: dict, kind: str, rng: random.Random) -> dict:
    """Return a copy of rec with one transcription error of the given kind (what ends up printed)."""
    r = copy.deepcopy(rec)
    steps = r["steps"]
    if kind == "decimal_shift":
        s = rng.choice([s for s in steps if s["cell_voltage_V"] is not None] or steps)
        s["cell_voltage_V"] = round((s["cell_voltage_V"] or 0.33) * 10, 2)
    elif kind == "digit_swap":
        candidates = [
            s for s in steps if s["fe_co"] >= 10 and str(f"{s['fe_co']:.2f}")[0] != str(f"{s['fe_co']:.2f}")[1]
        ]
        s = rng.choice(candidates or steps)
        t = f"{s['fe_co']:.2f}"
        s["fe_co"] = float(t[1] + t[0] + t[2:]) if t[1] != "." else s["fe_co"] + 30
    elif kind == "id_typo":
        sid = r["sample_id"]
        i = rng.randrange(len(sid))
        while not sid[i].isdigit():
            i = rng.randrange(len(sid))
        r["sample_id"] = sid[:i] + str((int(sid[i]) + rng.randint(1, 8)) % 10) + sid[i + 1 :]
    elif kind == "missing_step":
        steps.pop(rng.randrange(len(steps)))
    elif kind == "composition_swap":
        els = list(r["composition"])
        swapped = dict(zip(els, list(r["composition"].values())[::-1], strict=True))
        if swapped != r["composition"]:
            r["composition"] = swapped
        else:  # symmetric composition: report a different nominal ratio instead
            r["composition"] = {
                e: round(x * (1.2 if k == 0 else 0.8), 3) for k, (e, x) in enumerate(r["composition"].items())
            }
    elif kind == "voltage_typo":  # last two digits swapped: plausible value, still increasing
        s = rng.choice([s for s in steps if s["cell_voltage_V"] is not None] or steps)
        t = f"{s['cell_voltage_V'] or 3.141:.3f}"
        s["cell_voltage_V"] = float(t[:-2] + t[-1] + t[-2])
    elif kind == "date_typo":
        y, m, d = r["batch_date"].split("-")
        r["batch_date"] = f"{y}-{m}-{(int(d) % 28) + 1:02d}"
    return r


def _table_pdf(rec: dict, path: Path, spanish: bool) -> None:
    styles = getSampleStyleSheet()
    if spanish:
        title = "Informe de prueba: reducción electroquímica de CO2"
        meta = [
            f"Muestra: {rec['sample_id']}",
            f"Composición nominal: {comp_text(rec['composition'])}",
            f"Ruta de síntesis: {ROUTES_ES[rec['synthesis_route']]}",
            f"Fecha de lote: {rec['batch_date'][8:10]}/{rec['batch_date'][5:7]}/{rec['batch_date'][:4]}",
            f"Réplica: {rec['replicate']}",
        ]
        header = [
            "Densidad (A/m²)",
            "Voltaje (V)",
            "H2 %",
            "CO %",
            "CH4 %",
            "C2H4 %",
            "Gas %",
            "Líquido %",
        ]
        scale = 10.0  # 1 mA/cm2 = 10 A/m2
    else:
        title = "Test report: CO2 electroreduction"
        meta = [
            f"Sample ID: {rec['sample_id']}",
            f"Composition: {comp_text(rec['composition'])}",
            f"Synthesis route: {ROUTES_EN[rec['synthesis_route']]}",
            f"Batch date: {rec['batch_date']}",
            f"Replicate: {rec['replicate']}",
        ]
        header = [
            "j (mA/cm²)",
            "Cell V",
            "FE H2 %",
            "FE CO %",
            "FE CH4 %",
            "FE C2H4 %",
            "FE gas %",
            "FE liquid %",
        ]
        scale = 1.0
    rows = [header] + [
        [
            f"{s['current_density_mA_cm2'] * scale:g}",
            "n/a" if s["cell_voltage_V"] is None else f"{s['cell_voltage_V']:.3f}",
            *[
                f"{s[c]:.2f}"
                for c in [
                    "fe_h2",
                    "fe_co",
                    "fe_ch4",
                    "fe_c2h4",
                    "fe_gas_total",
                    "fe_liquid",
                ]
            ],
        ]
        for s in rec["steps"]
    ]
    table = Table(rows)
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story = [Paragraph(title, styles["Title"])] + [Paragraph(m, styles["Normal"]) for m in meta]
    story += [Spacer(1, 12), table]
    SimpleDocTemplate(str(path), pagesize=A4).build(story)


def _export_pdf(rec: dict, path: Path) -> None:
    """Instrument-style export: key=value lines, voltage in mV, Faradaic efficiencies as fractions."""
    styles = getSampleStyleSheet()
    lines = [
        "# POTENTIOSTAT EXPORT v2.3 / GC-HPLC MERGE",
        f"SAMPLE={rec['sample_id']}  REP={rec['replicate']}",
        f"MATERIAL={'-'.join(f'{e}{x:g}' for e, x in rec['composition'].items())}",
        f"ROUTE={rec['synthesis_route'].upper()}  LOT_DATE={rec['batch_date'].replace('-', '')}",
    ]
    for i, s in enumerate(rec["steps"], start=1):
        fe = " ".join(f"FE[{k}]={s['fe_' + k.lower()] / 100:.4f}" for k in ["H2", "CO", "CH4", "C2H4"])
        volt = "NA" if s["cell_voltage_V"] is None else f"{s['cell_voltage_V'] * 1000:.0f} mV"
        lines.append(
            f"STEP {i}: I={s['current_density_mA_cm2']:g} mA/cm2; E_cell={volt}; "
            f"{fe} FE[GAS]={s['fe_gas_total'] / 100:.4f} FE[LIQ]={s['fe_liquid'] / 100:.4f}"
        )
    story = [Paragraph(line, styles["Code"]) for line in lines]
    SimpleDocTemplate(str(path), pagesize=A4).build(story)


def render(rec: dict, layout: str, path: Path) -> None:
    if layout == "instrument_export":
        _export_pdf(rec, path)
    else:
        _table_pdf(rec, path, spanish=layout == "table_es")
