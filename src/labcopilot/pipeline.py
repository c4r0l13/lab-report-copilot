"""Build the report corpus, run the extractors and the validator, write reports/.

Run: python -m labcopilot.pipeline            (template extractor only, no API key needed)
     ANTHROPIC_API_KEY=... python -m labcopilot.pipeline --llm   (adds Claude; ~150 calls, cached)
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from . import data, evaluate, render, style
from .extract import LLMExtractor, TemplateExtractor, pdf_text
from .validate import validate

OUT = Path("reports")
DOCS = Path("data/reports")
N_DOCS = 150
ERROR_RATE = 0.35
SEED = 7


def build_corpus(recs: list[dict]) -> list[dict]:
    """Pick records, assign a format, inject a transcription error in ~35% of them, render PDFs."""
    rng = random.Random(SEED)
    DOCS.mkdir(parents=True, exist_ok=True)
    corpus = []
    for i, rec in enumerate(rng.sample(recs, N_DOCS)):
        layout = render.LAYOUTS[i % len(render.LAYOUTS)]
        error = rng.choice(render.ERRORS) if rng.random() < ERROR_RATE else None
        printed = render.inject_error(rec, error, rng) if error else rec
        doc_id = f"doc{i:03d}_{layout}"
        path = DOCS / f"{doc_id}.pdf"
        if not path.exists():
            render.render(printed, layout, path)
        corpus.append(
            {
                "doc_id": doc_id,
                "layout": layout,
                "error": error,
                "truth": rec,
                "printed": printed,
                "path": path,
            }
        )
    return corpus


def run_extractor(name: str, extractor, corpus: list[dict], master: dict) -> pd.DataFrame:
    rows = []
    for doc in corpus:
        text = pdf_text(doc["path"])
        if isinstance(extractor, LLMExtractor):
            out = extractor.extract(doc["doc_id"], text, validate_fn=lambda r: validate(r, master))
            got, first, repaired = out["final"], out["first"], out["repaired"]
        else:
            got = first = extractor(text)
            repaired = False
        issues = validate(got, master)
        misread = evaluate.field_diffs(got, doc["printed"])
        wrong_vs_truth = evaluate.field_diffs(got, doc["truth"])
        rows.append(
            {
                "extractor": name,
                "doc_id": doc["doc_id"],
                "layout": doc["layout"],
                "injected_error": doc["error"] or "none",
                "misread_fields": len(misread),
                "misread_fields_first_pass": len(evaluate.field_diffs(first, doc["printed"])),
                "field_accuracy": 1 - len(misread) / evaluate.n_fields(doc["printed"]),
                "record_exact": len(misread) == 0,
                "flagged": bool(issues),
                "rules": ";".join(sorted({i.rule for i in issues})),
                "auto_accepted": not issues,
                "wrong_data_accepted": (not issues) and bool(wrong_vs_truth),
                "repaired": repaired,
            }
        )
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, source_issue_docs: set[str]) -> dict:
    inj = df[df["injected_error"] != "none"]
    clean = df[(df["injected_error"] == "none") & ~df["doc_id"].isin(source_issue_docs)]
    return {
        "record_exact_pct": round(100 * df["record_exact"].mean(), 1),
        "field_accuracy_pct": round(100 * df["field_accuracy"].mean(), 2),
        "record_exact_by_layout": (100 * df.groupby("layout")["record_exact"].mean()).round(1).to_dict(),
        "injected_errors_caught_pct": round(100 * inj["flagged"].mean(), 1),
        "injected_caught_by_kind": (100 * inj.groupby("injected_error")["flagged"].mean()).round(1).to_dict(),
        "false_alarm_pct_on_clean_docs": round(100 * clean["flagged"].mean(), 1),
        "straight_through_pct": round(100 * df["auto_accepted"].mean(), 1),
        "wrong_records_accepted": int(df["wrong_data_accepted"].sum()),
        "repair_turns": int(df["repaired"].sum()),
    }


def fig_accuracy(results: pd.DataFrame, figs: Path) -> None:
    piv = (100 * results.groupby(["layout", "extractor"])["record_exact"].mean()).unstack()
    piv = piv.reindex(render.LAYOUTS)
    labels = {
        "table_en": "Table, English",
        "table_es": "Table, Spanish, A/m²",
        "instrument_export": "Instrument export",
    }
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    width = 0.8 / len(piv.columns)
    for k, col in enumerate(piv.columns):
        xs = [i + (k - (len(piv.columns) - 1) / 2) * width for i in range(len(piv))]
        label = "Template parser" if col == "template" else col.replace("llm:", "Claude ")
        ax.bar(xs, piv[col], width, label=label, color=style.SERIES[k])
        for x, v in zip(xs, piv[col], strict=True):
            ax.text(x, v + 1.5, f"{v:.0f}%", ha="center", fontsize=8.5, color=style.INK_2)
    ax.set_xticks(range(len(piv)), [labels[i] for i in piv.index])
    ax.set_ylim(0, 112)
    ax.set_ylabel("Reports transcribed with no error (%)")
    ax.set_title("Templates break on a new format")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(axis="x", visible=False)
    fig.savefig(figs / "extraction_accuracy.png")
    plt.close(fig)


def fig_caught(oracle: pd.DataFrame, figs: Path) -> None:
    inj = oracle[oracle["injected_error"] != "none"]
    rate = (100 * inj.groupby("injected_error")["flagged"].mean()).sort_values()
    n = inj["injected_error"].value_counts()
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    ax.barh(
        [f"{k.replace('_', ' ')} (n={n[k]})" for k in rate.index],
        rate,
        color=style.SERIES[2],
    )
    for i, v in enumerate(rate):
        ax.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=9, color=style.INK_2)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Injected errors flagged by the validation rules (%)")
    ax.set_title("Which transcription errors the LIMS rules catch")
    ax.grid(axis="y", visible=False)
    fig.savefig(figs / "errors_caught.png")
    plt.close(fig)


def main(use_llm: bool = False) -> None:
    style.apply_style()
    figs = OUT / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    raw = data.load_raw()
    audit = data.table_audit(raw)
    recs = data.records(raw)
    master = data.master_data(recs)
    real_issues = Counter(i.rule for r in recs for i in validate(r, master))

    corpus = build_corpus(recs)
    source_issue_docs = {d["doc_id"] for d in corpus if validate(d["truth"], master)}
    extractors = {"template": TemplateExtractor()}
    llm_cache = Path("reports/llm_outputs")
    if use_llm or (llm_cache.exists() and any(llm_cache.rglob("*.json"))):
        llm = LLMExtractor()
        extractors[llm.name] = llm

    # "Oracle": a perfect transcription of what is printed. Measures the rules on their own.
    oracle = pd.DataFrame(
        [
            {"doc_id": d["doc_id"], "injected_error": d["error"] or "none",
             "flagged": bool(validate(d["printed"], master))}
            for d in corpus
        ]
    )  # fmt: skip
    results = pd.concat(
        [run_extractor(n, e, corpus, master) for n, e in extractors.items()],
        ignore_index=True,
    )
    results.to_csv(OUT / "per_document.csv", index=False)
    fig_accuracy(results, figs)
    fig_caught(oracle, figs)

    clean = oracle[(oracle["injected_error"] == "none") & ~oracle["doc_id"].isin(source_issue_docs)]
    summary = {
        "raw_table_audit": audit,
        "real_records": len(recs),
        "real_records_flagged": sum(1 for r in recs if validate(r, master)),
        "real_issue_counts": dict(real_issues),
        "corpus": {
            "documents": len(corpus),
            "by_layout": dict(Counter(d["layout"] for d in corpus)),
            "injected_errors": dict(Counter(d["error"] for d in corpus if d["error"])),
            "documents_with_real_source_issues": len(source_issue_docs),
        },
        "rules_on_perfect_transcription": {
            "injected_errors_caught_pct": round(100 * oracle.loc[oracle.injected_error != "none", "flagged"].mean(), 1),
            "false_alarm_pct_on_clean_docs": round(100 * clean["flagged"].mean(), 1),
        },
        "extractors": {n: summarise(g, source_issue_docs) for n, g in results.groupby("extractor")},
    }
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="call Claude for documents not cached yet")
    main(ap.parse_args().llm)
