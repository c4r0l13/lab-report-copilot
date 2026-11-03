import random
from types import SimpleNamespace

import pandas as pd
import pytest

from labcopilot import data, evaluate, render
from labcopilot.extract import LLMExtractor, TemplateExtractor, pdf_text
from labcopilot.validate import validate

STEPS = [
    {"current_density_mA_cm2": j, "cell_voltage_V": v, "fe_h2": 40.0, "fe_co": 35.5, "fe_ch4": 0.03,
     "fe_c2h4": 0.0, "fe_gas_total": 75.53, "fe_liquid": 24.47}
    for j, v in zip(data.CO2R_STEPS, [3.0, 3.2, 3.4, 3.5, 3.7], strict=True)
]  # fmt: skip
REC = {"sample_id": "vsp1_240101_Ag-0.25-Au-0.75_pp0", "replicate": 1, "composition": {"Ag": 0.25, "Au": 0.75},
       "synthesis_route": "vsp", "batch_date": "2024-01-01", "reaction": "CO2R", "steps": STEPS}  # fmt: skip
MASTER = data.master_data([REC])


def test_raw_table_keeps_only_replicates():
    raw = pd.DataFrame({"rep": ["1", "2", "avg", "std"], "reaction": ["CO2R"] * 4})
    assert list(data.measurements(raw)["rep"]) == [1, 2]


def test_clean_record_passes_and_each_error_kind_is_caught():
    assert validate(REC, MASTER) == []
    rng = random.Random(0)
    expected = {"decimal_shift": "voltage_range", "digit_swap": "gas_balance", "id_typo": "unknown_sample",
                "missing_step": "incomplete_protocol", "composition_swap": "composition_mismatch"}  # fmt: skip
    for kind, rule in expected.items():
        bad = render.inject_error(REC, kind, rng)
        assert rule in {i.rule for i in validate(bad, MASTER)}, kind


def test_fe_over_unity_is_flagged():
    rec = {**REC, "steps": [dict(s) for s in STEPS]}
    rec["steps"][0]["fe_liquid"] = 30.0
    assert "fe_over_unity" in {i.rule for i in validate(rec, MASTER)}


@pytest.mark.parametrize("layout", ["table_en", "table_es"])
def test_template_round_trip(tmp_path, layout):
    path = tmp_path / f"{layout}.pdf"
    render.render(REC, layout, path)
    got = TemplateExtractor()(pdf_text(path))
    assert evaluate.field_diffs(got, REC) == []


def test_template_cannot_read_instrument_export(tmp_path):
    path = tmp_path / "x.pdf"
    render.render(REC, "instrument_export", path)
    got = TemplateExtractor()(pdf_text(path))
    assert got["steps"] == [] and "steps.count" in evaluate.field_diffs(got, REC)


class FakeClient:
    """Mimics anthropic.Anthropic().messages.create for tool-forced calls."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        block = SimpleNamespace(type="tool_use", id=f"t{len(self.calls)}", input=self.outputs.pop(0))
        return SimpleNamespace(content=[block])


def test_llm_extractor_repairs_after_failed_checks_and_caches(tmp_path):
    misread = {**REC, "steps": [dict(s) for s in STEPS]}
    misread["steps"][2]["cell_voltage_V"] = 34.0  # e.g. mV read as V/100
    llm = LLMExtractor(model="fake", cache_dir=tmp_path)
    llm._client = FakeClient([misread, REC])
    out = llm.extract("d1", "text", validate_fn=lambda r: validate(r, MASTER))
    assert out["repaired"] and out["final"] == REC
    assert len(llm._client.calls) == 2
    assert llm._client.calls[1]["messages"][-1]["content"][0]["type"] == "tool_result"
    llm._client = FakeClient([])  # cached: no further calls
    assert llm.extract("d1", "text")["final"] == REC
