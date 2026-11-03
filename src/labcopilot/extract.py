"""Turn report PDFs into records: a hand-written template parser and an LLM extractor with a repair step."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from pypdf import PdfReader

FE_KEYS = ["fe_h2", "fe_co", "fe_ch4", "fe_c2h4", "fe_gas_total", "fe_liquid"]
ROUTES = {"spark ablation": "vsp", "chemical reduction": "uoft", "ablación por chispa": "vsp",
          "reducción química": "uoft", "VSP": "vsp", "UOFT": "uoft"}  # fmt: skip


def pdf_text(path: str | Path) -> str:
    return "\n".join(page.extract_text() for page in PdfReader(str(path)).pages)


def _comp(text: str) -> dict[str, float]:
    return {el: float(x) for el, x in re.findall(r"([A-Z][a-z]?)\s*([\d.]+)", text)}


class TemplateExtractor:
    """Regex templates written for the two table formats. Anything else is returned (mostly) empty,
    which is what happens in a real lab when a new instrument or partner sends a new format."""

    name = "template"

    def __call__(self, text: str) -> dict:
        rec: dict = {"steps": []}
        es = "Muestra:" in text
        m = re.search(r"(?:Sample ID|Muestra):\s*(\S+)", text)
        rec["sample_id"] = m.group(1) if m else None
        m = re.search(r"(?:Composition|Composición nominal):\s*(.+)", text)
        rec["composition"] = _comp(m.group(1)) if m else None
        m = re.search(r"(?:Synthesis route|Ruta de síntesis):\s*(.+)", text)
        rec["synthesis_route"] = ROUTES.get(m.group(1).strip()) if m else None
        m = re.search(r"(?:Replicate|Réplica):\s*(\d+)", text)
        rec["replicate"] = int(m.group(1)) if m else None
        m = re.search(r"Batch date:\s*(\d{4}-\d{2}-\d{2})", text) or re.search(
            r"Fecha de lote:\s*(\d{2})/(\d{2})/(\d{4})", text
        )
        if m and m.lastindex == 3:
            rec["batch_date"] = f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
        else:
            rec["batch_date"] = m.group(1) if m else None
        head = "Líquido %" if es else "FE liquid %"
        if head in text:
            nums = re.findall(r"^(?:-?[\d.]+|n/a)$", text.split(head, 1)[1], flags=re.M)
            for i in range(0, len(nums) - 7, 8):
                row = [None if x == "n/a" else float(x) for x in nums[i : i + 8]]
                rec["steps"].append(
                    {
                        "current_density_mA_cm2": row[0] / 10 if es else row[0],
                        "cell_voltage_V": row[1],
                        **dict(zip(FE_KEYS, row[2:], strict=True)),
                    }
                )
        return rec


SCHEMA = {
    "type": "object",
    "properties": {
        "sample_id": {"type": "string"},
        "replicate": {"type": "integer"},
        "composition": {"type": "object", "additionalProperties": {"type": "number"},
                        "description": "element symbol -> atomic fraction, exactly as reported"},
        "synthesis_route": {"type": "string", "enum": ["vsp", "uoft"],
                            "description": "vsp = spark ablation, uoft = chemical reduction"},
        "batch_date": {"type": "string", "description": "YYYY-MM-DD"},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "current_density_mA_cm2": {"type": "number"},
                    "cell_voltage_V": {"type": "number"},
                    **{k: {"type": "number", "description": "Faradaic efficiency in percent (0-100 scale)"}
                       for k in FE_KEYS},
                },
                "required": ["current_density_mA_cm2", "cell_voltage_V", *FE_KEYS],
            },
        },
    },
    "required": ["sample_id", "replicate", "composition", "synthesis_route", "batch_date", "steps"],
}  # fmt: skip

PROMPT = """You are a LIMS data-entry assistant for an electrochemistry lab.
Transcribe the CO2 reduction test report below into the record_test_result tool.
Rules:
- Transcribe exactly what the document says, even if a value looks wrong. Do not fix the lab's numbers.
- Convert units only: current density to mA/cm2 (1 A/m2 = 0.1 mA/cm2), voltage to V (1000 mV = 1 V),
  Faradaic efficiencies to percent (a fraction 0.4744 is 47.44 %).
- One step per current density listed, in the order shown.

<document>
{text}
</document>"""

REPAIR = """Your transcription of the document failed these laboratory checks:
{issues}
Re-read the document. If you misread or mis-converted something, correct it. If the document itself shows these
values, keep them exactly: the lab will review the report. Call record_test_result again with the full record."""


class LLMExtractor:
    """Claude with a strict tool schema; optional repair turn driven by the validator.

    Responses are cached on disk (one JSON per document) so results are reproducible and reruns are free.
    """

    def __init__(
        self,
        model: str | None = None,
        cache_dir: str | Path = "reports/llm_outputs",
        repair: bool = True,
    ):
        self.model = model or os.getenv("LABCOPILOT_MODEL", "claude-haiku-5-5")
        self.name = f"llm:{self.model}" + ("+repair" if repair else "")
        self.cache = Path(cache_dir) / self.model
        self.cache.mkdir(parents=True, exist_ok=True)
        self.repair = repair
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def _call(self, messages: list[dict]) -> tuple[dict, list[dict]]:
        tool = {
            "name": "record_test_result",
            "description": "Store one test result in the LIMS.",
            "input_schema": SCHEMA,
        }
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            tools=[tool],
            tool_choice={"type": "tool", "name": "record_test_result"},
            messages=messages,
        )
        block = next(b for b in resp.content if b.type == "tool_use")
        return dict(block.input), [
            {"role": "assistant", "content": resp.content},
            block,
        ]

    def extract(self, doc_id: str, text: str, validate_fn=None) -> dict:
        path = self.cache / f"{doc_id}.json"
        if path.exists():
            return json.loads(path.read_text())
        messages = [{"role": "user", "content": PROMPT.format(text=text)}]
        first, (assistant, block) = self._call(messages)
        out = {"first": first, "final": first, "repaired": False}
        if self.repair and validate_fn is not None:
            issues = validate_fn(first)
            if issues:
                listing = "\n".join(f"- {i.rule}: {i.detail}" for i in issues)
                messages += [
                    assistant,
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": block.id,
                                                  "content": REPAIR.format(issues=listing)}]},
                ]  # fmt: skip
                second, _ = self._call(messages)
                out.update(final=second, repaired=True)
        path.write_text(json.dumps(out, indent=1))
        return out
