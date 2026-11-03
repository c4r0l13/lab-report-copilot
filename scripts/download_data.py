"""Download the OCx24 raw electrochemical test export and sample registry (Abed et al., 2024), pinned commit."""

from __future__ import annotations

import urllib.request
from pathlib import Path

COMMIT = "61266df4188eeca70fc56baebfba8abd74a0c695"
BASE = f"https://raw.githubusercontent.com/FAIR-Chem/fairchem/{COMMIT}/src/fairchem/applications/ocx/data/experimental_data"
FILES = {"ExpDataDump_241113.csv": "", "materials_id-241113.csv": "supporting_data/"}


def main() -> None:
    out = Path("data/raw")
    out.mkdir(parents=True, exist_ok=True)
    for name, sub in FILES.items():
        if not (out / name).exists():
            urllib.request.urlretrieve(f"{BASE}/{sub}{name}", out / name)
    print(f"Data ready in {out.resolve()}")


if __name__ == "__main__":
    main()
