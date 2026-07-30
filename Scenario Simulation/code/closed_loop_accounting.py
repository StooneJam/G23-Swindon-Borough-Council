from pathlib import Path
import json
import sys
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lsoa_scenario_core import build_all_outputs

build_all_outputs()
diagnostics = json.loads(
    (HERE / "lsoa_model_diagnostics.json").read_text(encoding="utf-8")
)["scenario"]
accounting = pd.read_csv(HERE / "closed_loop_gva_accounting.csv")
print(
    "LSOA allocation scenario (fixed total inbound commuting):\n"
    f"  top target: {diagnostics['top_return_label']}\n"
    f"  top single-zone effect: {diagnostics['top_return_pct']:+.2f}%\n"
    f"  equal-budget top-two effect: {diagnostics['top_two_strategy_pct']:+.2f}%\n"
    f"  two biggest hubs: {diagnostics['biggest_hubs_strategy_pct']:+.2f}%"
)
print("\nLargest LSOA contributions in the top-two strategy:")
print(
    accounting.reindex(
        accounting["dGVA_allocation"].abs().sort_values(ascending=False).index
    ).head(10).to_string(index=False)
)
