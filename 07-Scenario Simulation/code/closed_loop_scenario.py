from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lsoa_scenario_core import build_all_outputs

diagnostics = build_all_outputs()
scenario = diagnostics["scenario"]
print(
    f"closed-loop LSOA scenario complete | "
    f"top={scenario['top_return_label']} "
    f"({scenario['top_return_pct']:+.2f}%)"
)
