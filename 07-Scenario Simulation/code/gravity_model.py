from pathlib import Path
import json
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lsoa_scenario_core import build_all_outputs

build_all_outputs()
diagnostics = json.loads(
    (HERE / "lsoa_model_diagnostics.json").read_text(encoding="utf-8")
)["gravity_model"]
coef = diagnostics["coefficients"]
print("LSOA inbound gravity model")
print(
    "T_ij ∝ "
    f"O_i^{coef['origin_mass']:.3f} "
    f"D_j^{coef['destination_jobs']:.3f} "
    f"d_ij^{coef['distance']:.3f}"
)
print(
    f"origin-grouped 5-fold OOF r(all pairs)="
    f"{diagnostics['oof_correlation_all_pairs']:.3f}; "
    f"r(positive pairs)={diagnostics['oof_correlation_positive_pairs']:.3f}"
)
