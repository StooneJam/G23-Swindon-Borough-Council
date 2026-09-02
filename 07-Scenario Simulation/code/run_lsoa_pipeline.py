from pathlib import Path
import runpy
import sys

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lsoa_scenario_core import build_all_outputs

build_all_outputs()
scripts = [
    HERE / "gwr_accessibility.py",
    HERE / "all areas simulation" / "all_zone_simulation.py",
    HERE / "figure-basic" / "make_commute_figure.py",
    HERE / "figure-simulation" / "fig1_gravity_model.py",
    HERE / "figure-simulation" / "fig2_policy_simulation.py",
    HERE / "figure-simulation" / "fig3_investment_priority.py",
    HERE / "make_flow_map.py",
]
for script in scripts:
    print(f"\n--- {script.relative_to(HERE)} ---")
    runpy.run_path(str(script), run_name="__main__")
    plt.close("all")
print("\nAll LSOA scenario outputs are up to date.")
