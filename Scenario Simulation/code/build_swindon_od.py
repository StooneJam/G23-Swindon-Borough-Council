import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
OD_FILE = HERE / "ODWP01EW_MSOA.csv"
SWINDON_MSOA_FILE = HERE.parent / "commuting-regression" / "msoa_commute_features_swindon.csv"
OUT_FILE = HERE / "swindon_od_matrix.csv"

swindon = set(pd.read_csv(SWINDON_MSOA_FILE)["MSOA21CD"].astype(str))
print(f"Swindon MSOAs: {len(swindon)}")

df = pd.read_csv(OD_FILE)
df.columns = [c.strip() for c in df.columns]

origin_col = "Middle layer Super Output Areas code"
dest_col = "MSOA of workplace code"
cat_col = "Place of work indicator (4 categories) code"
count_col = "Count"

flows = df[df[cat_col] == 3].copy()

o_in = flows[origin_col].isin(swindon)
d_in = flows[dest_col].isin(swindon)

keep = flows[o_in | d_in].copy()

def flow_type(row):
    oi = row[origin_col] in swindon
    di = row[dest_col] in swindon
    if oi and di:
        return "internal"
    if oi and not di:
        return "outbound"
    return "inbound"

keep["flow_type"] = keep.apply(flow_type, axis=1)

out = keep.rename(columns={
    origin_col: "origin_msoa",
    "Middle layer Super Output Areas label": "origin_label",
    dest_col: "dest_msoa",
    "MSOA of workplace label": "dest_label",
    count_col: "count",
})[["origin_msoa", "origin_label", "dest_msoa", "dest_label", "count", "flow_type"]]
out = out[out["count"] > 0].sort_values("count", ascending=False)
out.to_csv(OUT_FILE, index=False)

print(f"\nSaved {len(out):,} O-D pairs -> {OUT_FILE.name}\n")
print("Total commuters by flow type:")
print(out.groupby("flow_type")["count"].agg(["sum", "size"]).rename(columns={"size": "n_pairs"}))

for ft in ["internal", "outbound", "inbound"]:
    sub = out[out["flow_type"] == ft].head(5)
    print(f"\nTop 5 {ft} flows:")
    for _, r in sub.iterrows():
        print(f"  {r['origin_label']:35s} -> {r['dest_label']:35s} {int(r['count']):>6}")
