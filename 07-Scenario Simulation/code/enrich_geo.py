import pandas as pd, numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
od = pd.read_csv(HERE / "swindon_od_matrix.csv")
cen = pd.read_csv(HERE / "msoa_centroids.csv")

c = cen.set_index("msoa")
for end in ["origin", "dest"]:
    od[f"{end}_lat"] = od[f"{end}_msoa"].map(c["lat"])
    od[f"{end}_lon"] = od[f"{end}_msoa"].map(c["lon"])

missing = od[od[["origin_lat", "dest_lat"]].isna().any(axis=1)]
print(f"rows missing a centroid: {len(missing)} / {len(od)}  "
      f"({missing['count'].sum()} commuters)")
if len(missing):
    print("missing MSOAs:", pd.unique(pd.concat([
        missing.loc[missing.origin_lat.isna(), "origin_msoa"],
        missing.loc[missing.dest_lat.isna(), "dest_msoa"]]))[:10])

od = od.dropna(subset=["origin_lat", "dest_lat"]).copy()

def haversine(la1, lo1, la2, lo2):
    R = 6371.0
    p1, p2 = np.radians(la1), np.radians(la2)
    dphi = np.radians(la2 - la1); dlmb = np.radians(lo2 - lo1)
    a = np.sin(dphi/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlmb/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

od["distance_km"] = haversine(od.origin_lat, od.origin_lon, od.dest_lat, od.dest_lon).round(2)
od.to_csv(HERE / "swindon_od_flows_geo.csv", index=False)
print(f"\nsaved swindon_od_flows_geo.csv  ({len(od):,} flows)")
print("distance (km) summary for commute flows:")
print(od["distance_km"].describe().round(1).to_string())
