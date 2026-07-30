from pathlib import Path
import json
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if not (HERE / "lsoa_od_flows_geo.csv").exists():
    from lsoa_scenario_core import build_all_outputs
    build_all_outputs()

flows = pd.read_csv(HERE / "lsoa_od_flows_geo.csv")
keep = (
    (flows["flow_type"].eq("internal_estimated") & flows["count"].ge(10))
    | (~flows["flow_type"].eq("internal_estimated") & flows["count"].ge(5))
)
flows = flows.loc[keep & flows["distance_km"].gt(0)].copy()

colors = {
    "internal_estimated": [80, 200, 255],
    "outbound": [255, 140, 60],
    "inbound": [90, 220, 150],
}
type_labels = {
    "internal_estimated": "internal (estimated)",
    "outbound": "outbound (observed)",
    "inbound": "inbound (observed)",
}
records = [{
    "s": [round(row.origin_lon, 5), round(row.origin_lat, 5)],
    "t": [round(row.dest_lon, 5), round(row.dest_lat, 5)],
    "c": round(float(row.count), 1),
    "col": colors[row.flow_type],
    "o": row.origin_lsoa_name,
    "d": row.dest_lsoa_name,
    "ft": type_labels[row.flow_type],
} for row in flows.itertuples()]

html = """<!doctype html><html><head><meta charset="utf-8">
<title>Swindon LSOA commuting flows</title>
<script src="https://unpkg.com/deck.gl@8.9.35/dist.min.js"></script>
<script src="https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.js"></script>
<link href="https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.css" rel="stylesheet"/>
<style>
html,body,#map{margin:0;height:100%;width:100%;background:#0b0f1a}
#legend{position:absolute;top:12px;left:12px;z-index:2;font:13px system-ui;
background:rgba(10,14,26,.88);color:#eee;padding:12px 14px;border-radius:8px}
.sw{display:inline-block;width:12px;height:12px;border-radius:2px;
margin-right:6px;vertical-align:middle}.row{margin-top:4px}
</style></head><body><div id="map"></div>
<div id="tip" style="display:none;position:absolute;z-index:3;pointer-events:none;
background:rgba(0,0,0,.88);color:#fff;font:12px system-ui;padding:7px 9px;border-radius:5px"></div>
<div id="legend"><b>Swindon LSOA commuting flows (2021)</b>
<div class="row"><span class="sw" style="background:rgb(80,200,255)"></span>internal, reconstructed (≥10)</div>
<div class="row"><span class="sw" style="background:rgb(255,140,60)"></span>outbound, observed (≥5)</div>
<div class="row"><span class="sw" style="background:rgb(90,220,150)"></span>inbound, observed (≥5)</div>
<div class="row" style="opacity:.65;margin-top:8px">arc width ∝ commuters</div></div>
<script>
const FLOWS=__DATA__;
const map=new maplibregl.Map({container:'map',
style:'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
center:[-1.78,51.56],zoom:8.4,pitch:55,bearing:-15,antialias:true});
const overlay=new deck.MapboxOverlay({interleaved:false,layers:[new deck.ArcLayer({
id:'lsoa-arcs',data:FLOWS,getSourcePosition:d=>d.s,getTargetPosition:d=>d.t,
getSourceColor:d=>d.col,getTargetColor:d=>d.col,
getWidth:d=>Math.max(1,Math.sqrt(d.c)),getHeight:0.35,
widthMinPixels:1,widthMaxPixels:10,opacity:0.58,pickable:true,
onHover:info=>{const el=document.getElementById('tip');
if(info.object){el.style.display='block';el.style.left=info.x+12+'px';
el.style.top=info.y+12+'px';el.innerHTML=`${info.object.o} → ${info.object.d}<br>
${info.object.c} commuters · ${info.object.ft}`;}else{el.style.display='none';}}
})]});
let added=false;const add=()=>{if(!added){added=true;map.addControl(overlay);}};
map.on('load',add);setTimeout(add,2500);
</script></body></html>"""

(HERE / "flow_map.html").write_text(
    html.replace("__DATA__", json.dumps(records)), encoding="utf-8"
)
print(
    f"wrote flow_map.html | {len(records):,} LSOA arcs | "
    f"{flows['count'].sum():,.0f} commuters represented"
)
