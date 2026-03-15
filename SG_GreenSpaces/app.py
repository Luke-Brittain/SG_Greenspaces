import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import geopandas as gpd
import folium
from folium import GeoJson, GeoJsonTooltip
from streamlit_folium import st_folium
import rasterio
import tempfile
import os
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).parent

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Singapore Land Cover & Demographics",
    page_icon="🇸🇬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ──────────────────────────────────────────────────────────────────
LC_COLORS = {
    "pct_green_res": "#639922",
    "pct_parkland":  "#1D9E75",
    "pct_urban":     "#888780",
    "pct_water":     "#378ADD",
}
LC_LABELS = {
    "pct_green_res": "Green residential",
    "pct_parkland":  "Parkland",
    "pct_urban":     "Urban",
    "pct_water":     "Water",
}
REGION_COLORS = {
    "CENTRAL":    "#534AB7",
    "EAST":       "#1D9E75",
    "NORTH":      "#BA7517",
    "WEST":       "#993556",
    "NORTH-EAST": "#185FA5",
}
RASTER_CLASSES = {
    0: ("Green residential", "#639922"),
    1: ("Parkland",          "#1D9E75"),
    2: ("Urban",             "#888780"),
    3: ("Water",             "#378ADD"),
}
INC_BANDS = [
    ("income_below_1000",  "< $1k",  "#E24B4A"),
    ("income_1000_1999",   "$1–2k",  "#EF9F27"),
    ("income_2000_2999",   "$2–3k",  "#F9CB42"),
    ("income_3000_3999",   "$3–4k",  "#97C459"),
    ("income_4000_4999",   "$4–5k",  "#1D9E75"),
    ("income_5000_9999",   "$5–10k", "#378ADD"),
    ("income_10000_over",  "$10k+",  "#534AB7"),
]

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stMetricValue"] { font-size: 1.6rem; }
  .block-container { padding-top: 1.5rem; }
  h1 { font-size: 1.4rem !important; }
  h2 { font-size: 1.15rem !important; }
  h3 { font-size: 1rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data
def load_csv():
    df = pd.read_csv(BASE_DIR / "SG_Zonal_statistics_w_Greenspaces.csv")
    df["region"] = df["region"].str.strip().str.upper()
    num_cols = [
        "pop2020_total", "pop2020_male", "pop2020_female",
        "pop2020_0_14", "pop2020_15_64", "pop2020_65plus",
        "pct_green_res", "pct_parkland", "pct_urban", "pct_water",
        "px_green_res", "px_parkland", "px_urban", "px_water", "px_total",
        "income_total_workers_thousands",
        "income_below_1000", "income_1000_1999", "income_2000_2999",
        "income_3000_3999", "income_4000_4999", "income_5000_9999", "income_10000_over",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["pct_green_total"] = df["pct_green_res"] + df["pct_parkland"]
    df["pct_not_green"]   = df["pct_urban"] + df["pct_water"]
    pop = df["pop2020_total"].replace(0, np.nan)
    df["pct_age_0_14"]   = df["pop2020_0_14"]   / pop * 100
    df["pct_age_15_64"]  = df["pop2020_15_64"]  / pop * 100
    df["pct_age_65plus"] = df["pop2020_65plus"]  / pop * 100
    df["name"] = df["PLN_AREA_N"].str.title()
    return df


@st.cache_data
def load_shapefile():
    for path in [
        BASE_DIR / "MasterPlan2019PlanningAreaBoundaryNoSea.geojson",
        BASE_DIR / "planning_areas.geojson",
        BASE_DIR / "planning_areas.shp",
    ]:
        if os.path.exists(path):
            gdf = gpd.read_file(path)
            if gdf.crs and gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(epsg=4326)
            return gdf
    return None


@st.cache_data
def load_raster_preview():
    for path in [
        BASE_DIR / "classified_nonodata.tif",
        BASE_DIR / "classified_nodata.tif",
        BASE_DIR / "classified.tif",
        BASE_DIR / "land_cover.tif",
    ]:
        if os.path.exists(path):
            with rasterio.open(path) as src:
                data   = src.read(1)
                bounds = src.bounds
                rgba   = np.zeros((*data.shape, 4), dtype=np.uint8)
                for val, (_, hex_col) in RASTER_CLASSES.items():
                    r = int(hex_col[1:3], 16)
                    g = int(hex_col[3:5], 16)
                    b = int(hex_col[5:7], 16)
                    rgba[data == val] = [r, g, b, 200]
            return rgba, bounds
    return None, None


@st.cache_data
def load_satellite_preview():
    for path in [
        BASE_DIR / "SG_coastal_complete_2025_small.tif",
        BASE_DIR / "SG_coastal_complete_2025.tif",
        BASE_DIR / "satellite_image.tif",
        BASE_DIR / "SG_satellite.tif",
        BASE_DIR / "basemap.tif",
    ]:
        if os.path.exists(path):
            with rasterio.open(path) as src:
                bounds = src.bounds
                count  = src.count
                if count >= 3:
                    r = src.read(1)
                    g = src.read(2)
                    b = src.read(3)
                else:
                    band = src.read(1)
                    r = g = b = band

                def normalise(arr):
                    arr     = arr.astype(float)
                    mn, mx  = np.percentile(arr[arr > 0], (2, 98))
                    return np.clip((arr - mn) / (mx - mn) * 255, 0, 255).astype(np.uint8)

                rgba = np.zeros((*r.shape, 4), dtype=np.uint8)
                rgba[..., 0] = normalise(r)
                rgba[..., 1] = normalise(g)
                rgba[..., 2] = normalise(b)
                rgba[..., 3] = 255
            return rgba, bounds
    return None, None


# ── Load data ──────────────────────────────────────────────────────────────────
df  = load_csv()
gdf = load_shapefile()

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("🇸🇬 Singapore Dashboard")
st.sidebar.markdown("Land cover · Demographics · Income")
st.sidebar.divider()

page = st.sidebar.radio(
    "Page",
    ["🗺️ Map", "🌿 Green vs Urban", "📊 Land Cover", "👥 Demographics", "💰 Income"],
)

st.sidebar.divider()
all_regions = sorted(df["region"].dropna().unique())
sel_regions = st.sidebar.multiselect("Filter by region", all_regions, default=all_regions)
show_residential_only = st.sidebar.checkbox("Residential areas only (pop > 1,000)", value=False)

dff = df[df["region"].isin(sel_regions)].copy()
if show_residential_only:
    dff = dff[dff["pop2020_total"] > 1000]


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — MAP
# ══════════════════════════════════════════════════════════════════════════════
if page == "🗺️ Map":
    st.title("Land cover map")

    sat_rgba, sat_bounds = load_satellite_preview()
    # ── Temporary debug — remove once working ──
    import os
    st.sidebar.markdown("**Debug — files in BASE_DIR:**")
    for f in sorted(os.listdir(BASE_DIR)):
        st.sidebar.caption(f)
    rgba, bounds         = load_raster_preview()
    
        # ── Encode both images to base64 for inline HTML ──
        import base64, io

    def img_to_b64(arr: np.ndarray) -> str:
        buf = io.BytesIO()
        Image.fromarray(arr, mode="RGBA").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    has_sat  = sat_rgba is not None
    has_lc   = rgba is not None

    # ── Map mode selector ──
    map_mode = st.radio(
        "Map mode",
        ["🛰 Satellite + classification swipe", "📍 Classification + boundaries"],
        horizontal=True,
    )

    col_map, col_info = st.columns([3, 1])

    with col_map:

        # ════════════════════════════════════════════════════════════════════
        # MODE A — Swipe slider (satellite left, classified right)
        # Uses a self-contained HTML component — no folium needed
        # ════════════════════════════════════════════════════════════════════
        if map_mode == "🛰 Satellite + classification swipe":

            if not has_sat or not has_lc:
                missing = []
                if not has_sat: missing.append("satellite image")
                if not has_lc:  missing.append("classified raster")
                st.warning(f"Missing: {' and '.join(missing)}. Both files are needed for swipe mode.")
            else:
                sat_b64 = img_to_b64(sat_rgba)
                lc_b64  = img_to_b64(rgba)

                swipe_html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html, body {{ margin:0; padding:0; height:100%; }}
  #map {{ width:100%; height:560px; }}
  .swipe-handle {{
    position:absolute; top:0; bottom:0; width:4px;
    background:#fff; cursor:ew-resize; z-index:1000;
    box-shadow:0 0 6px rgba(0,0,0,0.5);
  }}
  .swipe-arrow {{
    position:absolute; top:50%; transform:translateY(-50%);
    width:32px; height:32px; background:#fff; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    left:-14px; box-shadow:0 0 6px rgba(0,0,0,0.4);
    font-size:16px; user-select:none; pointer-events:none;
  }}
  .label-left, .label-right {{
    position:absolute; bottom:12px; z-index:1000;
    background:rgba(0,0,0,0.55); color:#fff;
    font-size:12px; padding:4px 10px; border-radius:4px;
    font-family:sans-serif; pointer-events:none;
  }}
  .label-left  {{ left:12px; }}
  .label-right {{ right:12px; }}
  .lc-legend {{
    position:absolute; bottom:12px; left:50%; transform:translateX(-50%);
    z-index:1000; background:rgba(255,255,255,0.92);
    padding:6px 12px; border-radius:6px; font-size:11px;
    font-family:sans-serif; display:flex; gap:10px;
    box-shadow:0 1px 4px rgba(0,0,0,0.2);
  }}
  .leg-item {{ display:flex; align-items:center; gap:4px; }}
  .leg-swatch {{ width:12px; height:12px; border-radius:2px; flex-shrink:0; }}
</style>
</head>
<body>
<div id="map">
  <div class="label-left">🛰 Satellite</div>
  <div class="label-right">🗂 Classified</div>
  <div class="lc-legend">
    <div class="leg-item"><div class="leg-swatch" style="background:#639922"></div>Green res.</div>
    <div class="leg-item"><div class="leg-swatch" style="background:#1D9E75"></div>Parkland</div>
    <div class="leg-item"><div class="leg-swatch" style="background:#888780"></div>Urban</div>
    <div class="leg-item"><div class="leg-swatch" style="background:#378ADD"></div>Water</div>
  </div>
</div>
<script>
var map = L.map('map', {{zoomControl:true}}).setView([1.3521, 103.8198], 11);

// Dark base tiles
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_nolabels/{{z}}/{{x}}/{{y}}{{r}}.png',
  {{attribution:'CartoDB', maxZoom:18}}).addTo(map);

var imgBounds = [
  [{sat_bounds.bottom}, {sat_bounds.left}],
  [{sat_bounds.top},    {sat_bounds.right}]
];

// Left pane — satellite
var leftPane  = map.createPane('left');
var rightPane = map.createPane('right');

var satLayer = L.imageOverlay(
  'data:image/png;base64,{sat_b64}', imgBounds,
  {{opacity:1.0, pane:'left'}}
).addTo(map);

var lcLayer = L.imageOverlay(
  'data:image/png;base64,{lc_b64}', imgBounds,
  {{opacity:0.85, pane:'right'}}
).addTo(map);

// ── Swipe handle ──
var mapDiv    = document.getElementById('map');
var handle    = document.createElement('div');
handle.className = 'swipe-handle';
var arrow     = document.createElement('div');
arrow.className = 'swipe-arrow';
arrow.innerHTML = '⇔';
handle.appendChild(arrow);
mapDiv.appendChild(handle);

var mapW = mapDiv.offsetWidth;
var pos  = mapW / 2;

function setClip(x) {{
  pos  = Math.max(0, Math.min(mapW, x));
  var pct = (pos / mapW * 100).toFixed(2);
  leftPane.style.clip  = 'rect(0px, ' + pos + 'px, 9999px, 0px)';
  rightPane.style.clip = 'rect(0px, 9999px, 9999px, ' + pos + 'px)';
  handle.style.left    = pos + 'px';
}}

// Initialise
map.whenReady(function() {{
  mapW = mapDiv.offsetWidth;
  setClip(mapW / 2);
}});

// Drag
var dragging = false;
handle.addEventListener('mousedown',  function(e) {{ dragging=true; e.preventDefault(); }});
handle.addEventListener('touchstart', function(e) {{ dragging=true; }}, {{passive:true}});
document.addEventListener('mouseup',   function() {{ dragging=false; }});
document.addEventListener('touchend',  function() {{ dragging=false; }});
document.addEventListener('mousemove', function(e) {{
  if (!dragging) return;
  var rect = mapDiv.getBoundingClientRect();
  setClip(e.clientX - rect.left);
}});
document.addEventListener('touchmove', function(e) {{
  if (!dragging) return;
  var rect = mapDiv.getBoundingClientRect();
  setClip(e.touches[0].clientX - rect.left);
}}, {{passive:true}});

window.addEventListener('resize', function() {{
  mapW = mapDiv.offsetWidth;
  setClip(pos);
}});
</script>
</body>
</html>"""

                st.components.v1.html(swipe_html, height=580, scrolling=False)

        # ════════════════════════════════════════════════════════════════════
        # MODE B — Standard folium map with boundaries + click stats
        # ════════════════════════════════════════════════════════════════════
        else:
            m = folium.Map(
                location=[1.3521, 103.8198],
                zoom_start=11,
                tiles=None,
            )

            folium.TileLayer(
                tiles="https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png",
                attr="CartoDB", name="Base", show=True,
            ).add_to(m)

            # Satellite underneath
            if has_sat:
                sat_img = Image.fromarray(sat_rgba, mode="RGBA")
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    sat_img.save(tmp.name)
                    folium.raster_layers.ImageOverlay(
                        image=tmp.name,
                        bounds=[[sat_bounds.bottom, sat_bounds.left],
                                [sat_bounds.top,    sat_bounds.right]],
                        opacity=1.0, name="Satellite image", zindex=1,
                    ).add_to(m)

            # Classified raster on top
            if has_lc:
                lc_img = Image.fromarray(rgba, mode="RGBA")
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    lc_img.save(tmp.name)
                    folium.raster_layers.ImageOverlay(
                        image=tmp.name,
                        bounds=[[bounds.bottom, bounds.left],
                                [bounds.top,    bounds.right]],
                        opacity=0.75, name="Classified land cover", zindex=2,
                    ).add_to(m)
            else:
                st.info("No classified raster found.")

            # Planning area boundaries
            if gdf is not None:
                merged = gdf.merge(
                    df[["PLN_AREA_N", "name", "region", "pct_urban", "pct_green_total",
                        "pct_parkland", "pct_green_res", "pct_water", "pop2020_total"]],
                    on="PLN_AREA_N", how="left",
                )
                GeoJson(
                    merged.__geo_interface__,
                    name="Planning areas",
                    style_function=lambda f: {
                        "fillColor": "transparent", "color": "#ffffff",
                        "weight": 1.0, "fillOpacity": 0,
                    },
                    highlight_function=lambda f: {
                        "fillColor": "#ffffff", "fillOpacity": 0.2,
                        "weight": 2.5, "color": "#ffffff",
                    },
                    tooltip=GeoJsonTooltip(
                        fields=["name", "region", "pop2020_total",
                                "pct_urban", "pct_green_total", "pct_water"],
                        aliases=["Area", "Region", "Population",
                                 "% Urban", "% Green (total)", "% Water"],
                        localize=True, sticky=True,
                    ),
                ).add_to(m)
            else:
                st.info("No shapefile found.")

            # Legend
            legend_html = """
            <div style='position:fixed;bottom:30px;left:30px;z-index:9999;
                         background:rgba(0,0,0,0.7);color:#fff;
                         padding:10px 14px;border-radius:8px;
                         font-size:12px;line-height:1.8'>
              <b>Land cover</b><br>
              <span style='color:#639922'>■</span> Green residential<br>
              <span style='color:#1D9E75'>■</span> Parkland<br>
              <span style='color:#888780'>■</span> Urban<br>
              <span style='color:#378ADD'>■</span> Water
            </div>"""
            m.get_root().html.add_child(folium.Element(legend_html))
            folium.LayerControl().add_to(m)
            map_data = st_folium(m, width="100%", height=580,
                                 returned_objects=["last_object_clicked_tooltip"])

    # ── Right panel — stats (Mode B only) ──
    with col_info:
        if map_mode == "🛰 Satellite + classification swipe":
            st.subheader("How to use")
            st.markdown("""
            **Drag the ⇔ handle** left and right to reveal the satellite image underneath the classification.

            Switch to **Classification + boundaries** mode to click planning areas for detailed stats.
            """)
            st.divider()
            st.markdown("**Singapore overall**")
            for key, label in LC_LABELS.items():
                st.metric(label, f"{df[key].mean():.1f}%")
        else:
            st.subheader("Planning area stats")
            clicked = map_data.get("last_object_clicked_tooltip") if map_data else None

            if clicked and "name" in str(clicked):
                try:
                    area_name = clicked.get("name", "") if isinstance(clicked, dict) else ""
                    row = df[df["name"] == area_name]
                    if not row.empty:
                        r = row.iloc[0]
                        st.markdown(f"### {r['name']}")
                        st.caption(r["region"].title() + " Region")
                        pop = int(r["pop2020_total"]) if pd.notna(r["pop2020_total"]) else None
                        st.metric("Population",      f"{pop:,}" if pop else "n/a")
                        st.metric("% Urban",         f"{r['pct_urban']:.1f}%")
                        st.metric("% Green (total)", f"{r['pct_green_total']:.1f}%")
                        st.metric("% Parkland",      f"{r['pct_parkland']:.1f}%")
                        st.metric("% Water",         f"{r['pct_water']:.1f}%")
                        fig_mini = go.Figure(go.Bar(
                            x=[r["pct_green_res"]], y=[""],
                            orientation="h", name="Green res.", marker_color="#639922",
                        ))
                        for key, label, color in [
                            ("pct_parkland", "Parkland", "#1D9E75"),
                            ("pct_urban",    "Urban",    "#888780"),
                            ("pct_water",    "Water",    "#378ADD"),
                        ]:
                            fig_mini.add_trace(go.Bar(
                                x=[r[key]], y=[""], orientation="h",
                                name=label, marker_color=color,
                            ))
                        fig_mini.update_layout(
                            barmode="stack", height=60,
                            margin=dict(l=0, r=0, t=0, b=0),
                            showlegend=False,
                            xaxis=dict(range=[0, 100], showticklabels=False),
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                        )
                        st.plotly_chart(fig_mini, use_container_width=True)
                except Exception:
                    pass
            else:
                st.caption("Click a planning area on the map to see its statistics here.")
                st.divider()
                st.markdown("**Singapore overall**")
                for key, label in LC_LABELS.items():
                    st.metric(label, f"{df[key].mean():.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — GREEN VS URBAN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌿 Green vs Urban":
    st.title("Green spaces vs urban cover")

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Avg % green (total)", f"{dff['pct_green_total'].mean():.1f}%")
    with c2: st.metric("Avg % urban",         f"{dff['pct_urban'].mean():.1f}%")
    with c3: st.metric("Greenest area",        dff.loc[dff["pct_green_total"].idxmax(), "name"])
    with c4: st.metric("Most urban area",      dff.loc[dff["pct_urban"].idxmax(), "name"])

    st.divider()
    col_l, col_r = st.columns([2, 1])

    with col_l:
        st.subheader("Urban vs total green — scatter")
        color_by = st.selectbox(
            "Colour by", ["Region", "% Parkland", "% Water", "% Aged 65+"],
            key="scatter_color",
        )
        size_by = st.selectbox("Size by", ["Population", "Equal"], key="scatter_size")

        plot_df = dff.dropna(subset=["pct_urban", "pct_green_total"])
        sizes   = np.sqrt(plot_df["pop2020_total"].fillna(0)) * 0.6 + 6 if size_by == "Population" else 10

        color_map = {
            "Region":     ("region",         REGION_COLORS),
            "% Parkland": ("pct_parkland",   None),
            "% Water":    ("pct_water",      None),
            "% Aged 65+": ("pct_age_65plus", None),
        }
        c_col, c_scale = color_map[color_by]

        if c_scale:
            fig = px.scatter(
                plot_df, x="pct_urban", y="pct_green_total",
                color=c_col, color_discrete_map=c_scale,
                size=sizes, size_max=40, hover_name="name",
                hover_data={"pct_urban": ":.1f", "pct_green_total": ":.1f",
                            "pct_parkland": ":.1f", "pct_water": ":.1f",
                            "pop2020_total": ":,", "region": True},
                labels={"pct_urban": "% Urban", "pct_green_total": "% Green (total)"},
            )
        else:
            fig = px.scatter(
                plot_df, x="pct_urban", y="pct_green_total",
                color=c_col, color_continuous_scale="Viridis",
                size=sizes, size_max=40, hover_name="name",
                hover_data={"pct_urban": ":.1f", "pct_green_total": ":.1f",
                            "pop2020_total": ":,", "region": True},
                labels={"pct_urban": "% Urban", "pct_green_total": "% Green (total)"},
            )

        fig.update_layout(height=460, margin=dict(t=20, b=40),
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        fig.update_xaxes(gridcolor="rgba(0,0,0,0.06)", zeroline=False)
        fig.update_yaxes(gridcolor="rgba(0,0,0,0.06)", zeroline=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Green index by region")
        reg_avg = (
            dff.groupby("region")[["pct_green_total", "pct_urban", "pct_water"]]
            .mean().round(1).reset_index()
            .sort_values("pct_green_total", ascending=True)
        )
        fig_reg = go.Figure()
        fig_reg.add_trace(go.Bar(
            y=reg_avg["region"], x=reg_avg["pct_green_total"],
            orientation="h", name="Green (total)",
            marker_color=[REGION_COLORS.get(r, "#888") for r in reg_avg["region"]],
        ))
        fig_reg.update_layout(
            height=280, margin=dict(l=10, r=10, t=10, b=30),
            xaxis_title="% green", yaxis_title="",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig_reg, use_container_width=True)

        st.subheader("Green breakdown")
        st.caption("Parkland vs green residential")
        gb = (
            dff.groupby("region")[["pct_green_res", "pct_parkland"]]
            .mean().round(1).reset_index()
            .sort_values("pct_green_res", ascending=False)
        )
        fig_gb = go.Figure()
        fig_gb.add_trace(go.Bar(
            y=gb["region"], x=gb["pct_green_res"],
            orientation="h", name="Green res.", marker_color="#639922",
        ))
        fig_gb.add_trace(go.Bar(
            y=gb["region"], x=gb["pct_parkland"],
            orientation="h", name="Parkland", marker_color="#1D9E75",
        ))
        fig_gb.update_layout(
            barmode="stack", height=240, margin=dict(l=10, r=10, t=10, b=30),
            xaxis_title="% coverage", yaxis_title="",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_gb, use_container_width=True)

    st.divider()
    st.subheader("Green vs non-green — all planning areas")
    sort_green = st.radio("Sort by", ["% Green (total)", "% Urban", "Name"], horizontal=True)
    sort_map   = {"% Green (total)": "pct_green_total", "% Urban": "pct_urban", "Name": "name"}
    bar_df     = dff.sort_values(sort_map[sort_green], ascending=(sort_map[sort_green] == "name"))

    fig_gv = go.Figure()
    for key, label, color in [
        ("pct_green_res", "Green residential", "#639922"),
        ("pct_parkland",  "Parkland",          "#1D9E75"),
        ("pct_urban",     "Urban",             "#888780"),
        ("pct_water",     "Water",             "#378ADD"),
    ]:
        fig_gv.add_trace(go.Bar(
            y=bar_df["name"], x=bar_df[key],
            orientation="h", name=label, marker_color=color,
        ))
    fig_gv.update_layout(
        barmode="stack", height=max(400, len(bar_df) * 16),
        margin=dict(l=10, r=20, t=10, b=30),
        xaxis=dict(title="% cover", range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_gv, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — LAND COVER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Land Cover":
    st.title("Land cover by planning area")

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Avg % urban",     f"{dff['pct_urban'].mean():.1f}%")
    with c2: st.metric("Avg % green res", f"{dff['pct_green_res'].mean():.1f}%")
    with c3: st.metric("Avg % parkland",  f"{dff['pct_parkland'].mean():.1f}%")
    with c4: st.metric("Avg % water",     f"{dff['pct_water'].mean():.1f}%")

    st.divider()

    sort_col = st.selectbox("Sort by", list(LC_LABELS.values()) + ["Name"])
    sort_key = {v: k for k, v in LC_LABELS.items()}
    sort_key["Name"] = "name"
    skey     = sort_key[sort_col]
    plot_df  = dff.sort_values(skey, ascending=(skey == "name"))

    fig = go.Figure()
    for key, label in LC_LABELS.items():
        fig.add_trace(go.Bar(
            y=plot_df["name"], x=plot_df[key],
            orientation="h", name=label, marker_color=LC_COLORS[key],
            hovertemplate=f"<b>%{{y}}</b><br>{label}: %{{x:.1f}}%<extra></extra>",
        ))
    fig.update_layout(
        barmode="stack", height=max(500, len(plot_df) * 16),
        margin=dict(l=10, r=20, t=10, b=30),
        xaxis=dict(title="% of planning area", range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Land cover treemap")
    tm_rows = []
    for _, row in dff.dropna(subset=["px_total"]).iterrows():
        for key, label in [("px_green_res", "Green res."), ("px_parkland", "Parkland"),
                            ("px_urban", "Urban"), ("px_water", "Water")]:
            if pd.notna(row.get(key)):
                tm_rows.append({"area": row["name"], "class": label,
                                 "pixels": row[key], "region": row["region"]})
    tm_data = pd.DataFrame(tm_rows)
    if not tm_data.empty:
        fig_tm = px.treemap(
            tm_data, path=["region", "area", "class"], values="pixels", color="class",
            color_discrete_map={"Green res.": "#639922", "Parkland": "#1D9E75",
                                 "Urban": "#888780", "Water": "#378ADD"},
        )
        fig_tm.update_layout(height=500, margin=dict(t=10, b=10))
        st.plotly_chart(fig_tm, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — DEMOGRAPHICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👥 Demographics":
    st.title("Demographics")

    view = st.radio("View", ["By planning area", "By region"], horizontal=True)

    if view == "By planning area":
        pa_list = sorted(dff.dropna(subset=["pop2020_total"])["name"].unique())
        sel_pa  = st.selectbox("Select planning area", pa_list)
        row     = dff[dff["name"] == sel_pa].iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        pop = int(row["pop2020_total"]) if pd.notna(row["pop2020_total"]) else None
        with c1: st.metric("Population",      f"{pop:,}" if pop else "n/a")
        with c2: st.metric("Under 15",        f"{row['pct_age_0_14']:.1f}%"   if pd.notna(row["pct_age_0_14"])   else "n/a")
        with c3: st.metric("Working (15–64)", f"{row['pct_age_15_64']:.1f}%"  if pd.notna(row["pct_age_15_64"])  else "n/a")
        with c4: st.metric("Aged 65+",        f"{row['pct_age_65plus']:.1f}%" if pd.notna(row["pct_age_65plus"]) else "n/a")

        st.divider()
        cl, cr = st.columns(2)
        with cl:
            st.subheader("Age profile")
            fig_age = go.Figure(go.Pie(
                labels=["Under 15", "15–64", "65+"],
                values=[row["pop2020_0_14"], row["pop2020_15_64"], row["pop2020_65plus"]],
                marker_colors=["#1D9E75", "#534AB7", "#BA7517"],
                hole=0.45,
            ))
            fig_age.update_layout(height=280, margin=dict(t=10, b=10, l=10, r=10),
                                   paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_age, use_container_width=True)

        with cr:
            st.subheader("Land cover profile")
            fig_lc = go.Figure(go.Bar(
                x=list(LC_LABELS.values()),
                y=[row[k] for k in LC_LABELS],
                marker_color=list(LC_COLORS.values()),
            ))
            fig_lc.update_layout(
                height=280, margin=dict(t=10, b=10),
                yaxis_title="% cover", showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_lc, use_container_width=True)

        st.subheader(f"vs {row['region'].title()} region average")
        reg_avg      = dff[dff["region"] == row["region"]].mean(numeric_only=True)
        compare_rows = [
            {"Class": label,
             sel_pa: round(row[key], 1),
             "Region avg": round(reg_avg[key], 1),
             "Difference": round(row[key] - reg_avg[key], 1)}
            for key, label in LC_LABELS.items()
        ]
        st.dataframe(pd.DataFrame(compare_rows), hide_index=True, use_container_width=True)

    else:
        sel_region = st.selectbox("Select region", sorted(dff["region"].dropna().unique()))
        reg_df     = (dff[(dff["region"] == sel_region) & (dff["pop2020_total"] > 100)]
                      .dropna(subset=["pop2020_total"])
                      .sort_values("pct_urban", ascending=False))

        total_pop = int(reg_df["pop2020_total"].sum())
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Region population", f"{total_pop:,}")
        with c2: st.metric("Avg % urban",        f"{reg_df['pct_urban'].mean():.1f}%")
        with c3: st.metric("Avg % green",         f"{reg_df['pct_green_total'].mean():.1f}%")
        with c4: st.metric("Avg aged 65+",        f"{reg_df['pct_age_65plus'].mean():.1f}%")

        st.divider()
        cl, cr = st.columns(2)
        with cl:
            st.subheader("Land cover by planning area")
            fig = go.Figure()
            for key, label in LC_LABELS.items():
                fig.add_trace(go.Bar(
                    y=reg_df["name"], x=reg_df[key],
                    orientation="h", name=label, marker_color=LC_COLORS[key],
                ))
            fig.update_layout(
                barmode="stack", height=max(300, len(reg_df) * 28),
                margin=dict(l=10, r=10, t=10, b=30),
                xaxis=dict(range=[0, 100], title="% cover"),
                legend=dict(orientation="h", y=1.08),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

        with cr:
            st.subheader("Population by age group")
            age_totals = reg_df[["pop2020_0_14", "pop2020_15_64", "pop2020_65plus"]].sum()
            fig_p = go.Figure(go.Bar(
                x=["Under 15", "15–64", "65+"], y=age_totals.values,
                marker_color=["#1D9E75", "#534AB7", "#BA7517"],
            ))
            fig_p.update_layout(
                height=260, margin=dict(t=10, b=30),
                yaxis_title="Population",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )
            st.plotly_chart(fig_p, use_container_width=True)

            st.subheader("Ageing vs green space")
            fig_ag = px.scatter(
                reg_df.dropna(subset=["pct_age_65plus", "pct_green_total"]),
                x="pct_green_total", y="pct_age_65plus",
                size="pop2020_total", text="name",
                color_discrete_sequence=[REGION_COLORS.get(sel_region, "#888")],
                labels={"pct_green_total": "% Green", "pct_age_65plus": "% Aged 65+"},
            )
            fig_ag.update_traces(textposition="top center", textfont_size=10)
            fig_ag.update_layout(
                height=300, margin=dict(t=10, b=30), showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_ag, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — INCOME
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💰 Income":
    st.title("Income distribution")
    st.caption("GHS 2015 — residential planning areas only (28 areas have income data)")

    inc_df     = dff.dropna(subset=["income_total_workers_thousands"]).copy()
    inc_keys   = [k for k, _, _ in INC_BANDS]
    inc_totals = inc_df[inc_keys].sum(axis=1)
    for k, _, _ in INC_BANDS:
        inc_df[f"pct_{k}"] = inc_df[k] / inc_totals * 100

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Most high earners",           inc_df.loc[inc_df["income_10000_over"].idxmax(), "name"])
    with c2: st.metric("Greenest (with income data)", inc_df.loc[inc_df["pct_green_total"].idxmax(), "name"])
    with c3:
        corr = inc_df[["pct_green_total", "income_10000_over"]].corr().iloc[0, 1]
        st.metric("Green vs high income (r)", f"{corr:.2f}")

    st.divider()

    sort_inc = st.selectbox(
        "Sort by",
        ["% High earners ($10k+)", "% Low earners (<$1k)", "% Urban", "% Green (total)", "Name"],
    )
    sort_inc_map = {
        "% High earners ($10k+)": "income_10000_over",
        "% Low earners (<$1k)":   "income_below_1000",
        "% Urban":                "pct_urban",
        "% Green (total)":        "pct_green_total",
        "Name":                   "name",
    }
    sk         = sort_inc_map[sort_inc]
    inc_sorted = inc_df.sort_values(sk, ascending=(sk == "name"))

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.72, 0.28],
        shared_yaxes=True,
        subplot_titles=["Income distribution", "Land cover"],
        horizontal_spacing=0.02,
    )
    for k, label, color in INC_BANDS:
        fig.add_trace(go.Bar(
            y=inc_sorted["name"], x=inc_sorted[f"pct_{k}"],
            orientation="h", name=label, marker_color=color,
            hovertemplate=f"<b>%{{y}}</b><br>{label}: %{{x:.1f}}%<extra></extra>",
        ), row=1, col=1)
    for key, label in LC_LABELS.items():
        fig.add_trace(go.Bar(
            y=inc_sorted["name"], x=inc_sorted[key],
            orientation="h", name=label,
            marker_color=LC_COLORS[key], showlegend=False,
            hovertemplate=f"<b>%{{y}}</b><br>{label}: %{{x:.1f}}%<extra></extra>",
        ), row=1, col=2)
    fig.update_layout(
        barmode="stack", height=max(500, len(inc_sorted) * 22),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(range=[0, 100], row=1, col=1, title_text="% of workers")
    fig.update_xaxes(range=[0, 100], row=1, col=2, title_text="% cover")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Green space vs high earner share")
    fig_sc = px.scatter(
        inc_sorted,
        x="pct_green_total", y="income_10000_over",
        size="pop2020_total", color="region",
        color_discrete_map=REGION_COLORS,
        hover_name="name",
        hover_data={"pct_urban": ":.1f", "pct_green_total": ":.1f",
                    "income_10000_over": ":.1f", "pop2020_total": ":,"},
        labels={"pct_green_total":   "% Green (total)",
                "income_10000_over": "Workers earning $10k+ (thousands)",
                "region":            "Region"},
        trendline="ols",
    )
    fig_sc.update_layout(
        height=400, margin=dict(t=20, b=40),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_sc, use_container_width=True)
