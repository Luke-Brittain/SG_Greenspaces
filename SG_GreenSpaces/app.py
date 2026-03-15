import streamlit as st
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
        # Full 5-year age bands
        "pop2020_m_0_4", "pop2020_f_0_4", "pop2020_t_0_4", "pop2020_m_5_9", "pop2020_f_5_9", "pop2020_t_5_9", "pop2020_m_10_14", "pop2020_f_10_14", "pop2020_t_10_14", "pop2020_m_15_19", "pop2020_f_15_19", "pop2020_t_15_19", "pop2020_m_20_24", "pop2020_f_20_24", "pop2020_t_20_24", "pop2020_m_25_29", "pop2020_f_25_29", "pop2020_t_25_29", "pop2020_m_30_34", "pop2020_f_30_34", "pop2020_t_30_34", "pop2020_m_35_39", "pop2020_f_35_39", "pop2020_t_35_39", "pop2020_m_40_44", "pop2020_f_40_44", "pop2020_t_40_44", "pop2020_m_45_49", "pop2020_f_45_49", "pop2020_t_45_49", "pop2020_m_50_54", "pop2020_f_50_54", "pop2020_t_50_54", "pop2020_m_55_59", "pop2020_f_55_59", "pop2020_t_55_59", "pop2020_m_60_64", "pop2020_f_60_64", "pop2020_t_60_64", "pop2020_m_65_69", "pop2020_f_65_69", "pop2020_t_65_69", "pop2020_m_70_74", "pop2020_f_70_74", "pop2020_t_70_74", "pop2020_m_75_79", "pop2020_f_75_79", "pop2020_t_75_79", "pop2020_m_80_84", "pop2020_f_80_84", "pop2020_t_80_84", "pop2020_m_85_89", "pop2020_f_85_89", "pop2020_t_85_89", "pop2020_m_90andOver", "pop2020_f_90andOver", "pop2020_t_90andOver",
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

    # ── Composite green metrics ───────────────────────────────────────────────
    # Green-Urban Balance (GUB): +1 = entirely green, -1 = entirely urban, 0 = equal
    # Bounded [-1, +1], avoids division instability
    green  = df["pct_green_total"].fillna(0)
    urban  = df["pct_urban"].fillna(0)
    denom_gub = (green + urban).replace(0, np.nan)
    df["gub"] = ((green - urban) / denom_gub).round(3)

    # Liveability Green Score (LGS): green cover as % of non-water land
    # Removes water from denominator so coastal/reservoir areas aren't penalised
    water      = df["pct_water"].fillna(0)
    non_water  = (100 - water).replace(0, np.nan)
    df["lgs"]  = ((df["pct_parkland"].fillna(0) + df["pct_green_res"].fillna(0))
                  / non_water * 100).round(1)
    pop = df["pop2020_total"].replace(0, np.nan)
    df["pct_age_0_14"]   = df["pop2020_0_14"]   / pop * 100
    df["pct_age_15_64"]  = df["pop2020_15_64"]  / pop * 100
    df["pct_age_65plus"] = df["pop2020_65plus"]  / pop * 100
    # 10-year band percentages
    def _sum_bands(cols): return sum(df[c].fillna(0) for c in cols) / pop * 100
    df["pct_age10_0_9"]   = _sum_bands(["pop2020_t_0_4",  "pop2020_t_5_9"])
    df["pct_age10_10_19"] = _sum_bands(["pop2020_t_10_14","pop2020_t_15_19"])
    df["pct_age10_20_29"] = _sum_bands(["pop2020_t_20_24","pop2020_t_25_29"])
    df["pct_age10_30_39"] = _sum_bands(["pop2020_t_30_34","pop2020_t_35_39"])
    df["pct_age10_40_49"] = _sum_bands(["pop2020_t_40_44","pop2020_t_45_49"])
    df["pct_age10_50_59"] = _sum_bands(["pop2020_t_50_54","pop2020_t_55_59"])
    df["pct_age10_60_69"] = _sum_bands(["pop2020_t_60_64","pop2020_t_65_69"])
    df["pct_age10_70_79"] = _sum_bands(["pop2020_t_70_74","pop2020_t_75_79"])
    df["pct_age10_80plus"]= _sum_bands(["pop2020_t_80_84","pop2020_t_85_89","pop2020_t_90andOver"])
    # Per-band percentage of total population
    df["pct_age_t_0_4"] = df["pop2020_t_0_4"] / pop * 100
    df["pct_age_t_5_9"] = df["pop2020_t_5_9"] / pop * 100
    df["pct_age_t_10_14"] = df["pop2020_t_10_14"] / pop * 100
    df["pct_age_t_15_19"] = df["pop2020_t_15_19"] / pop * 100
    df["pct_age_t_20_24"] = df["pop2020_t_20_24"] / pop * 100
    df["pct_age_t_25_29"] = df["pop2020_t_25_29"] / pop * 100
    df["pct_age_t_30_34"] = df["pop2020_t_30_34"] / pop * 100
    df["pct_age_t_35_39"] = df["pop2020_t_35_39"] / pop * 100
    df["pct_age_t_40_44"] = df["pop2020_t_40_44"] / pop * 100
    df["pct_age_t_45_49"] = df["pop2020_t_45_49"] / pop * 100
    df["pct_age_t_50_54"] = df["pop2020_t_50_54"] / pop * 100
    df["pct_age_t_55_59"] = df["pop2020_t_55_59"] / pop * 100
    df["pct_age_t_60_64"] = df["pop2020_t_60_64"] / pop * 100
    df["pct_age_t_65_69"] = df["pop2020_t_65_69"] / pop * 100
    df["pct_age_t_70_74"] = df["pop2020_t_70_74"] / pop * 100
    df["pct_age_t_75_79"] = df["pop2020_t_75_79"] / pop * 100
    df["pct_age_t_80_84"] = df["pop2020_t_80_84"] / pop * 100
    df["pct_age_t_85_89"] = df["pop2020_t_85_89"] / pop * 100
    df["pct_age_t_90andOver"] = df["pop2020_t_90andOver"] / pop * 100
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



# ── Load data ──────────────────────────────────────────────────────────────────
df  = load_csv()
gdf = load_shapefile()

def safe_m(row, col):
    """Safely get a numeric value from a DataFrame row."""
    try:
        v = row[col]
        return float(v) if pd.notna(v) else 0.0
    except (KeyError, TypeError, ValueError):
        return 0.0

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("🇸🇬 Singapore Dashboard")
st.sidebar.markdown("Land cover · Demographics · Income")
st.sidebar.divider()

page = st.sidebar.radio(
    "Page",
    ["🗺️ Map", "🌿 Green vs Urban", "📊 Land Cover", "👥 Demographics", "💰 Income", "⚖️ Compare", "🏆 Green Metrics"],
)

st.sidebar.divider()
all_regions = sorted(df["region"].dropna().unique())
sel_regions = st.sidebar.multiselect("Filter by region", all_regions, default=all_regions)
show_residential_only = st.sidebar.checkbox("Residential areas only (pop > 1,000)", value=False)

dff = df[df["region"].isin(sel_regions)].copy()
if show_residential_only:
    dff = dff[dff["pop2020_total"] > 1000]


# ═══════════════════════════════════════════════════════════════════════════# ==============================================================================
# PAGE 1 — MAP
# ══════════════════════════════════════════════════════════════════════════════
if page == "🗺️ Map":
    st.title("Land cover map")
    st.caption("Classified land cover overlaid with planning area boundaries. Hover over a planning area to see its stats in the panel (top-right).")

    rgba, bounds = load_raster_preview()

    m = folium.Map(location=[1.3521, 103.8198], zoom_start=11, tiles=None)

    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png",
        attr="CartoDB", name="Base", show=True,
    ).add_to(m)

    if rgba is not None:
        lc_img = Image.fromarray(rgba, mode="RGBA")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            lc_img.save(tmp.name)
            folium.raster_layers.ImageOverlay(
                image=tmp.name,
                bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
                opacity=0.75, name="Classified land cover", zindex=2,
            ).add_to(m)
    else:
        st.info("No classified raster found. Place your GeoTIFF in the app folder.")

    if gdf is not None:
        merged = gdf.merge(
            df[["PLN_AREA_N", "name", "region", "pct_urban", "pct_green_total",
                "pct_parkland", "pct_green_res", "pct_water", "pop2020_total",
                "gub", "lgs"]],
            on="PLN_AREA_N", how="left",
        )
        for col in ["pct_urban", "pct_green_total", "pct_parkland",
                    "pct_green_res", "pct_water", "pop2020_total", "gub", "lgs"]:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0).round(3)

        geojson_layer = folium.GeoJson(
            merged.__geo_interface__,
            name="Planning areas",
            style_function=lambda f: {
                "fillColor": "transparent", "color": "#ffffff",
                "weight": 1.0, "fillOpacity": 0,
            },
            highlight_function=lambda f: {
                "fillColor": "#ffffff", "fillOpacity": 0.15,
                "weight": 2.0, "color": "#ffffff",
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["name", "region", "pop2020_total",
                        "pct_urban", "pct_green_total", "pct_parkland", "pct_water"],
                aliases=["Area", "Region", "Population",
                         "% Urban", "% Green", "% Parkland", "% Water"],
                localize=True, sticky=True,
                style=(
                    "background-color:rgba(15,15,15,0.88);color:#fff;"
                    "font-family:sans-serif;font-size:12px;padding:8px 12px;"
                    "border-radius:8px;border:none;max-width:240px;"
                    "line-height:1.6;word-wrap:break-word;"
                ),
            ),
        )
        geojson_layer.add_to(m)
        _layer_var = geojson_layer.get_name()

    else:
        st.info("No shapefile found. Place your planning area .geojson in the app folder.")

    folium.LayerControl().add_to(m)
    map_data = st_folium(m, width="100%", height=640, returned_objects=[])




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
            "Colour by", ["Region", "% Parkland", "% Water", "% Aged 60+"],
            key="scatter_color",
        )
        size_by = st.selectbox("Size by", ["Population", "Equal"], key="scatter_size")

        plot_df = dff.dropna(subset=["pct_urban", "pct_green_total"])
        sizes   = np.sqrt(plot_df["pop2020_total"].fillna(0)) * 0.6 + 6 if size_by == "Population" else 10

        plot_df = plot_df.copy()
        plot_df["pct_age_60plus"] = (
            plot_df["pct_age10_60_69"].fillna(0) +
            plot_df["pct_age10_70_79"].fillna(0) +
            plot_df["pct_age10_80plus"].fillna(0)
        )
        color_map = {
            "Region":     ("region",         REGION_COLORS),
            "% Parkland": ("pct_parkland",   None),
            "% Water":    ("pct_water",      None),
            "% Aged 60+": ("pct_age_60plus", None),
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

    # ── Shared band definition ────────────────────────────────────────────────
    _BANDS10 = [
        ("0–9",   ["0_4",  "5_9"]),
        ("10–19", ["10_14","15_19"]),
        ("20–29", ["20_24","25_29"]),
        ("30–39", ["30_34","35_39"]),
        ("40–49", ["40_44","45_49"]),
        ("50–59", ["50_54","55_59"]),
        ("60–69", ["60_64","65_69"]),
        ("70–79", ["70_74","75_79"]),
        ("80+",   ["80_84","85_89","90andOver"]),
    ]
    _BAND_LBLS = [lbl for lbl, _ in _BANDS10]

    def _pyr_vals(row_or_series, sex, as_pct=False):
        total = max(float(row_or_series.get("pop2020_total") or 1), 1) if as_pct else 1
        prefix = "m" if sex == "male" else "f"
        return [
            sum(float(row_or_series.get(f"pop2020_{prefix}_{b}") or 0) for b in bnds) / total * (100 if as_pct else 1)
            for _, bnds in _BANDS10
        ]

    def _pyramid_fig(m_vals, f_vals, title="", height=400):
        max_v = max(max(m_vals), max(f_vals), 1)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Male", y=_BAND_LBLS, x=[-v for v in m_vals],
            orientation="h", marker_color="#534AB7", opacity=0.85,
            hovertemplate="%{customdata:,}<extra>Male</extra>",
            customdata=[round(v) for v in m_vals],
        ))
        fig.add_trace(go.Bar(
            name="Female", y=_BAND_LBLS, x=f_vals,
            orientation="h", marker_color="#D4537E", opacity=0.85,
            hovertemplate="%{x:,}<extra>Female</extra>",
        ))
        fig.update_layout(
            title=dict(text=title, font=dict(size=13)) if title else {},
            barmode="overlay", height=height,
            margin=dict(t=30 if title else 10, b=10, l=0, r=0),
            xaxis=dict(
                tickvals=[-max_v, -max_v//2, 0, max_v//2, max_v],
                ticktext=[f"{max_v:,.0f}", f"{max_v//2:,.0f}", "0",
                          f"{max_v//2:,.0f}", f"{max_v:,.0f}"],
                title="Population",
            ),
            legend=dict(orientation="h", y=1.05),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    def _scorecard(label, value, colour="var(--color-text-primary,#222)"):
        return (
            f"<div style='background:var(--color-background-secondary,#f5f5f5);"
            f"border-radius:8px;padding:10px 12px;margin-bottom:4px'>"
            f"<div style='font-size:13px;font-weight:500;"
            f"color:var(--color-text-secondary,#555);margin-bottom:6px'>{label}</div>"
            f"<div style='font-size:18px;font-weight:700;color:{colour}'>{value}</div>"
            f"</div>"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # BY PLANNING AREA
    # ══════════════════════════════════════════════════════════════════════════
    if view == "By planning area":
        pa_list = sorted(dff.dropna(subset=["pop2020_total"])["name"].unique())
        sel_pa  = st.selectbox("Select planning area", pa_list)
        row     = dff[dff["name"] == sel_pa].iloc[0]

        # ── Scorecards ───────────────────────────────────────────────────────
        st.divider()
        pop   = int(row["pop2020_total"]) if pd.notna(row["pop2020_total"]) else None
        p0_19 = safe_m(row,"pct_age10_0_9") + safe_m(row,"pct_age10_10_19")
        p20_59= sum(safe_m(row,f"pct_age10_{b}") for b in ["20_29","30_39","40_49","50_59"])
        p60p  = sum(safe_m(row,f"pct_age10_{b}") for b in ["60_69","70_79","80plus"])

        sc_metrics = [
            ("Population",    f"{pop:,}" if pop else "n/a",  "#534AB7"),
            ("% Urban",       f"{safe_m(row,'pct_urban'):.1f}%",       "#888780"),
            ("% Green",       f"{safe_m(row,'pct_green_total'):.1f}%", "#639922"),
            ("% Parkland",    f"{safe_m(row,'pct_parkland'):.1f}%",    "#1D9E75"),
            ("% Water",       f"{safe_m(row,'pct_water'):.1f}%",       "#378ADD"),
            ("Aged 60+",      f"{p60p:.1f}%",                          "#BA7517"),
        ]
        cols = st.columns(6)
        for col, (lbl, val, clr) in zip(cols, sc_metrics):
            with col:
                st.markdown(_scorecard(lbl, val, clr), unsafe_allow_html=True)

        # ── Age & sex profile ─────────────────────────────────────────────
        st.divider()
        st.subheader("Age & sex profile")
        st.caption("Males left · Females right")
        m_vals = _pyr_vals(row, "male")
        f_vals = _pyr_vals(row, "female")
        st.plotly_chart(_pyramid_fig(m_vals, f_vals, height=400),
                        use_container_width=True)

        # ── Land cover ────────────────────────────────────────────────────
        st.divider()
        st.subheader("Land cover")
        cl, cr = st.columns(2)
        with cl:
            fig_lc = go.Figure(go.Bar(
                x=list(LC_LABELS.values()),
                y=[row[k] for k in LC_LABELS],
                marker_color=list(LC_COLORS.values()),
                hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
            ))
            fig_lc.update_layout(
                height=300, margin=dict(t=10, b=10),
                yaxis_title="% cover", showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_lc, use_container_width=True)

        with cr:
            # Land cover vs region average — diverging bar
            reg_avg = dff[dff["region"] == row["region"]].mean(numeric_only=True)
            lc_diffs = [round(row[k] - reg_avg[k], 1) for k in LC_LABELS]
            lc_colors_div = ["#639922" if d >= 0 else "#888780" for d in lc_diffs]
            abs_max_lc = max(abs(d) for d in lc_diffs) * 1.3 or 1
            fig_lc_div = go.Figure(go.Bar(
                y=list(LC_LABELS.values()), x=lc_diffs,
                orientation="h", marker_color=lc_colors_div,
                hovertemplate="%{y}: %{x:+.1f}pp vs region avg<extra></extra>",
            ))
            fig_lc_div.add_vline(x=0, line_width=1,
                                  line_color="rgba(128,128,128,0.4)")
            fig_lc_div.update_layout(
                height=300, margin=dict(t=30, b=10, l=0, r=10),
                xaxis=dict(title="pp vs region avg", range=[-abs_max_lc, abs_max_lc],
                           ticksuffix="pp"),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )
            fig_lc_div.add_annotation(
                x=-abs_max_lc, y=1.1, xref="x", yref="paper",
                text="← Below avg", showarrow=False,
                font=dict(size=10, color="#888"), xanchor="left",
            )
            fig_lc_div.add_annotation(
                x=abs_max_lc, y=1.1, xref="x", yref="paper",
                text="Above avg →", showarrow=False,
                font=dict(size=10, color="#639922"), xanchor="right",
            )
            st.plotly_chart(fig_lc_div, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # BY REGION
    # ══════════════════════════════════════════════════════════════════════════
    else:
        sel_region = st.selectbox("Select region", sorted(dff["region"].dropna().unique()))
        reg_df     = (dff[(dff["region"] == sel_region) & (dff["pop2020_total"] > 100)]
                      .dropna(subset=["pop2020_total"])
                      .sort_values("pct_urban", ascending=False))

        # ── Scorecards ───────────────────────────────────────────────────────
        st.divider()
        total_pop  = int(reg_df["pop2020_total"].sum())
        avg_urban  = reg_df["pct_urban"].mean()
        avg_green  = reg_df["pct_green_total"].mean()
        avg_60plus = (reg_df["pct_age10_60_69"].fillna(0) +
                      reg_df["pct_age10_70_79"].fillna(0) +
                      reg_df["pct_age10_80plus"].fillna(0)).mean()

        reg_sc = [
            ("Population",  f"{total_pop:,}",        "#534AB7"),
            ("Avg % urban", f"{avg_urban:.1f}%",      "#888780"),
            ("Avg % green", f"{avg_green:.1f}%",      "#639922"),
            ("Avg aged 60+",f"{avg_60plus:.1f}%",     "#BA7517"),
        ]
        cols = st.columns(4)
        for col, (lbl, val, clr) in zip(cols, reg_sc):
            with col:
                st.markdown(_scorecard(lbl, val, clr), unsafe_allow_html=True)

        # ── Land cover ────────────────────────────────────────────────────
        st.divider()
        st.subheader("Land cover by planning area")
        fig = go.Figure()
        for key, label in LC_LABELS.items():
            fig.add_trace(go.Bar(
                y=reg_df["name"], x=reg_df[key],
                orientation="h", name=label, marker_color=LC_COLORS[key],
            ))
        fig.update_layout(
            barmode="stack", height=max(300, len(reg_df) * 28) + 50,
            margin=dict(l=10, r=10, t=10, b=60),
            xaxis=dict(range=[0, 100], title="% cover"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15,
                        xanchor="left", x=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Age & sex profile ─────────────────────────────────────────────
        st.divider()
        st.subheader("Age & sex profile")
        st.caption("Region total · Males left · Females right")

        def _col_sum(col):
            return reg_df[col].apply(
                lambda x: float(x) if x not in ("","-",None) else 0).sum()

        rm = [sum(_col_sum(f"pop2020_m_{b}") for b in bnds) for _, bnds in _BANDS10]
        rf = [sum(_col_sum(f"pop2020_f_{b}") for b in bnds) for _, bnds in _BANDS10]
        st.plotly_chart(_pyramid_fig(rm, rf, height=400), use_container_width=True)

        # ── Ageing vs green scatter ───────────────────────────────────────
        st.divider()
        st.subheader("Ageing vs green space")
        fig_ag = px.scatter(
            reg_df.assign(
                pct_age_60plus=(reg_df["pct_age10_60_69"].fillna(0) +
                                reg_df["pct_age10_70_79"].fillna(0) +
                                reg_df["pct_age10_80plus"].fillna(0))
            ).dropna(subset=["pct_green_total"]),
            x="pct_green_total", y="pct_age_60plus",
            size="pop2020_total", text="name",
            color_discrete_sequence=[REGION_COLORS.get(sel_region, "#888")],
            labels={"pct_green_total": "% Green", "pct_age_60plus": "% Aged 60+"},
        )
        fig_ag.update_traces(textposition="top center", textfont_size=10)
        fig_ag.update_layout(
            height=360, margin=dict(t=10, b=30), showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_ag, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — INCOME
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💰 Income":
    st.title("Income distribution")
    st.caption("GHS 2015 — residential planning areas only · 28 of 55 areas have income data")

    inc_df     = dff.dropna(subset=["income_total_workers_thousands"]).copy()
    inc_keys   = [k for k, _, _ in INC_BANDS]
    inc_totals = inc_df[inc_keys].sum(axis=1)
    for k, _, _ in INC_BANDS:
        inc_df[f"pct_{k}"] = inc_df[k] / inc_totals * 100

    # ── 1. Headline insight scorecards ────────────────────────────────────────
    st.divider()

    # Modal bracket — the income band with the highest % of workers per area
    bracket_labels = {k: l for k, l, _ in INC_BANDS}
    inc_df["modal_bracket"] = inc_df[[f"pct_{k}" for k in inc_keys]].idxmax(axis=1).str.replace("pct_", "")
    inc_df["modal_label"]   = inc_df["modal_bracket"].map(bracket_labels)

    top_high  = inc_df.loc[inc_df["pct_income_10000_over"].idxmax()]
    top_low   = inc_df.loc[inc_df["pct_income_below_1000"].idxmax()]
    top_green = inc_df.loc[inc_df["pct_green_total"].idxmax()]
    corr_gub  = inc_df[["gub", "pct_income_10000_over"]].corr().iloc[0, 1]
    corr_lgs  = inc_df[["lgs", "pct_income_10000_over"]].corr().iloc[0, 1]

    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    for col, lbl, val, sub, clr in [
        (sc1, "Highest % $10k+ earners",  top_high["name"],
              f"{top_high['pct_income_10000_over']:.1f}% of workers", "#534AB7"),
        (sc2, "Highest % low earners (<$1k)", top_low["name"],
              f"{top_low['pct_income_below_1000']:.1f}% of workers",  "#E24B4A"),
        (sc3, "Greenest area (income data)", top_green["name"],
              f"{top_green['pct_green_total']:.1f}% green cover",     "#639922"),
        (sc4, "GUB vs $10k+ (r)",  f"{corr_gub:+.2f}",
              "Positive = greener areas earn more",                    "#1D9E75"),
        (sc5, "LGS vs $10k+ (r)",  f"{corr_lgs:+.2f}",
              "Positive = liveable green → higher income",             "#1D9E75"),
    ]:
        with col:
            st.markdown(
                f"<div style='background:var(--color-background-secondary,#f5f5f5);"
                f"border-radius:8px;padding:10px 12px;margin-bottom:4px'>"
                f"<div style='font-size:13px;font-weight:500;"
                f"color:var(--color-text-secondary,#555);margin-bottom:6px'>{lbl}</div>"
                f"<div style='font-size:18px;font-weight:700;color:{clr}'>{val}</div>"
                f"<div style='font-size:11px;color:#888;margin-top:3px'>{sub}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.divider()

    # ── 3. Heatmap — brackets as columns, areas as rows ───────────────────────
    st.subheader("Income bracket heatmap")
    st.caption("Colour intensity shows share of workers in each bracket — easier to spot "
               "which bracket dominates and which areas are outliers.")

    sort_inc = st.selectbox(
        "Sort areas by",
        ["% High earners ($10k+)", "% Low earners (<$1k)", "% Green (total)", "GUB score", "Name"],
    )
    sort_map = {
        "% High earners ($10k+)": ("pct_income_10000_over", False),
        "% Low earners (<$1k)":   ("pct_income_below_1000", False),
        "% Green (total)":        ("pct_green_total",        False),
        "GUB score":              ("gub",                    False),
        "Name":                   ("name",                   True),
    }
    sk, sk_asc = sort_map[sort_inc]
    inc_sorted = inc_df.sort_values(sk, ascending=sk_asc)

    # ── Improvement 1: Diverging scale anchored at column average ─────────────
    # Compute deviation from Singapore average per bracket
    hm_labels  = [l for _, l, _ in INC_BANDS]
    pct_cols   = [f"pct_{k}" for k in inc_keys]
    sg_avg_row = inc_df[pct_cols].mean()                    # per-bracket SG avg
    hm_z_raw   = inc_sorted[pct_cols].values                # absolute values
    hm_z_dev   = (inc_sorted[pct_cols] - sg_avg_row).values # deviation from avg

    # ── Improvement 5: SG average reference row appended at bottom ───────────
    # Append a synthetic "Singapore avg" row to both matrices
    avg_label  = "── Singapore avg ──"
    area_names = inc_sorted["name"].tolist() + [avg_label]
    hm_z_dev_with_avg = np.vstack([hm_z_dev, np.zeros(len(inc_keys))])
    hm_z_raw_with_avg = np.vstack([hm_z_raw,  sg_avg_row.values])

    # ── Improvement 6: Only show labels for cells ≥ 5pp from column average ───
    threshold = 5.0
    hm_text = []
    for i, row_dev in enumerate(hm_z_dev_with_avg):
        row_raw = hm_z_raw_with_avg[i]
        if i == len(hm_z_dev):          # avg row — always show
            hm_text.append([f"{v:.1f}%" for v in row_raw])
        else:
            hm_text.append([
                f"{row_raw[j]:.1f}%" if abs(row_dev[j]) >= threshold else ""
                for j in range(len(inc_keys))
            ])

    # ── Improvement 2: Modal bracket column ──────────────────────────────────
    modal_colors = [c for _, _, c in INC_BANDS]
    modal_map    = {k: c for (k, _, c), pc in zip(INC_BANDS, pct_cols)}
    bracket_color_map = {k: c for k, _, c in INC_BANDS}
    inc_sorted["modal_key"] = inc_sorted[pct_cols].idxmax(axis=1).str.replace("pct_", "")
    modal_vals  = inc_sorted["modal_key"].tolist()
    # For avg row, find modal bracket of SG avg
    modal_vals.append(inc_keys[int(np.argmax(sg_avg_row.values))])

    # Diverging colorscale: red = below avg, white = avg, blue/purple = above avg
    diverging_scale = [
        [0.0,  "#C0392B"],
        [0.25, "#E8827A"],
        [0.5,  "rgba(240,240,240,0.3)"],
        [0.75, "#7B8FD4"],
        [1.0,  "#2C3E9E"],
    ]

    # Symmetric scale range so 0-deviation maps to centre
    max_dev = max(abs(hm_z_dev_with_avg).max(), 1.0)

    fig_hm = make_subplots(
        rows=1, cols=2,
        column_widths=[0.88, 0.12],
        shared_yaxes=True,
        horizontal_spacing=0.01,
        subplot_titles=["Deviation from Singapore average (pp)", "Dominant bracket"],
    )

    # Main heatmap — deviation from average
    fig_hm.add_trace(go.Heatmap(
        z=hm_z_dev_with_avg,
        x=hm_labels,
        y=area_names,
        text=hm_text,
        texttemplate="%{text}",
        textfont=dict(size=9),
        colorscale=diverging_scale,
        zmid=0,
        zmin=-max_dev, zmax=max_dev,
        showscale=True,
        colorbar=dict(
            title="pp vs avg", ticksuffix="pp",
            len=0.6, x=0.87,
            tickvals=[-max_dev, -max_dev/2, 0, max_dev/2, max_dev],
            ticktext=[f"{-max_dev:.0f}", f"{-max_dev/2:.0f}", "0",
                      f"{max_dev/2:.0f}", f"{max_dev:.0f}"],
        ),
        hovertemplate="<b>%{y}</b><br>%{x}<br>Deviation: %{z:+.1f}pp vs SG avg<extra></extra>",
    ), row=1, col=1)

    # Separator line above avg row
    n_areas = len(inc_sorted)
    fig_hm.add_shape(
        type="line", xref="x", yref="y",
        x0=-0.5, x1=len(inc_keys) - 0.5,
        y0=n_areas - 0.5, y1=n_areas - 0.5,
        line=dict(color="rgba(255,255,255,0.6)", width=2, dash="dot"),
        row=1, col=1,
    )

    # Modal bracket column — coloured cells
    modal_z     = [[0.5]] * len(area_names)   # uniform z, use colour from marker
    modal_texts = []
    modal_hover = []
    for mv in modal_vals:
        lbl = bracket_color_map.get(mv, "")
        # Map bracket key to label
        lbl_txt = next((l for k, l, _ in INC_BANDS if k == mv), mv)
        modal_texts.append([lbl_txt])
        modal_hover.append([lbl_txt])

    # Render as scatter with coloured markers instead of heatmap for exact colours
    for i, (area, mv) in enumerate(zip(area_names, modal_vals)):
        clr = bracket_color_map.get(mv, "#888")
        lbl_txt = next((l for k, l, _ in INC_BANDS if k == mv), mv)
        fig_hm.add_trace(go.Scatter(
            x=["Dominant"],
            y=[area],
            mode="markers+text",
            marker=dict(color=clr, size=14, symbol="square"),
            text=[lbl_txt],
            textposition="middle right",
            textfont=dict(size=8),
            showlegend=False,
            hovertemplate=f"<b>{area}</b><br>Dominant bracket: {lbl_txt}<extra></extra>",
        ), row=1, col=2)

    fig_hm.update_layout(
        height=max(460, len(area_names) * 20),
        margin=dict(l=10, r=20, t=40, b=10),
        xaxis=dict(side="top", tickfont=dict(size=11)),
        xaxis2=dict(side="top", tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10), autorange="reversed"),
        yaxis2=dict(autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_hm, use_container_width=True)
    st.caption("Blue = above Singapore average for that bracket · Red = below average · "
               "Labels shown only where deviation ≥ 5pp · Dashed line separates Singapore average reference row.")

    st.divider()

    # ── 5. Income vs green metrics scatter ────────────────────────────────────
    st.subheader("Income vs green metrics")
    st.caption("Each bubble is a planning area, sized by population. "
               "A positive trend suggests wealthier areas tend to have more — or better-quality — green space.")

    green_metric = st.radio(
        "Green metric", ["GUB (Green-Urban Balance)", "LGS (Liveability Green Score)",
                         "% Green (total)"],
        horizontal=True,
    )
    gm_col = {"GUB (Green-Urban Balance)":    "gub",
               "LGS (Liveability Green Score)": "lgs",
               "% Green (total)":              "pct_green_total"}[green_metric]
    gm_label = {"gub": "GUB score", "lgs": "LGS (%)",
                 "pct_green_total": "% Green (total)"}[gm_col]
    gm_fmt   = ":.3f" if gm_col == "gub" else ":.1f"

    inc_scatter = inc_sorted.dropna(subset=[gm_col, "pct_income_10000_over"]).copy()
    inc_scatter["pct_age_60plus"] = (
        inc_scatter["pct_age10_60_69"].fillna(0) +
        inc_scatter["pct_age10_70_79"].fillna(0) +
        inc_scatter["pct_age10_80plus"].fillna(0)
    )

    sc_left, sc_right = st.columns(2)
    for col, y_col, y_label, title in [
        (sc_left,  "pct_income_10000_over", "Workers earning $10k+ (%)", "Green space vs high earners"),
        (sc_right, "pct_income_below_1000", "Workers earning <$1k (%)",  "Green space vs low earners"),
    ]:
        with col:
            # Scatter coloured by region — no per-region trendline
            fig_sc = px.scatter(
                inc_scatter,
                x=gm_col, y=y_col,
                size="pop2020_total", color="region",
                color_discrete_map=REGION_COLORS,
                hover_name="name",
                hover_data={gm_col: gm_fmt, y_col: ":.1f", "pop2020_total": ":,"},
                labels={gm_col: gm_label, y_col: y_label, "region": "Region"},
                title=title,
            )
            # Add a single overall OLS trendline manually
            x_vals = inc_scatter[gm_col].values
            y_vals = inc_scatter[y_col].values
            mask   = ~(np.isnan(x_vals) | np.isnan(y_vals))
            if mask.sum() > 2:
                m, b   = np.polyfit(x_vals[mask], y_vals[mask], 1)
                x_line = np.array([x_vals[mask].min(), x_vals[mask].max()])
                y_line = m * x_line + b
                fig_sc.add_trace(go.Scatter(
                    x=x_line, y=y_line,
                    mode="lines",
                    line=dict(color="rgba(100,100,100,0.6)", width=2, dash="dot"),
                    showlegend=False, hoverinfo="skip",
                ))
            r = float(np.corrcoef(x_vals[mask], y_vals[mask])[0, 1]) if mask.sum() > 2 else 0.0
            fig_sc.add_annotation(
                x=0.97, y=0.05, xref="paper", yref="paper",
                text=f"r = {r:+.2f}",
                showarrow=False, font=dict(size=12, color="#888"),
                xanchor="right",
            )
            fig_sc.update_layout(
                height=360, margin=dict(t=40, b=30),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_sc, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — COMPARE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚖️ Compare":
    st.title("Planning area comparison")
    st.caption("Select two areas to compare — choose any planning area, a region average, or Singapore overall.")

    def safe(v, fmt=".1f"):
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    # ── Build synthetic aggregate rows for regions / Singapore overall ─────────
    AGE_BANDS_10 = ["0_4","5_9","10_14","15_19","20_24","25_29","30_34","35_39",
                    "40_44","45_49","50_54","55_59","60_64","65_69","70_74",
                    "75_79","80_84","85_89","90andOver"]

    def make_aggregate_row(subset, label, region_val):
        """Return a Series representing the population-weighted average of subset.

        Fixes:
        - Includes int64 columns (pop2020_m/f/t_*) not just float64
        - Age band counts are summed (not averaged) — total headcount per band
        - Income is averaged only over rows that actually have income data
        - pct_age10_* and pct_age_* are recomputed from aggregated counts
        """
        pop = subset["pop2020_total"].fillna(0)
        total_pop = pop.sum()
        agg = {}

        # Numeric columns — include both float64 and int64
        num_cols = [c for c in subset.columns
                    if subset[c].dtype in [float, "float64", int, "int64", "Int64"]]

        # Income columns — average only over rows WITH income data
        inc_keys = ["income_total_workers_thousands", "income_below_1000",
                    "income_1000_1999", "income_2000_2999", "income_3000_3999",
                    "income_4000_4999", "income_5000_9999", "income_10000_over"]
        inc_subset = subset[subset["income_total_workers_thousands"].notna() &
                            (subset["income_total_workers_thousands"] > 0)]

        for c in num_cols:
            if c == "pop2020_total":
                agg[c] = total_pop

            # Age band raw counts — sum across all areas
            elif (c.startswith("pop2020_m_") or c.startswith("pop2020_f_") or
                  c.startswith("pop2020_t_")):
                agg[c] = subset[c].fillna(0).sum()

            # pct_ land cover / green metrics — population-weighted mean
            elif c.startswith("pct_"):
                w = pop * subset[c].fillna(0)
                agg[c] = w.sum() / total_pop if total_pop > 0 else 0.0

            # Income columns — mean over income-reporting areas only
            elif c in inc_keys:
                if not inc_subset.empty:
                    agg[c] = inc_subset[c].fillna(0).mean()
                else:
                    agg[c] = np.nan

            # Broad age band counts (pop2020_0_14 etc) — sum
            elif c.startswith("pop2020_"):
                agg[c] = subset[c].fillna(0).sum()

            else:
                agg[c] = subset[c].fillna(0).mean()

        agg["name"]   = label
        agg["region"] = region_val

        # Recompute GUB and LGS from aggregated pcts
        g  = agg.get("pct_green_total", 0)
        u  = agg.get("pct_urban", 0)
        wa = agg.get("pct_water", 0)
        agg["gub"] = round((g - u) / (g + u), 3) if (g + u) > 0 else 0.0
        nw = 100 - wa
        agg["lgs"] = round((agg.get("pct_parkland", 0) + agg.get("pct_green_res", 0))
                           / nw * 100, 1) if nw > 0 else 0.0

        # Recompute pct_age10_* from summed age band counts
        t_pop = agg.get("pop2020_total", 1) or 1
        bands_10 = [
            ("pct_age10_0_9",   ["0_4",  "5_9"]),
            ("pct_age10_10_19", ["10_14","15_19"]),
            ("pct_age10_20_29", ["20_24","25_29"]),
            ("pct_age10_30_39", ["30_34","35_39"]),
            ("pct_age10_40_49", ["40_44","45_49"]),
            ("pct_age10_50_59", ["50_54","55_59"]),
            ("pct_age10_60_69", ["60_64","65_69"]),
            ("pct_age10_70_79", ["70_74","75_79"]),
            ("pct_age10_80plus",["80_84","85_89","90andOver"]),
        ]
        for pct_col, b_list in bands_10:
            agg[pct_col] = sum(agg.get(f"pop2020_t_{b}", 0) for b in b_list) / t_pop * 100

        # Recompute broad bands
        agg["pct_age_0_14"]   = sum(agg.get(f"pop2020_t_{b}", 0) for b in ["0_4","5_9","10_14"]) / t_pop * 100
        agg["pct_age_15_64"]  = sum(agg.get(f"pop2020_t_{b}", 0) for b in
                                    ["15_19","20_24","25_29","30_34","35_39",
                                     "40_44","45_49","50_54","55_59","60_64"]) / t_pop * 100
        agg["pct_age_65plus"] = sum(agg.get(f"pop2020_t_{b}", 0) for b in
                                    ["65_69","70_74","75_79","80_84","85_89","90andOver"]) / t_pop * 100

        return pd.Series(agg)

    AGGREGATE_OPTIONS = {
        "🌐 Singapore overall": make_aggregate_row(df, "Singapore overall", "ALL"),
        "🔵 Central region":    make_aggregate_row(df[df["region"]=="CENTRAL"], "Central region", "CENTRAL"),
        "🟢 East region":       make_aggregate_row(df[df["region"]=="EAST"],    "East region",    "EAST"),
        "🟡 North region":      make_aggregate_row(df[df["region"]=="NORTH"],   "North region",   "NORTH"),
        "🟠 North-East region": make_aggregate_row(df[df["region"]=="NORTH-EAST"],"North-East region","NORTH-EAST"),
        "🔴 West region":       make_aggregate_row(df[df["region"]=="WEST"],    "West region",    "WEST"),
    }

    all_pa      = sorted(df["name"].dropna().unique())
    all_options = list(AGGREGATE_OPTIONS.keys()) + all_pa

    def get_row(selection):
        if selection in AGGREGATE_OPTIONS:
            return AGGREGATE_OPTIONS[selection]
        return df[df["name"] == selection].iloc[0]

    col_a, col_b = st.columns(2)
    with col_a:
        sel_a = st.selectbox("Area A", all_options,
                             index=all_options.index("Bedok") if "Bedok" in all_options else 0,
                             key="cmp_a")
    with col_b:
        sel_b = st.selectbox("Area B", all_options,
                             index=all_options.index("Tampines") if "Tampines" in all_options else 1,
                             key="cmp_b")

    ra   = get_row(sel_a)
    rb   = get_row(sel_b)
    pa_a = ra["name"]
    pa_b = rb["name"]

    # ── Headline metrics ───────────────────────────────────────────────────────
    st.divider()

    def age60plus(r):
        return (safe(r.get("pct_age10_60_69", 0)) +
                safe(r.get("pct_age10_70_79", 0)) +
                safe(r.get("pct_age10_80plus", 0)))

    metrics = [
        ("Population",      "pop2020_total",        "pop"),
        ("% Urban",         "pct_urban",             "pct"),
        ("% Green (total)", "pct_green_total",       "pct"),
        ("% Parkland",      "pct_parkland",          "pct"),
        ("% Water",         "pct_water",             "pct"),
        ("Aged 60+",        "pct_age10_60plus_sum",  "pct"),
    ]

    def get_vals(field):
        if field == "pct_age10_60plus_sum":
            return age60plus(ra), age60plus(rb)
        return ra[field], rb[field]

    def fmt_val(v, kind):
        if kind == "pop":
            return f"{int(v):,}" if pd.notna(v) and v > 0 else "n/a"
        return f"{safe(v):.1f}%" if pd.notna(v) else "n/a"

    # Area name colour key above the scorecards
    st.markdown(
        f"<div style='display:flex;gap:24px;margin-bottom:8px;font-size:13px'>"
        f"<span><span style='display:inline-block;width:12px;height:12px;"
        f"border-radius:2px;background:#639922;margin-right:6px;vertical-align:middle'></span>"
        f"<strong>{pa_a}</strong></span>"
        f"<span><span style='display:inline-block;width:12px;height:12px;"
        f"border-radius:2px;background:#378ADD;margin-right:6px;vertical-align:middle'></span>"
        f"<strong>{pa_b}</strong></span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    for col, (label, field, kind) in zip([c1,c2,c3,c4,c5,c6], metrics):
        va, vb   = get_vals(field)
        sa       = fmt_val(va, kind)
        sb       = fmt_val(vb, kind)
        nva      = safe(va)
        nvb      = safe(vb)
        # Winner badge — ▲ on the higher value, dim the lower
        if nva > nvb:
            badge_a, badge_b   = "▲", ""
            opacity_a, opacity_b = "1", "0.5"
        elif nvb > nva:
            badge_a, badge_b   = "", "▲"
            opacity_a, opacity_b = "0.5", "1"
        else:
            badge_a = badge_b = ""
            opacity_a = opacity_b = "1"

        with col:
            st.markdown(f"""
            <div style="background:var(--color-background-secondary,#f5f5f5);
                        border-radius:8px;padding:10px 12px;margin-bottom:4px">
              <div style="font-size:13px;font-weight:500;color:var(--color-text-secondary,#555);margin-bottom:8px">{label}</div>
              <div style="opacity:{opacity_a}">
                <div style="font-size:10px;color:#639922;font-weight:500;
                            text-overflow:ellipsis;overflow:hidden;white-space:nowrap"
                     title="{pa_a}">{pa_a[:14]}{'…' if len(pa_a)>14 else ''}</div>
                <div style="font-size:16px;font-weight:700;color:#639922;line-height:1.3">
                  {sa}<span style="font-size:11px;margin-left:3px">{badge_a}</span>
                </div>
              </div>
              <div style="height:1px;background:rgba(128,128,128,0.15);margin:5px 0"></div>
              <div style="opacity:{opacity_b}">
                <div style="font-size:10px;color:#378ADD;font-weight:500;
                            text-overflow:ellipsis;overflow:hidden;white-space:nowrap"
                     title="{pa_b}">{pa_b[:14]}{'…' if len(pa_b)>14 else ''}</div>
                <div style="font-size:16px;font-weight:700;color:#378ADD;line-height:1.3">
                  {sb}<span style="font-size:11px;margin-left:3px">{badge_b}</span>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ── Row 1: Land cover side-by-side + radar ─────────────────────────────────
    st.subheader("Land cover")
    cl, cr = st.columns(2)

    with cl:
        lc_keys    = ["pct_green_res", "pct_parkland", "pct_urban", "pct_water"]
        lc_labels  = ["Green res.", "Parkland", "Urban", "Water"]
        lc_colors  = ["#639922", "#1D9E75", "#888780", "#378ADD"]

        fig_lc = go.Figure()
        fig_lc.add_trace(go.Bar(
            name=pa_a, x=lc_labels,
            y=[safe(ra[k]) for k in lc_keys],
            marker_color="#639922", opacity=0.85,
        ))
        fig_lc.add_trace(go.Bar(
            name=pa_b, x=lc_labels,
            y=[safe(rb[k]) for k in lc_keys],
            marker_color="#378ADD", opacity=0.85,
        ))
        fig_lc.update_layout(
            barmode="group", height=300,
            margin=dict(t=10, b=10, l=0, r=0),
            yaxis_title="% cover",
            legend=dict(orientation="h", y=1.08),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_lc, use_container_width=True)

    with cr:
        radar_lc = go.Figure()
        radar_lc.add_trace(go.Scatterpolar(
            r=[safe(ra[k]) for k in lc_keys] + [safe(ra[lc_keys[0]])],
            theta=lc_labels + [lc_labels[0]],
            fill="toself", name=pa_a,
            line_color="#639922", fillcolor="rgba(99,153,34,0.2)",
        ))
        radar_lc.add_trace(go.Scatterpolar(
            r=[safe(rb[k]) for k in lc_keys] + [safe(rb[lc_keys[0]])],
            theta=lc_labels + [lc_labels[0]],
            fill="toself", name=pa_b,
            line_color="#378ADD", fillcolor="rgba(55,138,221,0.2)",
        ))
        radar_lc.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            height=300, margin=dict(t=30, b=10, l=30, r=30),
            legend=dict(orientation="h", y=1.12),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(radar_lc, use_container_width=True)

    # ── Row 2: Overlaid population pyramid (% of population) ──────────────────
    st.divider()
    st.subheader("Age & sex profile")
    st.caption("Both areas normalised to % of their own population — directly comparable regardless of size. "
               f"Males left · Females right · {pa_a} filled · {pa_b} outlined.")

    bands_10_pyr = [
        ("0–9",   ["0_4",  "5_9"]),
        ("10–19", ["10_14","15_19"]),
        ("20–29", ["20_24","25_29"]),
        ("30–39", ["30_34","35_39"]),
        ("40–49", ["40_44","45_49"]),
        ("50–59", ["50_54","55_59"]),
        ("60–69", ["60_64","65_69"]),
        ("70–79", ["70_74","75_79"]),
        ("80+",   ["80_84","85_89","90andOver"]),
    ]
    age_labels_pyr = [lbl for lbl, _ in bands_10_pyr]

    def pyr_pct(row, sex):
        """Return each 10-year band as % of total population."""
        total  = max(safe(row.get("pop2020_total", 0)), 1)
        prefix = "m" if sex == "male" else "f"
        return [
            sum(safe(row.get(f"pop2020_{prefix}_{b}", 0)) for b in bnds) / total * 100
            for _, bnds in bands_10_pyr
        ]

    # Build overlaid pyramid — Area A filled, Area B outlined
    fig_pyr = go.Figure()

    # Area A — filled bars, males left
    m_a = pyr_pct(ra, "male")
    f_a = pyr_pct(ra, "female")
    fig_pyr.add_trace(go.Bar(
        name=f"{pa_a} — Male", y=age_labels_pyr, x=[-v for v in m_a],
        orientation="h", marker_color="#639922", opacity=0.6,
        hovertemplate="%{customdata:.2f}%<extra>" + pa_a + " Male</extra>",
        customdata=m_a, legendgroup="a",
    ))
    fig_pyr.add_trace(go.Bar(
        name=f"{pa_a} — Female", y=age_labels_pyr, x=f_a,
        orientation="h", marker_color="#639922", opacity=0.6,
        hovertemplate="%{x:.2f}%<extra>" + pa_a + " Female</extra>",
        legendgroup="a",
    ))

    # Area B — outlined bars, males left
    m_b = pyr_pct(rb, "male")
    f_b = pyr_pct(rb, "female")
    fig_pyr.add_trace(go.Bar(
        name=f"{pa_b} — Male", y=age_labels_pyr, x=[-v for v in m_b],
        orientation="h",
        marker=dict(color="rgba(55,138,221,0.15)",
                    line=dict(color="#378ADD", width=1.5)),
        hovertemplate="%{customdata:.2f}%<extra>" + pa_b + " Male</extra>",
        customdata=m_b, legendgroup="b",
    ))
    fig_pyr.add_trace(go.Bar(
        name=f"{pa_b} — Female", y=age_labels_pyr, x=f_b,
        orientation="h",
        marker=dict(color="rgba(55,138,221,0.15)",
                    line=dict(color="#378ADD", width=1.5)),
        hovertemplate="%{x:.2f}%<extra>" + pa_b + " Female</extra>",
        legendgroup="b",
    ))

    max_pct = max(max(m_a + f_a + m_b + f_b), 0.1)
    tick_step = round(max_pct / 2, 1)

    fig_pyr.update_layout(
        barmode="overlay",
        height=500,
        margin=dict(t=10, b=40, l=10, r=10),
        xaxis=dict(
            tickvals=[-max_pct, -tick_step, 0, tick_step, max_pct],
            ticktext=[f"{max_pct:.1f}%", f"{tick_step:.1f}%", "0",
                      f"{tick_step:.1f}%", f"{max_pct:.1f}%"],
            title="% of population",
            zeroline=True, zerolinecolor="rgba(255,255,255,0.3)", zerolinewidth=1,
        ),
        legend=dict(orientation="h", y=1.03, x=0, font=dict(size=11)),
        bargap=0.1,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_pyr, use_container_width=True)

    # Radar using broad bands for shape comparison
    st.subheader("Age structure — 10-year bands")

    # Group 5-year census bands into 10-year bands for radar
    # bands_10: list of (label, [5yr_band_keys_to_sum])
    bands_10 = [
        ("0–9",   ["0_4",  "5_9"]),
        ("10–19", ["10_14","15_19"]),
        ("20–29", ["20_24","25_29"]),
        ("30–39", ["30_34","35_39"]),
        ("40–49", ["40_44","45_49"]),
        ("50–59", ["50_54","55_59"]),
        ("60–69", ["60_64","65_69"]),
        ("70–79", ["70_74","75_79"]),
        ("80+",   ["80_84","85_89","90andOver"]),
    ]

    def pct_10yr(row, bands):
        """Sum 5-year total bands and express as % of population."""
        total = max(safe(row.get("pop2020_total", 0)), 1)
        return sum(safe(row.get(f"pop2020_t_{b}", 0)) for b in bands) / total * 100

    radar_labels_10 = [lbl for lbl, _ in bands_10]
    radar_vals_a    = [pct_10yr(ra, bnds) for _, bnds in bands_10]
    radar_vals_b    = [pct_10yr(rb, bnds) for _, bnds in bands_10]

    # Close the polygon by repeating first value
    r_a = radar_vals_a + [radar_vals_a[0]]
    r_b = radar_vals_b + [radar_vals_b[0]]
    theta = radar_labels_10 + [radar_labels_10[0]]

    radar_max = round(max(radar_vals_a + radar_vals_b) * 1.2)

    cl_r, cr_r = st.columns([3, 2])
    with cl_r:
        radar_age = go.Figure()
        radar_age.add_trace(go.Scatterpolar(
            r=r_a, theta=theta,
            fill="toself", name=pa_a,
            line_color="#639922", fillcolor="rgba(99,153,34,0.25)",
        ))
        radar_age.add_trace(go.Scatterpolar(
            r=r_b, theta=theta,
            fill="toself", name=pa_b,
            line_color="#378ADD", fillcolor="rgba(55,138,221,0.25)",
        ))
        radar_age.update_layout(
            polar=dict(radialaxis=dict(
                visible=True,
                range=[0, radar_max],
                ticksuffix="%",
                tickfont=dict(size=10),
            )),
            height=380,
            margin=dict(t=50, b=20, l=60, r=60),
            legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center"),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(radar_age, use_container_width=True)

    with cr_r:
        st.markdown("**Difference (A minus B)**")
        for label, vals_a, vals_b in zip(radar_labels_10, radar_vals_a, radar_vals_b):
            diff   = vals_a - vals_b
            arrow  = "▲" if diff > 0.05 else ("▼" if diff < -0.05 else "—")
            colour = "#639922" if diff > 0.05 else ("#378ADD" if diff < -0.05 else "#888")
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:5px 0;border-bottom:0.5px solid rgba(128,128,128,0.2);font-size:12px'>"
                f"<span style='color:#aaa'>{label}</span>"
                f"<span style='color:{colour};font-weight:600'>{arrow} {abs(diff):.1f}pp</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        st.caption("pp = percentage points. ▲ = Area A higher")

    # ── Row 3: Income (only if both have data) ─────────────────────────────────
    st.divider()
    st.subheader("Income distribution")

    inc_keys   = [k for k, _, _ in INC_BANDS]
    inc_labels = [l for _, l, _ in INC_BANDS]
    inc_colors = [c for _, _, c in INC_BANDS]

    # Aggregate rows have income averaged across PAs — treat as valid if non-zero
    _inc_a = ra.get("income_total_workers_thousands", None) if hasattr(ra, "get") else ra["income_total_workers_thousands"]
    _inc_b = rb.get("income_total_workers_thousands", None) if hasattr(rb, "get") else rb["income_total_workers_thousands"]
    has_inc_a = pd.notna(_inc_a) and safe(_inc_a) > 0
    has_inc_b = pd.notna(_inc_b) and safe(_inc_b) > 0

    if not has_inc_a and not has_inc_b:
        st.info("Neither planning area has income data (GHS 2015 covers residential areas only).")
    else:
        def income_pcts(row):
            vals  = [safe(row[k]) for k in inc_keys]
            total = sum(vals) or 1
            return [v / total * 100 for v in vals]

        if not has_inc_a:
            st.caption(f"⚠ No income data for {pa_a} — showing {pa_b} only")
        if not has_inc_b:
            st.caption(f"⚠ No income data for {pa_b} — showing {pa_a} only")

        cl3, cr3 = st.columns([3, 2])

        with cl3:
            # Grouped bar — side-by-side per bracket, easy to compare adjacent bars
            fig_inc = go.Figure()
            if has_inc_a:
                fig_inc.add_trace(go.Bar(
                    name=pa_a, x=inc_labels,
                    y=income_pcts(ra),
                    marker_color="#639922", opacity=0.85,
                    hovertemplate="%{x}: %{y:.1f}%<extra>" + pa_a + "</extra>",
                ))
            if has_inc_b:
                fig_inc.add_trace(go.Bar(
                    name=pa_b, x=inc_labels,
                    y=income_pcts(rb),
                    marker_color="#378ADD", opacity=0.85,
                    hovertemplate="%{x}: %{y:.1f}%<extra>" + pa_b + "</extra>",
                ))
            fig_inc.update_layout(
                barmode="group", height=320,
                margin=dict(t=10, b=40, l=0, r=0),
                yaxis_title="% of workers",
                xaxis_tickangle=-30,
                legend=dict(orientation="h", y=1.08),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_inc, use_container_width=True)

        with cr3:
            # Diverging bar — shows A minus B directly, zero line at centre
            # Immediately shows which area skews higher/lower income
            if has_inc_a and has_inc_b:
                pcts_a = income_pcts(ra)
                pcts_b = income_pcts(rb)
                diffs  = [a - b for a, b in zip(pcts_a, pcts_b)]
                colors = ["#639922" if d >= 0 else "#378ADD" for d in diffs]
                abs_max = max(abs(d) for d in diffs) * 1.3 or 1

                fig_div = go.Figure()
                fig_div.add_trace(go.Bar(
                    y=inc_labels,
                    x=diffs,
                    orientation="h",
                    marker_color=colors,
                    hovertemplate="%{y}: %{x:+.2f}%<extra></extra>",
                ))
                fig_div.add_vline(x=0, line_width=1,
                                  line_color="rgba(128,128,128,0.4)")
                fig_div.update_layout(
                    height=320,
                    margin=dict(t=30, b=40, l=0, r=10),
                    xaxis=dict(
                        title="% point difference (A − B)",
                        range=[-abs_max, abs_max],
                        ticksuffix="%",
                        zeroline=False,
                    ),
                    yaxis=dict(title=""),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                )
                # Annotations for area labels on each side
                fig_div.add_annotation(
                    x=-abs_max, y=1.07, xref="x", yref="paper",
                    text=f"← {pa_b} higher", showarrow=False,
                    font=dict(size=10, color="#378ADD"), xanchor="left",
                )
                fig_div.add_annotation(
                    x=abs_max, y=1.07, xref="x", yref="paper",
                    text=f"{pa_a} higher →", showarrow=False,
                    font=dict(size=10, color="#639922"), xanchor="right",
                )
                st.plotly_chart(fig_div, use_container_width=True)
            else:
                st.caption("Diverging chart requires both areas to have income data.")




# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — GREEN METRICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏆 Green Metrics":
    st.title("Green metrics")
    st.caption(
        "Two composite scores summarising each planning area's green character in a single number. "
        "Use the sidebar filters to focus on residential areas or specific regions."
    )

    gdf_m = dff.dropna(subset=["gub", "lgs"]).copy()

    # ── Explainer cards ────────────────────────────────────────────────────────
    c_ex1, c_ex2 = st.columns(2)
    with c_ex1:
        st.markdown("""
        <div style="background:var(--color-background-secondary,#f5f5f5);
                    border-radius:8px;padding:14px 16px;margin-bottom:4px">
          <div style="font-size:14px;font-weight:600;margin-bottom:6px">
            Green-Urban Balance (GUB)
          </div>
          <div style="font-size:13px;color:var(--color-text-secondary,#555);line-height:1.6">
            <code>(green − urban) / (green + urban)</code><br>
            Ranges from <strong>−1</strong> (entirely urban) to
            <strong>+1</strong> (entirely green). Zero means green and urban
            are equal. Useful for ranking areas on their net green character
            regardless of size.
          </div>
        </div>
        """, unsafe_allow_html=True)
    with c_ex2:
        st.markdown("""
        <div style="background:var(--color-background-secondary,#f5f5f5);
                    border-radius:8px;padding:14px 16px;margin-bottom:4px">
          <div style="font-size:14px;font-weight:600;margin-bottom:6px">
            Liveability Green Score (LGS)
          </div>
          <div style="font-size:13px;color:var(--color-text-secondary,#555);line-height:1.6">
            <code>(parkland + green residential) / (100 − water) × 100</code><br>
            Green cover as a % of <em>habitable</em> land (water excluded).
            Ranges <strong>0–100</strong>. Removes the distortion of coastal
            and reservoir areas having naturally high water percentages.
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Summary scorecards ─────────────────────────────────────────────────────
    res_only = gdf_m[gdf_m["pop2020_total"] > 1000]
    top_gub  = res_only.loc[res_only["gub"].idxmax(), "name"]  if not res_only.empty else "n/a"
    bot_gub  = res_only.loc[res_only["gub"].idxmin(), "name"]  if not res_only.empty else "n/a"
    top_lgs  = res_only.loc[res_only["lgs"].idxmax(), "name"]  if not res_only.empty else "n/a"
    avg_gub  = gdf_m["gub"].mean()
    avg_lgs  = gdf_m["lgs"].mean()

    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    for col, lbl, val, clr in [
        (sc1, "Avg GUB (all)",        f"{avg_gub:+.3f}",  "#639922"),
        (sc2, "Avg LGS (all)",        f"{avg_lgs:.1f}",   "#1D9E75"),
        (sc3, "Greenest area (GUB)",  top_gub,             "#639922"),
        (sc4, "Greenest area (LGS)",  top_lgs,             "#1D9E75"),
        (sc5, "Most urban (GUB)",     bot_gub,             "#888780"),
    ]:
        with col:
            st.markdown(
                f"<div style='background:var(--color-background-secondary,#f5f5f5);"
                f"border-radius:8px;padding:10px 12px;margin-bottom:4px'>"
                f"<div style='font-size:13px;font-weight:500;"
                f"color:var(--color-text-secondary,#555);margin-bottom:6px'>{lbl}</div>"
                f"<div style='font-size:18px;font-weight:700;color:{clr}'>{val}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.divider()

    # ── GUB ranked bar ─────────────────────────────────────────────────────────
    st.subheader("Green-Urban Balance — all planning areas")
    st.caption("+1 = entirely green · −1 = entirely urban · 0 = equal split")

    gub_sorted = gdf_m.sort_values("gub", ascending=True)
    gub_colors = ["#639922" if v >= 0 else "#888780" for v in gub_sorted["gub"]]

    fig_gub = go.Figure(go.Bar(
        y=gub_sorted["name"], x=gub_sorted["gub"],
        orientation="h", marker_color=gub_colors,
        hovertemplate="%{y}<br>GUB: %{x:+.3f}<extra></extra>",
    ))
    fig_gub.add_vline(x=0, line_width=1, line_color="rgba(128,128,128,0.5)")
    fig_gub.update_layout(
        height=max(500, len(gub_sorted) * 16),
        margin=dict(l=10, r=20, t=10, b=30),
        xaxis=dict(title="GUB score", range=[-1, 1],
                   tickvals=[-1, -0.5, 0, 0.5, 1]),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    st.plotly_chart(fig_gub, use_container_width=True)

    st.divider()

    # ── LGS ranked bar ─────────────────────────────────────────────────────────
    st.subheader("Liveability Green Score — all planning areas")
    st.caption("% of habitable land that is green (parkland + green residential, water excluded)")

    lgs_sorted = gdf_m.sort_values("lgs", ascending=True)
    lgs_colors = [
        "#639922" if v >= 60 else ("#1D9E75" if v >= 40 else "#BA7517")
        for v in lgs_sorted["lgs"]
    ]

    fig_lgs = go.Figure(go.Bar(
        y=lgs_sorted["name"], x=lgs_sorted["lgs"],
        orientation="h", marker_color=lgs_colors,
        hovertemplate="%{y}<br>LGS: %{x:.1f}%<extra></extra>",
    ))
    fig_lgs.add_vline(x=lgs_sorted["lgs"].mean(), line_width=1,
                      line_dash="dash", line_color="rgba(128,128,128,0.6)",
                      annotation_text=f"avg {lgs_sorted['lgs'].mean():.1f}%",
                      annotation_position="top right",
                      annotation_font_size=11)
    fig_lgs.update_layout(
        height=max(500, len(lgs_sorted) * 16),
        margin=dict(l=10, r=20, t=10, b=30),
        xaxis=dict(title="LGS (%)", range=[0, 100]),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    st.plotly_chart(fig_lgs, use_container_width=True)

    st.divider()

    # ── GUB vs LGS scatter ─────────────────────────────────────────────────────
    st.subheader("GUB vs LGS — do the two metrics agree?")
    st.caption("Areas near the diagonal agree across both metrics. "
               "Outliers reveal where water distorts the LGS.")

    fig_sc = px.scatter(
        gdf_m.dropna(subset=["gub","lgs","pop2020_total"]),
        x="gub", y="lgs",
        size="pop2020_total", color="region",
        color_discrete_map=REGION_COLORS,
        hover_name="name",
        hover_data={"gub":":.3f","lgs":":.1f",
                    "pct_urban":":.1f","pct_green_total":":.1f",
                    "pop2020_total":":,"},
        labels={"gub":"GUB score","lgs":"LGS (%)","region":"Region"},
        size_max=40,
    )
    fig_sc.update_layout(
        height=440, margin=dict(t=20, b=40),
        xaxis=dict(range=[-1.05, 1.05],
                   tickvals=[-1,-0.5,0,0.5,1], zeroline=True,
                   zerolinecolor="rgba(128,128,128,0.3)"),
        yaxis=dict(range=[0, 105]),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_sc, use_container_width=True)
