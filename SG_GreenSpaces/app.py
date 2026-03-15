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
st.sidebar.markdown("Exploring green space, demographics and income across Singapore's 55 planning areas.")
st.sidebar.markdown("📖 Start with **Introduction** if this is your first visit.")
st.sidebar.divider()

page = st.sidebar.radio(
    "Page",
    ["📖 Introduction", "🗺️ Map", "📊 Land Cover", "👥 Demographics", "💰 Income", "⚖️ Compare", "🏆 Green Metrics", "🔬 Model Assessment"],
)

st.sidebar.divider()
all_regions = sorted(df["region"].dropna().unique())
sel_regions = st.sidebar.multiselect("Filter by region", all_regions, default=all_regions)
show_residential_only = st.sidebar.checkbox("Residential areas only (pop > 1,000)", value=False)
st.sidebar.caption("Filters apply to all pages except 🗺️ Map and 📖 Introduction.")

dff = df[df["region"].isin(sel_regions)].copy()
if show_residential_only:
    dff = dff[dff["pop2020_total"] > 1000]


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 0 — INTRODUCTION
# ══════════════════════════════════════════════════════════════════════════════
if page == "📖 Introduction":
    st.title("Singapore Green Space & Demographics Dashboard")
    st.caption("An analytical tool for exploring the distribution of green space, demographics, and income across Singapore's 55 planning areas.")

    # ── Purpose ───────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Purpose")
    st.markdown("""
    This dashboard was built to answer a central question:

    > **How is green space distributed across Singapore's planning areas, and how does that distribution relate to the demographics and economic characteristics of the people who live there?**

    It combines satellite-derived land cover classification with census demographics and household income data to enable spatial comparison across Singapore's 55 planning areas.
    The dashboard is intended for urban planners, policy researchers, and anyone interested in the relationship between green space and liveability.
    """)

    # ── Provisional finding ──────────────────────────────────────────────────
    st.divider()
    st.markdown(
        "<div style='background:linear-gradient(135deg,rgba(99,153,34,0.12),rgba(29,158,117,0.12));"
        "border-left:4px solid #639922;border-radius:8px;padding:16px 18px;margin:4px 0'>"
        "<div style='font-size:13px;font-weight:600;color:#639922;margin-bottom:6px'>"
        "📊 What the data shows</div>"
        "<div style='font-size:13px;line-height:1.7'>"
        "Preliminary analysis of the 2020 data finds that green space is most concentrated in "
        "Singapore's <strong>North and West regions</strong>, driven by large nature reserves and water catchments. "
        "Among residential areas, <strong>green space coverage correlates positively with income</strong> "
        "(r ≈ +0.50 for LGS vs high earners) — wealthier planning areas tend to have meaningfully more "
        "habitable green space. At the same time, <strong>older populations (aged 60+) tend to live in "
        "greener areas</strong>, suggesting established residential neighbourhoods retain more tree cover "
        "than newer high-density developments."
        "</div></div>",
        unsafe_allow_html=True,
    )

    # ── Data sources ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Data sources")

    col_a, col_b = st.columns(2)
    with col_a:
        for title, body, clr in [
            ("🛰️ Satellite imagery — Google Earth Engine",
             "Multi-spectral satellite imagery of Singapore was exported from Google Earth Engine (GEE). "
             "The imagery captures surface reflectance across visible and near-infrared bands, enabling "
             "the distinction of vegetated, built, and water surfaces.",
             "#1D9E75"),
            ("🤖 Land cover classification — XGBoost + QGIS",
             "A supervised machine learning classifier (XGBoost) was trained on **manually tagged training samples** "
             "drawn directly from the satellite imagery. Each sample was hand-labelled by a human analyst "
             "into one of four classes. The trained model was then applied to the full Singapore extent "
             "inside QGIS, producing a pixel-level land cover raster.",
             "#534AB7"),
        ]:
            st.markdown(
                f"<div style='background:var(--color-background-secondary,#f5f5f5);"
                f"border-radius:8px;padding:14px 16px;margin-bottom:12px'>"
                f"<div style='font-size:14px;font-weight:600;color:{clr};margin-bottom:8px'>{title}</div>"
                f"<div style='font-size:13px;line-height:1.7'>{body}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
    with col_b:
        for title, body, clr in [
            ("👥 Demographics — Singapore Census 2020",
             "Resident population data by planning area, subzone, age group (5-year bands), and sex from the "
             "**2020 Census of Population**, sourced from data.gov.sg. "
             "Covers all 55 planning areas with full age-sex breakdowns.",
             "#BA7517"),
            ("💰 Income — General Household Survey 2015",
             "Gross monthly income from work by planning area from the "
             "**General Household Survey (GHS) 2015**, sourced from data.gov.sg. "
             "Available for 28 of 55 planning areas — industrial zones, military areas, "
             "and non-residential areas are not covered.",
             "#E24B4A"),
        ]:
            st.markdown(
                f"<div style='background:var(--color-background-secondary,#f5f5f5);"
                f"border-radius:8px;padding:14px 16px;margin-bottom:12px'>"
                f"<div style='font-size:14px;font-weight:600;color:{clr};margin-bottom:8px'>{title}</div>"
                f"<div style='font-size:13px;line-height:1.7'>{body}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Land cover classes ────────────────────────────────────────────────────
    st.divider()
    st.subheader("Land cover classification")
    st.markdown("The XGBoost classifier assigns every pixel to one of four classes. "
                "Understanding how these are defined is important for interpreting all charts in this dashboard.")

    lc_col1, lc_col2, lc_col3, lc_col4 = st.columns(4)
    for col, clr, name, defn in [
        (lc_col1, "#639922", "🌿 Green residential",
         "Vegetated land within residential areas — gardens, street trees, verges, and private green cover. "
         "Reflects incidental green space rather than managed public parks."),
        (lc_col2, "#1D9E75", "🏞️ Parkland",
         "Managed public green space — nature reserves, parks, recreational fields, and forested areas. "
         "Higher ecological and recreational value than green residential cover."),
        (lc_col3, "#888780", "🏙️ Urban",
         "Built surfaces — roads, buildings, industrial areas, and impervious surfaces. "
         "Includes all non-vegetated, non-water land cover."),
        (lc_col4, "#378ADD", "💧 Water",
         "Open water bodies — reservoirs, rivers, coastal water, and canals. "
         "Excluded from the LGS denominator since it is not habitable land."),
    ]:
        with col:
            st.markdown(
                f"<div style='background:var(--color-background-secondary,#f5f5f5);"
                f"border-left:4px solid {clr};border-radius:8px;"
                f"padding:14px 16px;height:100%'>"
                f"<div style='font-size:14px;font-weight:700;color:{clr};margin-bottom:8px'>{name}</div>"
                f"<div style='font-size:12px;line-height:1.6'>{defn}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Composite metrics ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("Composite green metrics")
    st.markdown("Two summary scores are computed from the classified land cover to enable single-number comparison:")

    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.markdown(
            "<div style='background:var(--color-background-secondary,#f5f5f5);"
            "border-radius:8px;padding:14px 16px'>"
            "<div style='font-size:14px;font-weight:600;color:#639922;margin-bottom:8px'>"
            "Green-Urban Balance (GUB)</div>"
            "<div style='font-size:13px;line-height:1.7'>"
            "<code>(green − urban) / (green + urban)</code><br><br>"
            "Ranges from <strong>−1</strong> (entirely urban) to <strong>+1</strong> (entirely green). "
            "Zero means green and urban cover are equal. Avoids division instability from areas with near-zero urban cover."
            "</div></div>",
            unsafe_allow_html=True,
        )
    with m_col2:
        st.markdown(
            "<div style='background:var(--color-background-secondary,#f5f5f5);"
            "border-radius:8px;padding:14px 16px'>"
            "<div style='font-size:14px;font-weight:600;color:#1D9E75;margin-bottom:8px'>"
            "Liveability Green Score (LGS)</div>"
            "<div style='font-size:13px;line-height:1.7'>"
            "<code>(parkland + green residential) / (100 − water) × 100</code><br><br>"
            "Green cover as a percentage of <em>habitable</em> land. "
            "Removing water from the denominator prevents coastal and reservoir-adjacent areas "
            "from appearing artificially low in green coverage."
            "</div></div>",
            unsafe_allow_html=True,
        )

    # ── Navigation guide ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("How to use this dashboard")
    st.markdown("""
    The pages are designed to be explored in order, building from spatial overview to thematic analysis to synthesis:

    | Page | Purpose |
    |---|---|
    | 🗺️ **Map** | Start here — see the classified land cover spatially and hover any area for a quick stats summary |
    | 📊 **Land Cover** | Explore how green, urban, and water cover varies across all 55 planning areas and five regions |
    | 👥 **Demographics** | Understand the age and sex profile of each planning area and how it relates to green cover |
    | 💰 **Income** | Examine whether wealthier areas have more or better-quality green space |
    | ⚖️ **Compare** | Select any two planning areas or regions for a side-by-side comparison |
    | 🏆 **Green Metrics** | See all 55 areas ranked by GUB and LGS — the dashboard's central analytical output |
    | 🔬 **Model Assessment** | Understand the accuracy and limitations of the ML classifier behind all green space data |

    The **sidebar filters** (region and residential areas only) apply to all pages except Map and Introduction.
    """)

    # ── Important caveats ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("Important caveats")
    st.warning(
        "**Classification uncertainty** — the land cover data is produced by an ML model trained on "
        "manually tagged samples. All classifications carry uncertainty, particularly at class boundaries "
        "(e.g. dense street trees may be classified as parkland rather than green residential). "
        "See the **🔬 Model Assessment** page for full accuracy metrics and known limitations.\n\n"
        "**Income data vintage** — GHS 2015 income data is a decade old. Income distributions have likely "
        "shifted, particularly in areas with significant development since 2015.\n\n"
        "**Census 2020 context** — demographic data reflects the population at the time of the 2020 Census, "
        "prior to post-pandemic population changes."
    )

# ═══════════════════════════════════════════════════════════════════════════# ==============================================================================
# PAGE 1 — MAP
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Map":
    st.title("Land cover map")
    st.caption("Classified land cover overlaid with planning area boundaries. Hover over a planning area to see its stats in the panel (top-right).")

    rgba, bounds = load_raster_preview()

    # Derive map constraints from raster bounds when available
    if bounds is not None:
        map_center = [(bounds.bottom + bounds.top) / 2,
                      (bounds.left  + bounds.right) / 2]
        sw = [bounds.bottom - 0.02, bounds.left  - 0.02]
        ne = [bounds.top   + 0.02, bounds.right + 0.02]
    else:
        map_center = [1.3521, 103.8198]
        sw = [1.18, 103.58]
        ne = [1.52, 104.10]

    m = folium.Map(
        location=map_center,
        zoom_start=11,
        min_zoom=10,
        max_zoom=16,
        max_bounds=True,
        tiles=None,
    )
    # Restrict pan to raster extent without changing initial zoom
    m.options["maxBounds"] = [sw, ne]
    m.options["maxBoundsViscosity"] = 1.0

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

        # ── Inject panel HTML, tooltip CSS, clamp JS, hover handler ───────────
        _panel_and_js = """
        <style>
          #stats-panel {
            position:absolute; top:12px; right:12px; z-index:9999;
            background:rgba(15,15,15,0.9); color:#fff;
            padding:14px 16px; border-radius:10px;
            font-family:sans-serif; font-size:12px; line-height:1.7;
            min-width:200px; max-width:220px;
            box-shadow:0 4px 20px rgba(0,0,0,0.6);
            pointer-events:none;
          }
          #stats-panel .sp-title  {font-size:14px;font-weight:700;margin-bottom:2px;}
          #stats-panel .sp-region {font-size:11px;color:#aaa;margin-bottom:10px;}
          #stats-panel .sp-row    {display:flex;justify-content:space-between;padding:3px 0;border-bottom:0.5px solid rgba(255,255,255,0.08);}
          #stats-panel .sp-row:last-child{border-bottom:none;}
          #stats-panel .sp-label  {color:#bbb;}
          #stats-panel .sp-val    {font-weight:600;color:#fff;}
          #stats-panel .sp-divider{border:none;border-top:0.5px solid rgba(255,255,255,0.15);margin:8px 0;}
          #stats-panel .sp-section{font-size:10px;color:#888;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;}
          .leaflet-tooltip {
            background:rgba(15,15,15,0.88) !important; border:none !important;
            border-radius:8px !important; box-shadow:0 4px 16px rgba(0,0,0,0.5) !important;
            padding:8px 12px !important; color:#fff !important;
            font-size:12px !important; width:260px !important;
            max-width:260px !important; min-width:260px !important;
            line-height:1.6 !important;
          }
          .leaflet-tooltip::before {display:none !important;}
          .leaflet-tooltip table  {border:none !important;border-collapse:collapse !important;width:100% !important;table-layout:fixed !important;}
          .leaflet-tooltip th     {color:#bbb !important;font-weight:400 !important;font-size:11px !important;padding:2px 8px 2px 0 !important;border:none !important;white-space:nowrap !important;text-align:left !important;vertical-align:middle !important;width:45% !important;}
          .leaflet-tooltip td     {color:#fff !important;font-weight:600 !important;font-size:12px !important;padding:2px 0 !important;border:none !important;text-align:right !important;white-space:nowrap !important;overflow:hidden !important;text-overflow:ellipsis !important;width:55% !important;}
          .leaflet-tooltip tr     {border:none !important;}
        </style>

        <div id="stats-panel">
          <div class="sp-title">Singapore overall</div>
          <div class="sp-region">Hover a planning area to update</div>
          <div class="sp-section">Land cover (avg)</div>
          <div class="sp-row"><span class="sp-label">Green res.</span><span class="sp-val">20.6%</span></div>
          <div class="sp-row"><span class="sp-label">Parkland</span><span class="sp-val">37.1%</span></div>
          <div class="sp-row"><span class="sp-label">Urban</span><span class="sp-val">32.5%</span></div>
          <div class="sp-row"><span class="sp-label">Water</span><span class="sp-val">9.7%</span></div>
          <hr class="sp-divider">
          <div class="sp-section">Green metrics (avg)</div>
          <div class="sp-row"><span class="sp-label">GUB</span><span class="sp-val">+0.282</span></div>
          <div class="sp-row"><span class="sp-label">LGS</span><span class="sp-val">64.1%</span></div>
        </div>

        <div style="position:absolute;bottom:30px;left:10px;z-index:9999;
                    background:rgba(0,0,0,0.7);color:#fff;padding:10px 14px;
                    border-radius:8px;font-size:12px;line-height:1.8">
          <b>Land cover</b><br>
          <span style="color:#639922">&#9632;</span> Green residential<br>
          <span style="color:#1D9E75">&#9632;</span> Parkland<br>
          <span style="color:#888780">&#9632;</span> Urban<br>
          <span style="color:#378ADD">&#9632;</span> Water
        </div>

        <script>
        (function() {
          var DATA  = {"BEDOK": {"name": "Bedok", "region": "EAST", "pop": 276990, "gr": 26.7, "pk": 25.8, "ur": 41.0, "wa": 6.5, "gub": 0.123, "lgs": 56.1}, "BOON LAY": {"name": "Boon Lay", "region": "WEST", "pop": 40, "gr": 19.3, "pk": 19.2, "ur": 48.6, "wa": 12.9, "gub": -0.116, "lgs": 44.2}, "BUKIT BATOK": {"name": "Bukit Batok", "region": "WEST", "pop": 158030, "gr": 23.0, "pk": 46.8, "ur": 28.6, "wa": 1.6, "gub": 0.419, "lgs": 70.9}, "BUKIT MERAH": {"name": "Bukit Merah", "region": "CENTRAL", "pop": 151250, "gr": 21.1, "pk": 33.3, "ur": 39.8, "wa": 5.8, "gub": 0.155, "lgs": 57.7}, "BUKIT PANJANG": {"name": "Bukit Panjang", "region": "WEST", "pop": 138270, "gr": 20.0, "pk": 56.8, "ur": 22.0, "wa": 1.3, "gub": 0.555, "lgs": 77.8}, "BUKIT TIMAH": {"name": "Bukit Timah", "region": "CENTRAL", "pop": 77860, "gr": 32.6, "pk": 42.5, "ur": 24.3, "wa": 0.6, "gub": 0.511, "lgs": 75.6}, "CENTRAL WATER CATCHMENT": {"name": "Central Water Catchment", "region": "NORTH", "pop": null, "gr": 4.0, "pk": 75.5, "ur": 3.3, "wa": 17.2, "gub": 0.92, "lgs": 96.0}, "CHANGI": {"name": "Changi", "region": "EAST", "pop": 1850, "gr": 14.4, "pk": 33.9, "ur": 40.8, "wa": 10.9, "gub": 0.084, "lgs": 54.2}, "CHOA CHU KANG": {"name": "Choa Chu Kang", "region": "WEST", "pop": 192070, "gr": 33.8, "pk": 21.1, "ur": 43.1, "wa": 1.9, "gub": 0.12, "lgs": 56.0}, "CLEMENTI": {"name": "Clementi", "region": "WEST", "pop": 91990, "gr": 28.1, "pk": 28.2, "ur": 38.5, "wa": 5.2, "gub": 0.188, "lgs": 59.4}, "HOUGANG": {"name": "Hougang", "region": "NORTH-EAST", "pop": 227560, "gr": 27.4, "pk": 24.0, "ur": 44.8, "wa": 3.8, "gub": 0.069, "lgs": 53.4}, "JURONG EAST": {"name": "Jurong East", "region": "WEST", "pop": 78600, "gr": 19.4, "pk": 19.6, "ur": 35.6, "wa": 25.4, "gub": 0.046, "lgs": 52.3}, "JURONG WEST": {"name": "Jurong West", "region": "WEST", "pop": 262730, "gr": 28.6, "pk": 22.5, "ur": 44.9, "wa": 3.9, "gub": 0.065, "lgs": 53.2}, "PASIR RIS": {"name": "Pasir Ris", "region": "EAST", "pop": 147110, "gr": 20.4, "pk": 35.6, "ur": 34.6, "wa": 9.4, "gub": 0.236, "lgs": 61.8}, "PIONEER": {"name": "Pioneer", "region": "WEST", "pop": 80, "gr": 18.9, "pk": 12.5, "ur": 53.5, "wa": 15.1, "gub": -0.26, "lgs": 37.0}, "PUNGGOL": {"name": "Punggol", "region": "NORTH-EAST", "pop": 174450, "gr": 26.6, "pk": 37.9, "ur": 28.8, "wa": 6.7, "gub": 0.383, "lgs": 69.1}, "QUEENSTOWN": {"name": "Queenstown", "region": "CENTRAL", "pop": 95930, "gr": 18.4, "pk": 30.8, "ur": 39.6, "wa": 11.2, "gub": 0.108, "lgs": 55.4}, "SELETAR": {"name": "Seletar", "region": "NORTH-EAST", "pop": 300, "gr": 14.8, "pk": 50.4, "ur": 25.9, "wa": 8.9, "gub": 0.431, "lgs": 71.6}, "SEMBAWANG": {"name": "Sembawang", "region": "NORTH", "pop": 102640, "gr": 24.0, "pk": 28.9, "ur": 40.7, "wa": 6.4, "gub": 0.13, "lgs": 56.5}, "SENGKANG": {"name": "Sengkang", "region": "NORTH-EAST", "pop": 249370, "gr": 29.4, "pk": 34.5, "ur": 32.9, "wa": 3.2, "gub": 0.32, "lgs": 66.0}, "SERANGOON": {"name": "Serangoon", "region": "NORTH-EAST", "pop": 116900, "gr": 27.8, "pk": 15.4, "ur": 53.7, "wa": 3.2, "gub": -0.108, "lgs": 44.6}, "KALLANG": {"name": "Kallang", "region": "CENTRAL", "pop": 101290, "gr": 24.6, "pk": 22.4, "ur": 40.7, "wa": 12.3, "gub": 0.072, "lgs": 53.6}, "LIM CHU KANG": {"name": "Lim Chu Kang", "region": "NORTH", "pop": 110, "gr": 11.7, "pk": 69.4, "ur": 8.8, "wa": 10.1, "gub": 0.804, "lgs": 90.2}, "NORTH-EASTERN ISLANDS": {"name": "North-Eastern Islands", "region": "NORTH-EAST", "pop": 50, "gr": 8.6, "pk": 53.3, "ur": 15.4, "wa": 22.6, "gub": 0.602, "lgs": 80.0}, "NOVENA": {"name": "Novena", "region": "CENTRAL", "pop": 49330, "gr": 27.9, "pk": 43.2, "ur": 26.9, "wa": 2.0, "gub": 0.451, "lgs": 72.6}, "SIMPANG": {"name": "Simpang", "region": "NORTH", "pop": null, "gr": 5.6, "pk": 59.3, "ur": 2.8, "wa": 32.3, "gub": 0.917, "lgs": 95.9}, "SOUTHERN ISLANDS": {"name": "Southern Islands", "region": "CENTRAL", "pop": 1940, "gr": 15.9, "pk": 52.3, "ur": 14.5, "wa": 17.2, "gub": 0.649, "lgs": 82.4}, "SUNGEI KADUT": {"name": "Sungei Kadut", "region": "NORTH", "pop": 750, "gr": 15.5, "pk": 46.7, "ur": 23.2, "wa": 14.6, "gub": 0.457, "lgs": 72.8}, "TOA PAYOH": {"name": "Toa Payoh", "region": "CENTRAL", "pop": 121850, "gr": 29.4, "pk": 24.1, "ur": 43.8, "wa": 2.6, "gub": 0.1, "lgs": 54.9}, "TUAS": {"name": "Tuas", "region": "WEST", "pop": 70, "gr": 11.6, "pk": 19.1, "ur": 43.3, "wa": 25.9, "gub": -0.17, "lgs": 41.4}, "WESTERN ISLANDS": {"name": "Western Islands", "region": "WEST", "pop": 10, "gr": 11.6, "pk": 33.7, "ur": 35.3, "wa": 19.4, "gub": 0.124, "lgs": 56.2}, "WESTERN WATER CATCHMENT": {"name": "Western Water Catchment", "region": "WEST", "pop": 640, "gr": 11.4, "pk": 65.7, "ur": 10.9, "wa": 12.0, "gub": 0.752, "lgs": 87.6}, "WOODLANDS": {"name": "Woodlands", "region": "NORTH", "pop": 255130, "gr": 30.2, "pk": 26.2, "ur": 39.5, "wa": 4.2, "gub": 0.176, "lgs": 58.9}, "RIVER VALLEY": {"name": "River Valley", "region": "CENTRAL", "pop": 10070, "gr": 36.3, "pk": 29.6, "ur": 33.0, "wa": 1.1, "gub": 0.333, "lgs": 66.6}, "ROCHOR": {"name": "Rochor", "region": "CENTRAL", "pop": 13120, "gr": 17.7, "pk": 9.1, "ur": 66.1, "wa": 7.2, "gub": -0.423, "lgs": 28.9}, "SINGAPORE RIVER": {"name": "Singapore River", "region": "CENTRAL", "pop": 3260, "gr": 21.9, "pk": 11.7, "ur": 51.0, "wa": 15.4, "gub": -0.206, "lgs": 39.7}, "STRAITS VIEW": {"name": "Straits View", "region": "CENTRAL", "pop": null, "gr": 10.8, "pk": 50.9, "ur": 9.3, "wa": 29.0, "gub": 0.738, "lgs": 86.9}, "CHANGI BAY": {"name": "Changi Bay", "region": "EAST", "pop": null, "gr": 19.2, "pk": 51.8, "ur": 23.6, "wa": 5.5, "gub": 0.501, "lgs": 75.1}, "MARINE PARADE": {"name": "Marine Parade", "region": "CENTRAL", "pop": 46220, "gr": 23.7, "pk": 37.0, "ur": 37.0, "wa": 2.3, "gub": 0.243, "lgs": 62.1}, "DOWNTOWN CORE": {"name": "Downtown Core", "region": "CENTRAL", "pop": 3190, "gr": 16.2, "pk": 12.7, "ur": 45.3, "wa": 25.8, "gub": -0.221, "lgs": 38.9}, "MARINA EAST": {"name": "Marina East", "region": "CENTRAL", "pop": null, "gr": 8.8, "pk": 51.9, "ur": 16.8, "wa": 22.4, "gub": 0.566, "lgs": 78.2}, "MARINA SOUTH": {"name": "Marina South", "region": "CENTRAL", "pop": null, "gr": 16.2, "pk": 54.9, "ur": 9.6, "wa": 19.3, "gub": 0.762, "lgs": 88.1}, "MUSEUM": {"name": "Museum", "region": "CENTRAL", "pop": 510, "gr": 22.1, "pk": 41.9, "ur": 33.2, "wa": 2.7, "gub": 0.317, "lgs": 65.8}, "NEWTON": {"name": "Newton", "region": "CENTRAL", "pop": 8260, "gr": 28.3, "pk": 47.1, "ur": 23.8, "wa": 0.9, "gub": 0.52, "lgs": 76.1}, "ORCHARD": {"name": "Orchard", "region": "CENTRAL", "pop": 920, "gr": 24.1, "pk": 14.6, "ur": 51.5, "wa": 9.7, "gub": -0.142, "lgs": 42.9}, "OUTRAM": {"name": "Outram", "region": "CENTRAL", "pop": 18340, "gr": 16.6, "pk": 26.4, "ur": 51.3, "wa": 5.7, "gub": -0.088, "lgs": 45.6}, "TAMPINES": {"name": "Tampines", "region": "EAST", "pop": 259900, "gr": 25.2, "pk": 32.1, "ur": 37.8, "wa": 4.9, "gub": 0.205, "lgs": 60.3}, "TANGLIN": {"name": "Tanglin", "region": "CENTRAL", "pop": 21810, "gr": 31.6, "pk": 50.0, "ur": 17.9, "wa": 0.4, "gub": 0.64, "lgs": 81.9}, "TENGAH": {"name": "Tengah", "region": "WEST", "pop": 10, "gr": 12.5, "pk": 49.6, "ur": 28.6, "wa": 9.2, "gub": 0.369, "lgs": 68.4}, "MANDAI": {"name": "Mandai", "region": "NORTH", "pop": 2090, "gr": 10.4, "pk": 79.9, "ur": 9.1, "wa": 0.6, "gub": 0.817, "lgs": 90.8}, "BISHAN": {"name": "Bishan", "region": "CENTRAL", "pop": 87320, "gr": 24.7, "pk": 30.4, "ur": 41.7, "wa": 3.1, "gub": 0.138, "lgs": 56.9}, "ANG MO KIO": {"name": "Ang Mo Kio", "region": "CENTRAL", "pop": 162280, "gr": 25.3, "pk": 38.7, "ur": 33.3, "wa": 2.6, "gub": 0.316, "lgs": 65.7}, "GEYLANG": {"name": "Geylang", "region": "CENTRAL", "pop": 110110, "gr": 27.1, "pk": 16.7, "ur": 52.4, "wa": 3.7, "gub": -0.089, "lgs": 45.5}, "PAYA LEBAR": {"name": "Paya Lebar", "region": "EAST", "pop": 40, "gr": 14.1, "pk": 58.3, "ur": 21.4, "wa": 6.1, "gub": 0.544, "lgs": 77.1}, "YISHUN": {"name": "Yishun", "region": "NORTH", "pop": 221610, "gr": 19.9, "pk": 37.1, "ur": 25.8, "wa": 17.2, "gub": 0.377, "lgs": 68.8}};
          var panel = document.getElementById('stats-panel');

          function row(l,v) {
            return '<div class="sp-row"><span class="sp-label">'+l+'</span><span class="sp-val">'+v+'</span></div>';
          }
          function fmtPop(v)  { return (v&&v>0)?Number(v).toLocaleString():'n/a'; }
          function fmtGub(v)  { if(v===null||v===undefined)return 'n/a'; return(v>=0?'+':'')+parseFloat(v).toFixed(3); }
          function fmtN(v,dp,s) { return(v!==null&&v!==undefined)?parseFloat(v).toFixed(dp)+s:'n/a'; }

          function updatePanel(d) {
            if(!d||!panel)return;
            panel.innerHTML=
              '<div class="sp-title">'+(d.name||'')+'</div>'+
              '<div class="sp-region">'+(d.region||'')+' Region</div>'+
              '<div class="sp-section">Population</div>'+
              row('Residents',fmtPop(d.pop))+
              '<hr class="sp-divider">'+
              '<div class="sp-section">Land cover</div>'+
              row('Green res.',fmtN(d.gr,1,'%'))+
              row('Parkland',  fmtN(d.pk,1,'%'))+
              row('Urban',     fmtN(d.ur,1,'%'))+
              row('Water',     fmtN(d.wa,1,'%'))+
              '<hr class="sp-divider">'+
              '<div class="sp-section">Green metrics</div>'+
              row('GUB',fmtGub(d.gub))+
              row('LGS',fmtN(d.lgs,1,'%'));
          }

          // Update panel from tooltip content (hover)
          var obs = new MutationObserver(function() {
            var tt = document.querySelector('.leaflet-tooltip');
            if(!tt)return;
            var trs = tt.querySelectorAll('tr');
            var name = null;
            trs.forEach(function(r) {
              var cells = r.querySelectorAll('th,td');
              if(cells.length>=2 && cells[0].textContent.trim()==='Area')
                name = cells[1].textContent.trim().toUpperCase();
            });
            if(name && DATA[name]) updatePanel(DATA[name]);
          });
          obs.observe(document.body, {childList:true,subtree:true,characterData:true});

          // Tooltip clamp
          var pad=16;
          function clamp(tt) {
            if(!tt||!tt.offsetWidth)return;
            var r=tt.getBoundingClientRect();
            var vW=document.documentElement.clientWidth;
            var vH=document.documentElement.clientHeight;
            var dx=0,dy=0;
            if(r.right >vW-pad) dx=vW-pad-r.right;
            if(r.bottom>vH-pad) dy=vH-pad-r.bottom;
            if(r.left+dx<pad)   dx=pad-r.left;
            if(r.top +dy<pad)   dy=pad-r.top;
            if(dx===0&&dy===0)return;
            var cur=tt.style.transform||'';
            var parts=cur.match(/translate3d[(](-?[0-9.]+)px,[^-0-9]*(-?[0-9.]+)px/);
            if(parts){
              tt.style.transform='translate3d('+(parseFloat(parts[1])+dx)+'px,'+(parseFloat(parts[2])+dy)+'px,0px)';
            }
          }
          var obs2=new MutationObserver(function(muts){
            muts.forEach(function(mu){
              mu.addedNodes.forEach(function(n){
                if(n.classList&&n.classList.contains('leaflet-tooltip'))
                  setTimeout(function(){clamp(n);},0);
              });
              if(mu.type==='attributes'&&mu.target.classList&&
                 mu.target.classList.contains('leaflet-tooltip'))
                setTimeout(function(){clamp(mu.target);},0);
            });
          });
          obs2.observe(document.body,{childList:true,subtree:true,attributes:true,attributeFilter:['style']});
        })();
        </script>
        """
        m.get_root().html.add_child(folium.Element(_panel_and_js))

    else:
        st.info("No shapefile found. Place your planning area .geojson in the app folder.")

    folium.LayerControl().add_to(m)
    map_data = st_folium(m, width="100%", height=800, returned_objects=[])





# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — LAND COVER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Land Cover":
    st.title("Land cover by planning area")
    st.caption("Explore the spatial distribution of green, urban, and water cover. For composite scores ranking all areas, see 🏆 Green Metrics.")

    # ── 1. Story-driven scorecards ─────────────────────────────────────────────
    st.divider()
    greenest  = dff.loc[dff["pct_green_total"].idxmax()]
    most_urban= dff.loc[dff["pct_urban"].idxmax()]
    most_park = dff.loc[dff["pct_parkland"].idxmax()]
    most_water= dff.loc[dff["pct_water"].idxmax()]

    sc1, sc2, sc3, sc4, sc5, sc6 = st.columns(6)
    for col, lbl, name_val, stat_val, clr in [
        (sc1, "Greenest area",       greenest["name"],
              f"{greenest['pct_green_total']:.1f}% green",    "#639922"),
        (sc2, "Most urban area",     most_urban["name"],
              f"{most_urban['pct_urban']:.1f}% urban",        "#888780"),
        (sc3, "Most parkland",       most_park["name"],
              f"{most_park['pct_parkland']:.1f}% parkland",   "#1D9E75"),
        (sc4, "Most water coverage", most_water["name"],
              f"{most_water['pct_water']:.1f}% water",        "#378ADD"),
        (sc5, "Avg GUB (filtered)",
              f"{dff['gub'].mean():+.3f}",
              f"Range {dff['gub'].min():+.2f} → {dff['gub'].max():+.2f}", "#BA7517"),
        (sc6, "Avg LGS (filtered)",
              f"{dff['lgs'].mean():.1f}%",
              f"Range {dff['lgs'].min():.0f}% → {dff['lgs'].max():.0f}%", "#534AB7"),
    ]:
        with col:
            st.markdown(
                f"<div style='background:var(--color-background-secondary,#f5f5f5);"
                f"border-radius:8px;padding:10px 12px;margin-bottom:4px'>"
                f"<div style='font-size:13px;font-weight:500;"
                f"color:var(--color-text-secondary,#555);margin-bottom:6px'>{lbl}</div>"
                f"<div style='font-size:18px;font-weight:700;color:{clr}'>{name_val}</div>"
                f"<div style='font-size:11px;color:#888;margin-top:3px'>{stat_val}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown("#### 📊 Planning area detail")
    st.caption("The primary view — sort and filter all 55 planning areas by any land cover class or green metric.")

    # ── 3. Sort + region filter + reference line ───────────────────────────────
    ctrl1, ctrl2 = st.columns([2, 3])
    with ctrl1:
        sort_col = st.selectbox("Sort by", list(LC_LABELS.values()) + ["GUB score", "LGS", "Name"])
    with ctrl2:
        all_regions  = sorted(dff["region"].dropna().unique())
        sel_regions_lc = st.multiselect("Filter by region", all_regions, default=all_regions,
                                         key="lc_region_filter")

    sort_key = {v: k for k, v in LC_LABELS.items()}
    sort_key.update({"GUB score": "gub", "LGS": "lgs", "Name": "name"})
    skey    = sort_key[sort_col]
    plot_df = dff[dff["region"].isin(sel_regions_lc)].sort_values(
        skey, ascending=(skey == "name")
    )

    # SG average for the sort column (for reference line)
    if skey in dff.columns and skey != "name":
        sg_ref = dff[skey].mean()
    else:
        sg_ref = None

    # ── Main stacked bar ───────────────────────────────────────────────────────
    fig = go.Figure()
    for key, label in LC_LABELS.items():
        fig.add_trace(go.Bar(
            y=plot_df["name"], x=plot_df[key],
            orientation="h", name=label, marker_color=LC_COLORS[key],
            hovertemplate=f"<b>%{{y}}</b><br>{label}: %{{x:.1f}}%<extra></extra>",
        ))

    # Reference line for sort column average
    if sg_ref is not None and skey not in ("gub", "lgs", "name"):
        fig.add_vline(
            x=sg_ref, line_width=1.5, line_dash="dash",
            line_color="rgba(255,255,255,0.5)",
            annotation_text=f"SG avg {sg_ref:.1f}%",
            annotation_position="top",
            annotation_font=dict(size=10, color="rgba(255,255,255,0.7)"),
        )

    fig.update_layout(
        barmode="stack",
        height=max(400, len(plot_df) * 16),
        margin=dict(l=10, r=20, t=10, b=30),
        xaxis=dict(title="% of planning area", range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── 2. GUB context strip ───────────────────────────────────────────────────
    st.subheader("Green-Urban Balance (GUB) by planning area")
    st.caption("Same sort and region filter applied. "
               "Green bars = net green · Grey bars = net urban · Dashed line = Singapore average.")

    gub_df = plot_df.dropna(subset=["gub"]).copy()
    sg_gub_avg = dff["gub"].mean()

    fig_gub = go.Figure(go.Bar(
        y=gub_df["name"],
        x=gub_df["gub"],
        orientation="h",
        marker_color=["#639922" if v >= 0 else "#888780" for v in gub_df["gub"]],
        hovertemplate="<b>%{y}</b><br>GUB: %{x:+.3f}<extra></extra>",
    ))
    fig_gub.add_vline(x=0,         line_width=1,   line_color="rgba(128,128,128,0.4)")
    fig_gub.add_vline(x=sg_gub_avg, line_width=1.5, line_dash="dash",
                      line_color="rgba(255,255,255,0.5)",
                      annotation_text=f"SG avg {sg_gub_avg:+.3f}",
                      annotation_position="top",
                      annotation_font=dict(size=10, color="rgba(255,255,255,0.7)"))
    fig_gub.update_layout(
        height=max(400, len(gub_df) * 16),
        margin=dict(l=10, r=20, t=10, b=30),
        xaxis=dict(title="GUB score", range=[-1, 1],
                   tickvals=[-1, -0.5, 0, 0.5, 1]),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    st.plotly_chart(fig_gub, use_container_width=True)

    st.markdown(
        "<div style='background:rgba(239,159,39,0.1);border-left:3px solid #EF9F27;"
        "border-radius:6px;padding:10px 14px;margin:4px 0;font-size:12px;line-height:1.6'>"
        "⚠️ <strong>Model note:</strong> The XGBoost classifier has a Parkland recall of 0.75 — "
        "2 of 8 Parkland test samples were misclassified as Green residential. "
        "Parkland % may be slightly underestimated; GUB and LGS aggregate both green classes "
        "and are less sensitive to this. See 🔬 <strong>Model Assessment</strong> for full details."
        "</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── 4. Region-aggregated stacked bar (replaces treemap) ────────────────────
    st.markdown("#### 🌏 Regional summary")
    st.caption("Population-weighted average % per region — shows structural differences "
               "without being skewed by the physical size of individual planning areas.")

    reg_lc = (
        dff.groupby("region")
        .apply(lambda g: pd.Series({
            k: (g[k].fillna(0) * g["pop2020_total"].fillna(0)).sum()
               / g["pop2020_total"].fillna(0).sum()
            for k in LC_LABELS
        }))
        .reset_index()
    )
    # Sort regions by pct_green_total descending
    reg_lc["pct_green_total"] = reg_lc["pct_green_res"] + reg_lc["pct_parkland"]
    reg_lc = reg_lc.sort_values("pct_green_total", ascending=False)

    fig_reg = go.Figure()
    for key, label in LC_LABELS.items():
        fig_reg.add_trace(go.Bar(
            x=reg_lc["region"], y=reg_lc[key],
            name=label, marker_color=LC_COLORS[key],
            hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y:.1f}}%<extra></extra>",
        ))
    fig_reg.update_layout(
        barmode="stack", height=300,
        margin=dict(l=10, r=10, t=10, b=40),
        yaxis=dict(title="% cover (pop-weighted avg)", range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )

    # Overlay SG average lines for each class
    for key, label in LC_LABELS.items():
        sg_class_avg = dff[key].mean()
        fig_reg.add_hline(
            y=sg_class_avg, line_dash="dot", line_width=1,
            line_color=LC_COLORS[key],
            opacity=0.4,
        )

    cl_reg, cr_reg = st.columns([3, 2])
    with cl_reg:
        st.plotly_chart(fig_reg, use_container_width=True)

    with cr_reg:
        # Concentration callout — auto-generated insight text
        st.markdown("**Regional highlights**")
        for key, label in LC_LABELS.items():
            max_reg = reg_lc.loc[reg_lc[key].idxmax(), "region"].title()
            min_reg = reg_lc.loc[reg_lc[key].idxmin(), "region"].title()
            max_val = reg_lc[key].max()
            min_val = reg_lc[key].min()
            st.markdown(
                f"<div style='padding:6px 0;border-bottom:0.5px solid "
                f"rgba(128,128,128,0.2);font-size:12px'>"
                f"<span style='color:{LC_COLORS[key]};font-weight:600'>{label}</span>"
                f" — highest in <strong>{max_reg}</strong> ({max_val:.1f}%), "
                f"lowest in <strong>{min_reg}</strong> ({min_val:.1f}%)"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Migrated from Green vs Urban: parkland vs residential by region ────────
    st.divider()
    st.subheader("Parkland vs green residential by region")
    st.caption("Breakdown of green cover type — parkland is managed public space, "
               "green residential is gardens and vegetated private land.")

    gb = (
        dff.groupby("region")[["pct_green_res", "pct_parkland"]]
        .mean().round(1).reset_index()
        .sort_values("pct_parkland", ascending=False)
    )
    fig_gb = go.Figure()
    fig_gb.add_trace(go.Bar(
        x=gb["region"], y=gb["pct_parkland"],
        name="Parkland", marker_color="#1D9E75",
        hovertemplate="<b>%{x}</b><br>Parkland: %{y:.1f}%<extra></extra>",
    ))
    fig_gb.add_trace(go.Bar(
        x=gb["region"], y=gb["pct_green_res"],
        name="Green residential", marker_color="#639922",
        hovertemplate="<b>%{x}</b><br>Green res.: %{y:.1f}%<extra></extra>",
    ))
    fig_gb.update_layout(
        barmode="group", height=300,
        margin=dict(l=10, r=10, t=10, b=40),
        yaxis_title="% coverage (avg)",
        legend=dict(orientation="h", y=1.08),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_gb, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — DEMOGRAPHICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👥 Demographics":
    st.title("Demographics")
    st.caption("Age, sex and population profiles by planning area or region. To compare two areas side by side, use ⚖️ Compare.")

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
            height=400, margin=dict(t=10, b=30), showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_ag, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — INCOME
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💰 Income":
    st.title("Income distribution")
    st.caption("GHS 2015 — residential planning areas only · 28 of 55 areas have income data · To compare income profiles between two areas, use ⚖️ Compare.")

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

    # ── Income vs green metrics scatter ───────────────────────────────────────
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

    inc_scatter = inc_df.dropna(subset=[gm_col, "pct_income_10000_over"]).copy()
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
                height=400, margin=dict(t=40, b=30),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_sc, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — COMPARE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚖️ Compare":
    st.title("Planning area comparison")
    st.caption("Select any two planning areas, regions, or Singapore overall for a head-to-head comparison across land cover, demographics, and income. Explore individual areas first on the Land Cover and Demographics pages.")

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

    st.markdown(
        "<div style='background:rgba(239,159,39,0.08);border-left:3px solid #EF9F27;"
        "border-radius:6px;padding:8px 14px;margin:4px 0;font-size:11px;color:#888'>"
        "⚠️ Parkland % may be slightly underestimated due to classifier limitations (recall 0.75). "
        "GUB and LGS are less affected. See 🔬 Model Assessment."
        "</div>",
        unsafe_allow_html=True,
    )

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
            height=400,
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
                barmode="group", height=300,
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
                    height=300,
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
    st.title("🏆 Green metrics — the bottom line")
    st.caption(
        "GUB and LGS distil everything on the Land Cover page into a single ranked score per area. "
        "This is the dashboard's central analytical output — every other page builds toward this view. "
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
        height=400, margin=dict(t=20, b=40),
        xaxis=dict(range=[-1.05, 1.05],
                   tickvals=[-1,-0.5,0,0.5,1], zeroline=True,
                   zerolinecolor="rgba(128,128,128,0.3)"),
        yaxis=dict(range=[0, 105]),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_sc, use_container_width=True)

    # ── Migrated from Green vs Urban: green vs ageing population ──────────────
    st.divider()
    st.subheader("Green metrics vs ageing population")
    st.caption("Areas with older populations — do they tend to have more or less green space? "
               "Sized by population, coloured by region.")

    gdf_age = gdf_m.copy()
    gdf_age["pct_age_60plus"] = (
        gdf_age["pct_age10_60_69"].fillna(0) +
        gdf_age["pct_age10_70_79"].fillna(0) +
        gdf_age["pct_age10_80plus"].fillna(0)
    )
    gdf_age = gdf_age.dropna(subset=["pct_age_60plus", "gub", "lgs"])

    age_metric = st.radio(
        "Green metric", ["GUB", "LGS"], horizontal=True, key="gm_age_metric"
    )
    gm_x    = "gub"   if age_metric == "GUB" else "lgs"
    gm_xlbl = "GUB score" if age_metric == "GUB" else "LGS (%)"
    gm_xrng = [-1.05, 1.05] if age_metric == "GUB" else [0, 105]
    gm_xtk  = dict(tickvals=[-1,-0.5,0,0.5,1], zeroline=True,
                   zerolinecolor="rgba(128,128,128,0.3)") if age_metric == "GUB" else {}

    fig_age = px.scatter(
        gdf_age,
        x=gm_x, y="pct_age_60plus",
        size="pop2020_total", color="region",
        color_discrete_map=REGION_COLORS,
        hover_name="name",
        hover_data={gm_x: ":.3f" if age_metric == "GUB" else ":.1f",
                    "pct_age_60plus": ":.1f", "pop2020_total": ":,"},
        labels={gm_x: gm_xlbl, "pct_age_60plus": "% Aged 60+", "region": "Region"},
        trendline="ols",
        size_max=40,
    )
    # Override per-region trendlines with a single overall line
    r_all = gdf_age[[gm_x, "pct_age_60plus"]].dropna()
    if len(r_all) > 2:
        m, b = np.polyfit(r_all[gm_x], r_all["pct_age_60plus"], 1)
        x_ln = np.array([r_all[gm_x].min(), r_all[gm_x].max()])
        fig_age = px.scatter(
            gdf_age,
            x=gm_x, y="pct_age_60plus",
            size="pop2020_total", color="region",
            color_discrete_map=REGION_COLORS,
            hover_name="name",
            hover_data={gm_x: ":.3f" if age_metric == "GUB" else ":.1f",
                        "pct_age_60plus": ":.1f", "pop2020_total": ":,"},
            labels={gm_x: gm_xlbl, "pct_age_60plus": "% Aged 60+", "region": "Region"},
            size_max=40,
        )
        fig_age.add_trace(go.Scatter(
            x=x_ln, y=m * x_ln + b, mode="lines",
            line=dict(color="rgba(100,100,100,0.5)", width=2, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ))
        r_val = float(np.corrcoef(r_all[gm_x], r_all["pct_age_60plus"])[0, 1])
        fig_age.add_annotation(
            x=0.97, y=0.05, xref="paper", yref="paper",
            text=f"r = {r_val:+.2f}", showarrow=False,
            font=dict(size=12, color="#888"), xanchor="right",
        )
    fig_age.update_layout(
        height=400, margin=dict(t=20, b=40),
        xaxis=dict(range=gm_xrng, **gm_xtk),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_age, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 8 — MODEL ASSESSMENT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Model Assessment":
    st.title("Model assessment")
    st.caption(
        "Accuracy and limitations of the XGBoost land cover classifier "
        "used to produce all green space data in this dashboard. "
        "Evaluated on a held-out test set of 35 labelled samples."
    )

    # ── Overall accuracy scorecards ───────────────────────────────────────────
    st.divider()
    ma1, ma2, ma3, ma4 = st.columns(4)
    for col, lbl, val, sub, clr in [
        (ma1, "Overall accuracy",  "94.3%",  "33 of 35 samples correct",     "#639922"),
        (ma2, "Macro avg F1",      "0.94",   "Unweighted mean across classes","#1D9E75"),
        (ma3, "Weighted avg F1",   "0.94",   "Weighted by class support",     "#534AB7"),
        (ma4, "Test set size",     "35",     "Held-out labelled samples",     "#BA7517"),
    ]:
        with col:
            st.markdown(
                f"<div style='background:var(--color-background-secondary,#f5f5f5);"
                f"border-radius:8px;padding:10px 12px;margin-bottom:4px'>"
                f"<div style='font-size:13px;font-weight:500;color:var(--color-text-secondary,#555);"
                f"margin-bottom:6px'>{lbl}</div>"
                f"<div style='font-size:22px;font-weight:700;color:{clr}'>{val}</div>"
                f"<div style='font-size:11px;color:#888;margin-top:3px'>{sub}</div>"
                f"</div>", unsafe_allow_html=True,
            )

    st.divider()

    # ── Per-class metrics + confusion matrix ──────────────────────────────────
    col_metrics, col_cm = st.columns([1, 1])

    with col_metrics:
        st.subheader("Per-class performance")
        st.caption("Precision = how often a predicted class is correct · "
                   "Recall = how often the true class is detected · "
                   "F1 = harmonic mean of both")

        classes    = ["Green residential", "Parkland", "Urban", "Water"]
        precisions = [0.83, 1.00, 1.00, 1.00]
        recalls    = [1.00, 0.75, 1.00, 1.00]
        f1s        = [0.91, 0.86, 1.00, 1.00]
        supports   = [10, 8, 8, 9]
        colors_cls = ["#639922", "#1D9E75", "#888780", "#378ADD"]

        fig_cls = go.Figure()
        for vals, name, opacity in [
            (precisions, "Precision", 0.45),
            (recalls,    "Recall",    0.65),
            (f1s,        "F1",        1.00),
        ]:
            fig_cls.add_trace(go.Bar(
                name=name, x=classes, y=vals,
                marker_color=colors_cls,
                opacity=opacity,
                hovertemplate=f"<b>%{{x}}</b><br>{name}: %{{y:.2f}}<extra></extra>",
            ))
        fig_cls.add_hline(y=1.0, line_width=1, line_dash="dot",
                          line_color="rgba(128,128,128,0.3)")
        fig_cls.update_layout(
            barmode="group", height=300,
            margin=dict(t=10, b=30, l=0, r=0),
            yaxis=dict(title="Score", range=[0, 1.1],
                       tickvals=[0, 0.25, 0.5, 0.75, 1.0]),
            legend=dict(orientation="h", y=1.08),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_cls, use_container_width=True)

        # Support table
        st.caption("Test set support (samples per class)")
        sup_df = pd.DataFrame({
            "Class":     classes,
            "Samples":   supports,
            "Precision": [f"{v:.2f}" for v in precisions],
            "Recall":    [f"{v:.2f}" for v in recalls],
            "F1":        [f"{v:.2f}" for v in f1s],
        })
        st.dataframe(sup_df, hide_index=True, use_container_width=True)

    with col_cm:
        st.subheader("Confusion matrix")
        st.caption("Rows = actual class · Columns = predicted class · "
                   "Diagonal = correct predictions · Off-diagonal = misclassifications")

        # Confusion matrix data
        cm = [[10, 0, 0, 0],
              [ 2, 6, 0, 0],
              [ 0, 0, 8, 0],
              [ 0, 0, 0, 9]]

        # Normalise by row (recall per class) for colour, keep raw counts as text
        cm_pct  = [[v / sum(row) * 100 for v in row] for row in cm]
        cm_text = [[str(v) for v in row] for row in cm]

        fig_cm = go.Figure(go.Heatmap(
            z=cm_pct,
            x=classes,
            y=classes,
            text=cm_text,
            texttemplate="%{text}",
            textfont=dict(size=16, color="white"),
            colorscale=[
                [0.0, "rgba(40,40,40,0.1)"],
                [0.5, "#1D9E75"],
                [1.0, "#0a4f3c"],
            ],
            showscale=False,
            hovertemplate="<b>Actual: %{y}</b><br>Predicted: %{x}<br>"
                          "Count: %{text} (%{z:.0f}%)<extra></extra>",
        ))
        fig_cm.update_layout(
            height=300,
            margin=dict(t=10, b=60, l=10, r=10),
            xaxis=dict(title="Predicted", tickfont=dict(size=11), side="bottom"),
            yaxis=dict(title="Actual",    tickfont=dict(size=11), autorange="reversed"),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # ── Key findings ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Key findings")

    kf1, kf2 = st.columns(2)
    with kf1:
        st.success(
            "**Urban and Water: perfect classification (F1 = 1.00)**  \n"
            "Both classes were identified without error on the test set. "
            "Urban surfaces have distinctive low-reflectance spectral signatures; "
            "water has very low near-infrared reflectance that is easy to separate."
        )
        st.success(
            "**Green residential: high recall (1.00), moderate precision (0.83)**  \n"
            "All 10 green residential samples were correctly identified. "
            "The 2 Parkland samples misclassified as Green residential suggest "
            "the model slightly over-predicts residential green cover in densely vegetated areas."
        )
    with kf2:
        st.warning(
            "**Parkland: lower recall (0.75) — the main weakness**  \n"
            "2 of 8 Parkland samples were misclassified as Green residential. "
            "This means the model likely **underestimates parkland** and "
            "**overestimates green residential** coverage in some planning areas. "
            "The GUB and LGS scores are directionally correct but the "
            "Parkland/Green residential split should be interpreted with ~15% margin."
        )
        st.info(
            "**Small test set caveat**  \n"
            "The test set contains only 35 samples. "
            "While accuracy metrics are strong, confidence intervals are wide — "
            "the true accuracy could range from approximately 85% to 99% at 95% confidence. "
            "A larger validation set would improve reliability of per-class estimates."
        )

    # ── Pipeline ──────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Classification pipeline")
    st.markdown("""
    1. **Satellite imagery** — multi-spectral imagery exported from Google Earth Engine covering Singapore's full extent
    2. **Manual labelling** — training samples hand-tagged by a human analyst across all four land cover classes
    3. **Feature extraction** — spectral bands and derived indices extracted per pixel in QGIS
    4. **Model training** — XGBoost classifier trained and evaluated with train/test split
    5. **Classification** — trained model applied to the full raster extent in QGIS
    6. **Zonal statistics** — pixel counts aggregated to planning area polygons using QGIS Zonal Histogram
    """)

    # ── Implications ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Implications for interpretation")
    st.warning(
        "All green space percentages, GUB scores, and LGS scores in this dashboard are derived from "
        "ML-classified data. The main risk is that some Parkland pixels are classified as Green residential, "
        "which would slightly **inflate Green residential %** and **deflate Parkland %** in affected areas.  \n\n"
        "Practical guidance:  \n"
        "- Rankings between areas with **similar scores (within ~2–3pp)** should be treated cautiously  \n"
        "- The **Parkland %** column specifically may be slightly underestimated  \n"
        "- **GUB and LGS** aggregate both green classes so are less sensitive to this misclassification  \n"
        "- Urban and Water classifications are reliable"
    )
