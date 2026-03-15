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


# ═══════════════════════════════════════════════════════════════════════════# ==============================================================================
# PAGE 1 — MAP
# ══════════════════════════════════════════════════════════════════════════════
if page == "🗺️ Map":
    st.title("Land cover map")
    st.caption("Classified land cover overlaid with planning area boundaries. Click a polygon for stats.")

    col_map, col_info = st.columns([3, 1])

    with col_map:
        rgba, bounds = load_raster_preview()

        m = folium.Map(
            location=[1.3521, 103.8198],
            zoom_start=11,
            tiles=None,
        )

        folium.TileLayer(
            tiles="https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png",
            attr="CartoDB", name="Base", show=True,
        ).add_to(m)

        # Classified raster overlay
        if rgba is not None:
            lc_img = Image.fromarray(rgba, mode="RGBA")
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                lc_img.save(tmp.name)
                folium.raster_layers.ImageOverlay(
                    image=tmp.name,
                    bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
                    opacity=0.75,
                    name="Classified land cover",
                    zindex=2,
                ).add_to(m)
        else:
            st.info("No classified raster found. Place your GeoTIFF in the app folder.")

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
                    "fillColor":   "transparent",
                    "color":       "#ffffff",
                    "weight":      1.0,
                    "fillOpacity": 0,
                },
                highlight_function=lambda f: {
                    "fillColor":   "#ffffff",
                    "fillOpacity": 0.2,
                    "weight":      2.5,
                    "color":       "#ffffff",
                },
                tooltip=GeoJsonTooltip(
                    fields=["name", "region", "pop2020_total",
                            "pct_urban", "pct_green_total", "pct_water"],
                    aliases=["Area", "Region", "Population",
                             "% Urban", "% Green (total)", "% Water"],
                    localize=True,
                    sticky=True,
                ),
            ).add_to(m)
        else:
            st.info("No shapefile found. Place your planning area .geojson in the app folder.")

        # Legend
        legend_html = """
        <div style='position:fixed;bottom:30px;left:30px;z-index:9999;
                     background:rgba(0,0,0,0.7);color:#fff;
                     padding:10px 14px;border-radius:8px;
                     font-size:12px;line-height:1.8'>
          <b>Land cover</b><br>
          <span style='color:#639922'>&#9632;</span> Green residential<br>
          <span style='color:#1D9E75'>&#9632;</span> Parkland<br>
          <span style='color:#888780'>&#9632;</span> Urban<br>
          <span style='color:#378ADD'>&#9632;</span> Water
        </div>"""
        m.get_root().html.add_child(folium.Element(legend_html))
        folium.LayerControl().add_to(m)
        map_data = st_folium(m, width="100%", height=580,
                             returned_objects=["last_object_clicked_tooltip"])

    with col_info:
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
    st.plotly_chart(fig_sc, use_container_width=True)# ==============================================================================
# PAGE 1 — MAP
# ==============================================================================
if page == "🗺️ Map":
    st.title("Land cover map")
    st.caption("Drag the ⇔ handle to swipe between satellite and classification. Click a planning area for stats.")

    sat_rgba, sat_bounds = load_satellite_preview()
    rgba, bounds         = load_raster_preview()

    import base64, io, json as _json

    def img_to_b64(arr):
        buf = io.BytesIO()
        Image.fromarray(arr, mode="RGBA").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    has_sat = sat_rgba is not None
    has_lc  = rgba is not None

    if not has_sat: st.warning("Satellite image not found — swipe will show classification only.")
    if not has_lc:  st.warning("Classified raster not found.")

    sat_b64 = img_to_b64(sat_rgba) if has_sat else ""
    lc_b64  = img_to_b64(rgba)     if has_lc  else ""
    b       = bounds if has_lc else (sat_bounds if has_sat else None)

    if b is None:
        st.error("No raster files found.")
    else:
        geojson_str = "{}"
        if gdf is not None:
            merged = gdf.merge(
                df[["PLN_AREA_N","name","region","pct_urban","pct_green_total",
                    "pct_parkland","pct_green_res","pct_water","pop2020_total"]],
                on="PLN_AREA_N", how="left",
            )
            geojson_str = merged.to_json()

        PA_LOOKUP_JSON = '{"BEDOK": {"name": "Bedok", "region": "East", "pop": "276,990", "pct_urban": 41.0, "pct_green_res": 26.7, "pct_parkland": 25.8, "pct_water": 6.5, "pct_green_total": 52.5}, "BOON LAY": {"name": "Boon Lay", "region": "West", "pop": "40", "pct_urban": 48.6, "pct_green_res": 19.3, "pct_parkland": 19.2, "pct_water": 12.9, "pct_green_total": 38.5}, "BUKIT BATOK": {"name": "Bukit Batok", "region": "West", "pop": "158,030", "pct_urban": 28.6, "pct_green_res": 23.0, "pct_parkland": 46.8, "pct_water": 1.6, "pct_green_total": 69.8}, "BUKIT MERAH": {"name": "Bukit Merah", "region": "Central", "pop": "151,250", "pct_urban": 39.8, "pct_green_res": 21.1, "pct_parkland": 33.3, "pct_water": 5.8, "pct_green_total": 54.4}, "BUKIT PANJANG": {"name": "Bukit Panjang", "region": "West", "pop": "138,270", "pct_urban": 22.0, "pct_green_res": 20.0, "pct_parkland": 56.8, "pct_water": 1.3, "pct_green_total": 76.8}, "BUKIT TIMAH": {"name": "Bukit Timah", "region": "Central", "pop": "77,860", "pct_urban": 24.3, "pct_green_res": 32.6, "pct_parkland": 42.5, "pct_water": 0.6, "pct_green_total": 75.1}, "CENTRAL WATER CATCHMENT": {"name": "Central Water Catchment", "region": "North", "pop": "n/a", "pct_urban": 3.3, "pct_green_res": 4.0, "pct_parkland": 75.5, "pct_water": 17.2, "pct_green_total": 79.5}, "CHANGI": {"name": "Changi", "region": "East", "pop": "1,850", "pct_urban": 40.8, "pct_green_res": 14.4, "pct_parkland": 33.9, "pct_water": 10.9, "pct_green_total": 48.3}, "CHOA CHU KANG": {"name": "Choa Chu Kang", "region": "West", "pop": "192,070", "pct_urban": 43.1, "pct_green_res": 33.8, "pct_parkland": 21.1, "pct_water": 1.9, "pct_green_total": 54.9}, "CLEMENTI": {"name": "Clementi", "region": "West", "pop": "91,990", "pct_urban": 38.5, "pct_green_res": 28.1, "pct_parkland": 28.2, "pct_water": 5.2, "pct_green_total": 56.3}, "HOUGANG": {"name": "Hougang", "region": "North-East", "pop": "227,560", "pct_urban": 44.8, "pct_green_res": 27.4, "pct_parkland": 24.0, "pct_water": 3.8, "pct_green_total": 51.4}, "JURONG EAST": {"name": "Jurong East", "region": "West", "pop": "78,600", "pct_urban": 35.6, "pct_green_res": 19.4, "pct_parkland": 19.6, "pct_water": 25.4, "pct_green_total": 39.0}, "JURONG WEST": {"name": "Jurong West", "region": "West", "pop": "262,730", "pct_urban": 44.9, "pct_green_res": 28.6, "pct_parkland": 22.5, "pct_water": 3.9, "pct_green_total": 51.1}, "PASIR RIS": {"name": "Pasir Ris", "region": "East", "pop": "147,110", "pct_urban": 34.6, "pct_green_res": 20.4, "pct_parkland": 35.6, "pct_water": 9.4, "pct_green_total": 56.0}, "PIONEER": {"name": "Pioneer", "region": "West", "pop": "80", "pct_urban": 53.5, "pct_green_res": 18.9, "pct_parkland": 12.5, "pct_water": 15.1, "pct_green_total": 31.4}, "PUNGGOL": {"name": "Punggol", "region": "North-East", "pop": "174,450", "pct_urban": 28.8, "pct_green_res": 26.6, "pct_parkland": 37.9, "pct_water": 6.7, "pct_green_total": 64.5}, "QUEENSTOWN": {"name": "Queenstown", "region": "Central", "pop": "95,930", "pct_urban": 39.6, "pct_green_res": 18.4, "pct_parkland": 30.8, "pct_water": 11.2, "pct_green_total": 49.2}, "SELETAR": {"name": "Seletar", "region": "North-East", "pop": "300", "pct_urban": 25.9, "pct_green_res": 14.8, "pct_parkland": 50.4, "pct_water": 8.9, "pct_green_total": 65.2}, "SEMBAWANG": {"name": "Sembawang", "region": "North", "pop": "102,640", "pct_urban": 40.7, "pct_green_res": 24.0, "pct_parkland": 28.9, "pct_water": 6.4, "pct_green_total": 52.9}, "SENGKANG": {"name": "Sengkang", "region": "North-East", "pop": "249,370", "pct_urban": 32.9, "pct_green_res": 29.4, "pct_parkland": 34.5, "pct_water": 3.2, "pct_green_total": 63.9}, "SERANGOON": {"name": "Serangoon", "region": "North-East", "pop": "116,900", "pct_urban": 53.7, "pct_green_res": 27.8, "pct_parkland": 15.4, "pct_water": 3.2, "pct_green_total": 43.2}, "KALLANG": {"name": "Kallang", "region": "Central", "pop": "101,290", "pct_urban": 40.7, "pct_green_res": 24.6, "pct_parkland": 22.4, "pct_water": 12.3, "pct_green_total": 47.0}, "LIM CHU KANG": {"name": "Lim Chu Kang", "region": "North", "pop": "110", "pct_urban": 8.8, "pct_green_res": 11.7, "pct_parkland": 69.4, "pct_water": 10.1, "pct_green_total": 81.1}, "NORTH-EASTERN ISLANDS": {"name": "North-Eastern Islands", "region": "North-East", "pop": "50", "pct_urban": 15.4, "pct_green_res": 8.6, "pct_parkland": 53.3, "pct_water": 22.6, "pct_green_total": 61.9}, "NOVENA": {"name": "Novena", "region": "Central", "pop": "49,330", "pct_urban": 26.9, "pct_green_res": 27.9, "pct_parkland": 43.2, "pct_water": 2.0, "pct_green_total": 71.1}, "SIMPANG": {"name": "Simpang", "region": "North", "pop": "n/a", "pct_urban": 2.8, "pct_green_res": 5.6, "pct_parkland": 59.3, "pct_water": 32.3, "pct_green_total": 64.9}, "SOUTHERN ISLANDS": {"name": "Southern Islands", "region": "Central", "pop": "1,940", "pct_urban": 14.5, "pct_green_res": 15.9, "pct_parkland": 52.3, "pct_water": 17.2, "pct_green_total": 68.2}, "SUNGEI KADUT": {"name": "Sungei Kadut", "region": "North", "pop": "750", "pct_urban": 23.2, "pct_green_res": 15.5, "pct_parkland": 46.7, "pct_water": 14.6, "pct_green_total": 62.2}, "TOA PAYOH": {"name": "Toa Payoh", "region": "Central", "pop": "121,850", "pct_urban": 43.8, "pct_green_res": 29.4, "pct_parkland": 24.1, "pct_water": 2.6, "pct_green_total": 53.5}, "TUAS": {"name": "Tuas", "region": "West", "pop": "70", "pct_urban": 43.3, "pct_green_res": 11.6, "pct_parkland": 19.1, "pct_water": 25.9, "pct_green_total": 30.7}, "WESTERN ISLANDS": {"name": "Western Islands", "region": "West", "pop": "10", "pct_urban": 35.3, "pct_green_res": 11.6, "pct_parkland": 33.7, "pct_water": 19.4, "pct_green_total": 45.3}, "WESTERN WATER CATCHMENT": {"name": "Western Water Catchment", "region": "West", "pop": "640", "pct_urban": 10.9, "pct_green_res": 11.4, "pct_parkland": 65.7, "pct_water": 12.0, "pct_green_total": 77.1}, "WOODLANDS": {"name": "Woodlands", "region": "North", "pop": "255,130", "pct_urban": 39.5, "pct_green_res": 30.2, "pct_parkland": 26.2, "pct_water": 4.2, "pct_green_total": 56.4}, "RIVER VALLEY": {"name": "River Valley", "region": "Central", "pop": "10,070", "pct_urban": 33.0, "pct_green_res": 36.3, "pct_parkland": 29.6, "pct_water": 1.1, "pct_green_total": 65.9}, "ROCHOR": {"name": "Rochor", "region": "Central", "pop": "13,120", "pct_urban": 66.1, "pct_green_res": 17.7, "pct_parkland": 9.1, "pct_water": 7.2, "pct_green_total": 26.8}, "SINGAPORE RIVER": {"name": "Singapore River", "region": "Central", "pop": "3,260", "pct_urban": 51.0, "pct_green_res": 21.9, "pct_parkland": 11.7, "pct_water": 15.4, "pct_green_total": 33.6}, "STRAITS VIEW": {"name": "Straits View", "region": "Central", "pop": "n/a", "pct_urban": 9.3, "pct_green_res": 10.8, "pct_parkland": 50.9, "pct_water": 29.0, "pct_green_total": 61.7}, "CHANGI BAY": {"name": "Changi Bay", "region": "East", "pop": "n/a", "pct_urban": 23.6, "pct_green_res": 19.2, "pct_parkland": 51.8, "pct_water": 5.5, "pct_green_total": 71.0}, "MARINE PARADE": {"name": "Marine Parade", "region": "Central", "pop": "46,220", "pct_urban": 37.0, "pct_green_res": 23.7, "pct_parkland": 37.0, "pct_water": 2.3, "pct_green_total": 60.7}, "DOWNTOWN CORE": {"name": "Downtown Core", "region": "Central", "pop": "3,190", "pct_urban": 45.3, "pct_green_res": 16.2, "pct_parkland": 12.7, "pct_water": 25.8, "pct_green_total": 28.9}, "MARINA EAST": {"name": "Marina East", "region": "Central", "pop": "n/a", "pct_urban": 16.8, "pct_green_res": 8.8, "pct_parkland": 51.9, "pct_water": 22.4, "pct_green_total": 60.7}, "MARINA SOUTH": {"name": "Marina South", "region": "Central", "pop": "n/a", "pct_urban": 9.6, "pct_green_res": 16.2, "pct_parkland": 54.9, "pct_water": 19.3, "pct_green_total": 71.1}, "MUSEUM": {"name": "Museum", "region": "Central", "pop": "510", "pct_urban": 33.2, "pct_green_res": 22.1, "pct_parkland": 41.9, "pct_water": 2.7, "pct_green_total": 64.0}, "NEWTON": {"name": "Newton", "region": "Central", "pop": "8,260", "pct_urban": 23.8, "pct_green_res": 28.3, "pct_parkland": 47.1, "pct_water": 0.9, "pct_green_total": 75.4}, "ORCHARD": {"name": "Orchard", "region": "Central", "pop": "920", "pct_urban": 51.5, "pct_green_res": 24.1, "pct_parkland": 14.6, "pct_water": 9.7, "pct_green_total": 38.7}, "OUTRAM": {"name": "Outram", "region": "Central", "pop": "18,340", "pct_urban": 51.3, "pct_green_res": 16.6, "pct_parkland": 26.4, "pct_water": 5.7, "pct_green_total": 43.0}, "TAMPINES": {"name": "Tampines", "region": "East", "pop": "259,900", "pct_urban": 37.8, "pct_green_res": 25.2, "pct_parkland": 32.1, "pct_water": 4.9, "pct_green_total": 57.3}, "TANGLIN": {"name": "Tanglin", "region": "Central", "pop": "21,810", "pct_urban": 17.9, "pct_green_res": 31.6, "pct_parkland": 50.0, "pct_water": 0.4, "pct_green_total": 81.6}, "TENGAH": {"name": "Tengah", "region": "West", "pop": "10", "pct_urban": 28.6, "pct_green_res": 12.5, "pct_parkland": 49.6, "pct_water": 9.2, "pct_green_total": 62.1}, "MANDAI": {"name": "Mandai", "region": "North", "pop": "2,090", "pct_urban": 9.1, "pct_green_res": 10.4, "pct_parkland": 79.9, "pct_water": 0.6, "pct_green_total": 90.3}, "BISHAN": {"name": "Bishan", "region": "Central", "pop": "87,320", "pct_urban": 41.7, "pct_green_res": 24.7, "pct_parkland": 30.4, "pct_water": 3.1, "pct_green_total": 55.1}, "ANG MO KIO": {"name": "Ang Mo Kio", "region": "Central", "pop": "162,280", "pct_urban": 33.3, "pct_green_res": 25.3, "pct_parkland": 38.7, "pct_water": 2.6, "pct_green_total": 64.0}, "GEYLANG": {"name": "Geylang", "region": "Central", "pop": "110,110", "pct_urban": 52.4, "pct_green_res": 27.1, "pct_parkland": 16.7, "pct_water": 3.7, "pct_green_total": 43.8}, "PAYA LEBAR": {"name": "Paya Lebar", "region": "East", "pop": "40", "pct_urban": 21.4, "pct_green_res": 14.1, "pct_parkland": 58.3, "pct_water": 6.1, "pct_green_total": 72.4}, "YISHUN": {"name": "Yishun", "region": "North", "pop": "221,610", "pct_urban": 25.8, "pct_green_res": 19.9, "pct_parkland": 37.1, "pct_water": 17.2, "pct_green_total": 57.0}}'

        sat_overlay = (
            f"var satLayer=L.imageOverlay('data:image/png;base64,{sat_b64}',"
            f"imgBounds,{{opacity:1.0,pane:'leftPane'}}).addTo(map);"
            if has_sat else "// no satellite"
        )
        lc_overlay = (
            f"var lcLayer=L.imageOverlay('data:image/png;base64,{lc_b64}',"
            f"imgBounds,{{opacity:0.9,pane:'rightPane'}}).addTo(map);"
            if has_lc else "// no lc"
        )

        css = """
html,body{margin:0;padding:0;height:100%}
#map{width:100%;height:600px;position:relative;overflow:hidden}
#swipe-handle{position:absolute;top:0;bottom:0;width:3px;background:#fff;
  cursor:ew-resize;z-index:800;box-shadow:0 0 8px rgba(0,0,0,.5);pointer-events:all}
#swipe-btn{position:absolute;top:50%;transform:translateY(-50%);width:34px;height:34px;
  background:#fff;border-radius:50%;display:flex;align-items:center;justify-content:center;
  left:-17px;box-shadow:0 0 8px rgba(0,0,0,.35);font-size:15px;
  user-select:none;pointer-events:none}
.lbl{position:absolute;bottom:40px;z-index:850;background:rgba(0,0,0,.6);color:#fff;
  font-size:11px;padding:3px 9px;border-radius:4px;font-family:sans-serif;pointer-events:none}
#lbl-left{left:10px} #lbl-right{right:10px}
#legend{position:absolute;bottom:8px;left:50%;transform:translateX(-50%);z-index:850;
  background:rgba(255,255,255,.93);padding:5px 12px;border-radius:6px;font-size:11px;
  font-family:sans-serif;display:flex;gap:10px;box-shadow:0 1px 5px rgba(0,0,0,.2);pointer-events:none}
.li{display:flex;align-items:center;gap:4px}
.ls{width:11px;height:11px;border-radius:2px;flex-shrink:0}
#stats-panel{display:none;position:absolute;top:10px;right:10px;z-index:900;
  background:rgba(255,255,255,.97);border-radius:8px;padding:12px 14px;
  font-family:sans-serif;font-size:12px;min-width:190px;
  box-shadow:0 2px 12px rgba(0,0,0,.2)}
#stats-close{float:right;cursor:pointer;font-size:14px;color:#888;margin-left:8px;line-height:1}
#stats-title{font-size:14px;font-weight:600;margin-bottom:2px;color:#222}
#stats-region{color:#888;margin-bottom:8px;font-size:11px}
.stat-row{display:flex;justify-content:space-between;padding:3px 0;
  border-bottom:1px solid #f0f0f0;gap:16px}
.stat-label{color:#555}.stat-val{font-weight:500;color:#222}
.sbar{height:6px;border-radius:3px;margin-top:8px;display:flex;overflow:hidden}
.sbar-seg{height:100%}
"""

        js = """
var paData = """ + PA_LOOKUP_JSON + """;
var map = L.map('map',{zoomControl:true,dragging:true}).setView([1.3521,103.8198],11);
L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png',
  {attribution:'CartoDB',maxZoom:18}).addTo(map);
var imgBounds = """ + f"[[{b.bottom},{b.left}],[{b.top},{b.right}]]" + """;
var leftPane  = map.createPane('leftPane');
var rightPane = map.createPane('rightPane');
leftPane.style.zIndex=300; rightPane.style.zIndex=350;
""" + sat_overlay + """
""" + lc_overlay  + """
var geojsonData = """ + geojson_str + """;
if(geojsonData&&geojsonData.features){
  L.geoJSON(geojsonData,{
    style:{color:'#ffffff',weight:1.0,fillOpacity:0,opacity:0.6},
    onEachFeature:function(feat,layer){
      layer.on('mouseover',function(){layer.setStyle({fillColor:'#ffffff',fillOpacity:0.15,weight:2})});
      layer.on('mouseout', function(){layer.setStyle({fillColor:'transparent',fillOpacity:0,weight:1})});
      layer.on('click',function(e){
        var pa=feat.properties&&feat.properties.PLN_AREA_N;
        if(!pa)return;
        var d=paData[pa];
        if(!d)return;
        document.getElementById('stats-title').textContent=d.name;
        document.getElementById('stats-region').textContent=d.region+' Region';
        var rows=[
          ['Population',d.pop],['% Urban',(d.pct_urban||0)+'%'],
          ['% Green (total)',(d.pct_green_total||0)+'%'],['% Parkland',(d.pct_parkland||0)+'%'],
          ['% Green res.',(d.pct_green_res||0)+'%'],['% Water',(d.pct_water||0)+'%'],
        ];
        var html='';
        rows.forEach(function(r){html+='<div class="stat-row"><span class="stat-label">'+r[0]+'</span><span class="stat-val">'+r[1]+'</span></div>';});
        document.getElementById('stats-rows').innerHTML=html;
        var segs=[[d.pct_green_res,'#639922'],[d.pct_parkland,'#1D9E75'],[d.pct_urban,'#888780'],[d.pct_water,'#378ADD']];
        var bar='';
        segs.forEach(function(s){bar+='<div class="sbar-seg" style="width:'+(s[0]||0)+'%;background:'+s[1]+'"></div>';});
        document.getElementById('stats-bar').innerHTML=bar;
        document.getElementById('stats-panel').style.display='block';
        L.DomEvent.stopPropagation(e);
      });
    }
  }).addTo(map);
}
var mapDiv=document.getElementById('map');
var handle=document.getElementById('swipe-handle');
var mapW=mapDiv.offsetWidth,pos=mapW/2,dragging=false;
function setClip(x){
  mapW=mapDiv.offsetWidth;
  pos=Math.max(0,Math.min(mapW,x));
  leftPane.style.clip='rect(0px,'+pos+'px,9999px,0px)';
  rightPane.style.clip='rect(0px,9999px,9999px,'+pos+'px)';
  handle.style.left=pos+'px';
}
map.whenReady(function(){mapW=mapDiv.offsetWidth;setClip(mapW/2);});
handle.addEventListener('mousedown',function(e){dragging=true;map.dragging.disable();e.preventDefault();e.stopPropagation();});
handle.addEventListener('touchstart',function(e){dragging=true;map.dragging.disable();e.stopPropagation();},{passive:true});
document.addEventListener('mouseup',  function(){if(dragging){dragging=false;map.dragging.enable();}});
document.addEventListener('touchend', function(){if(dragging){dragging=false;map.dragging.enable();}});
document.addEventListener('mousemove',function(e){if(!dragging)return;var r=mapDiv.getBoundingClientRect();setClip(e.clientX-r.left);});
document.addEventListener('touchmove',function(e){if(!dragging)return;var r=mapDiv.getBoundingClientRect();setClip(e.touches[0].clientX-r.left);},{passive:true});
window.addEventListener('resize',function(){mapW=mapDiv.offsetWidth;setClip(pos);});
"""

        map_html = (
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            "<link rel='stylesheet' href='https://unpkg.com/leaflet@1.9.4/dist/leaflet.css'/>"
            "<script src='https://unpkg.com/leaflet@1.9.4/dist/leaflet.js'></script>"
            f"<style>{css}</style></head><body>"
            "<div id='map'>"
            "  <div id='swipe-handle'><div id='swipe-btn'>&#8660;</div></div>"
            "  <div class='lbl' id='lbl-left'>&#128752; Satellite</div>"
            "  <div class='lbl' id='lbl-right'>&#128202; Classified</div>"
            "  <div id='legend'>"
            "    <div class='li'><div class='ls' style='background:#639922'></div>Green res.</div>"
            "    <div class='li'><div class='ls' style='background:#1D9E75'></div>Parkland</div>"
            "    <div class='li'><div class='ls' style='background:#888780'></div>Urban</div>"
            "    <div class='li'><div class='ls' style='background:#378ADD'></div>Water</div>"
            "  </div>"
            "  <div id='stats-panel'>"
            "    <span id='stats-close' onclick='this.parentElement.style.display=\"none\"'>&#10005;</span>"
            "    <div id='stats-title'></div>"
            "    <div id='stats-region'></div>"
            "    <div id='stats-rows'></div>"
            "    <div class='sbar' id='stats-bar'></div>"
            "  </div>"
            "</div>"
            f"<script>{js}</script>"
            "</body></html>"
        )

        col_map, col_info = st.columns([3, 1])
        with col_map:
            st.components.v1.html(map_html, height=620, scrolling=False)
        with col_info:
            st.subheader("Planning area stats")
            st.caption("Click any planning area on the map.")
            st.divider()
            st.markdown("**Singapore overall**")
            for key, label in LC_LABELS.items():
                st.metric(label, f"{df[key].mean():.1f}%")
