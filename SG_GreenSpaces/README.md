# Singapore Land Cover & Demographics Dashboard

## Files needed in your repository

```
your-repo/
├── app.py
├── requirements.txt
├── SG_Zonal_statistics_w_Greenspaces.csv        ← your CSV
├── classified_nonodata.tif                       ← your classified raster (after NoData fix)
└── MasterPlan2019PlanningAreaBoundaryNoSea.shp   ← your shapefile (+ .dbf .shx .prj)
```

The app will run without the raster or shapefile — it just won't show those layers on the map page.

## Deploy to Streamlit Community Cloud

1. Push all files to a **public GitHub repository**
2. Go to https://share.streamlit.io
3. Sign in with GitHub
4. Click **New app**
5. Select your repo, branch (main), and set Main file path to `app.py`
6. Click **Deploy** — you'll get a shareable URL in ~2 minutes

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Raster file name

The app looks for your classified raster under these names (in order):
- `classified_nonodata.tif`   ← preferred (NoData-stripped version)
- `classified_3414.tif`
- `classified.tif`
- `land_cover.tif`

Rename your file to match one of these, or edit the `load_raster_preview()` function in app.py.

## Shapefile name

The app looks for:
- `MasterPlan2019PlanningAreaBoundaryNoSea.shp`
- `MP19_PLNG_AREA_NO_SEA_PL.shp`
- `planning_areas.shp`

## Pages

| Page | What it shows |
|------|---------------|
| 🗺️ Map | Classified raster + planning area boundaries. Click a polygon for stats. |
| 🌿 Green vs Urban | Scatter + stacked bars comparing green and urban cover |
| 📊 Land Cover | Full land cover breakdown by area, treemap by region |
| 👥 Demographics | Age profiles + land cover by planning area or region |
| 💰 Income | Income distribution alongside land cover, green vs high earner scatter |
