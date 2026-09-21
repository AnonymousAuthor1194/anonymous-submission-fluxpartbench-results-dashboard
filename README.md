# FluxPart-E Dashboard

Standalone Streamlit app for exploring carbon-flux partitioning metrics.
Evaluation tables and site catalogs are included in this repository.

## Requirements

- Python **3.12** (64-bit)
- `pip`

On Debian/Ubuntu, Cartopy may also need system libraries:

```bash
sudo apt-get update
sudo xargs -a packages.txt apt-get install -y
```

## Run locally

From the repository root (after a `git clone`, or after downloading and extracting the repository ZIP from the repo's "Code → Download ZIP"):

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows (cmd)
.venv\Scripts\activate

# Windows (Git Bash)
source .venv/Scripts/activate
```

Then:

```bash
pip install -r requirements.txt
streamlit run web/metrics_dashboard_public.py
```

The app opens at [http://localhost:8501](http://localhost:8501). Use the **Generate** buttons on each section to draw plots.

If `pip install` fails on Cartopy, install it from conda-forge and retry the remaining packages:

```bash
conda install -c conda-forge cartopy
pip install -r requirements.txt
```

## Data

Paths are defined in `configurations/base_paths.yaml` (relative to that file):

- `results/{method}/` — yearly, across-years, and rain-pulse evaluation tables
- `data/pulses/` — pulse event catalog
- `data/meta_information/` — flux-tower metadata
- `data/geo/` — offline Natural Earth shapefiles for the site map (no download required)

## Anonymous review

The app runs entirely on your machine and makes no external network requests.
Git author metadata is anonymized, and the app does not collect Streamlit usage
statistics (`.streamlit/config.toml`).