import functools
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Sequence
from datetime import date

import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import yaml

try:
    from mpl_toolkits.basemap import Basemap
except ImportError:  # pragma: no cover
    Basemap = None

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except Exception:  # pragma: no cover
    ccrs = None
    cfeature = None


def _read_yaml(file: Path) -> Any:
    with open(file, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def get_repository_root() -> Path:
    """Repository root: directory that contains configurations/base_paths.yaml."""
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "configurations" / "base_paths.yaml").exists():
            return candidate
    raise FileNotFoundError(
        "Could not find configurations/base_paths.yaml relative to this file."
    )


def _geo_data_dir() -> Path:
    """Directory with bundled Natural Earth files (Cartopy + Matplotlib basemap)."""
    try:
        root = get_repository_root()
    except FileNotFoundError:
        root = Path(__file__).resolve().parent.parent
    return root / "data" / "geo"


def _configure_cartopy_offline() -> None:
    """Point Cartopy at repo shapefiles so coastlines/BORDERS never download."""
    data_dir = _geo_data_dir()
    if not (data_dir / "shapefiles" / "natural_earth").is_dir():
        return
    try:
        import cartopy
        cartopy.config["pre_existing_data_dir"] = str(data_dir)
    except Exception:
        pass


_configure_cartopy_offline()


def get_configuration(**kwargs: Any) -> Dict[str, Any]:
    """Load default paths from configurations/base_paths.yaml."""
    config: Dict[str, Any] = {}
    repo_root = get_repository_root()
    path_config_path = repo_root / "configurations"
    path_config_file = path_config_path / "base_paths.yaml"
    path_configs = _read_yaml(path_config_file)

    config["paths"] = {}
    for key, value in path_configs.items():
        config["paths"][key] = str(Path(path_config_path / value).resolve())

    for path_key, path_value in config["paths"].items():
        if path_value is not None and not Path(path_value).exists():
            raise AssertionError(
                f"Path {path_value} does not exist. "
                f"Please check paths in {path_config_file}"
            )

    def overwrite_dict(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in new.items():
            if isinstance(v, dict):
                if k in old:
                    old[k] = overwrite_dict(old[k], v)
                else:
                    old[k] = v
            else:
                old[k] = v
        return old

    return overwrite_dict(config, kwargs)


DATA_ID = "216medium2"
RUN_ID = "full1"
# Bump when load_results matching logic changes (invalidates Streamlit cache).
METRICS_LOAD_VERSION = 6
MODELS = ["quincy", "cablepop"]
MODEL_DISPLAY_NAMES = {"quincy": "Quincy", "cablepop": "Cable-Pop"}
LEADERBOARD_FLUXES = ("NEE", "GPP", "RECO")
LEADERBOARD_METRICS = ("NSE", "Bias")
# All metrics the user can pick from in the leaderboard selector.
LEADERBOARD_METRICS_OPTIONS = ("NSE", "Bias", "cRMSE", "RMSE")
PERIOD_CONFIG = {
    "yearly": {"subdir": "yearly", "suffix": "_yearly.csv"},
    "across_years": {"subdir": "across_years", "suffix": "_across_years.csv"},
    "harvest_event": {"subdir": "events/harvest", "suffix": "_harvest_events.csv"},
    "pulse_event": {"subdir": "events/rain_pulse", "suffix": "_rain_pulse_events.csv"},
}
PERIOD_OPTIONS = ("yearly", "across_years", "pulse_event")
EVENT_PERIODS = frozenset({"harvest_event", "pulse_event"})
PERIOD_FILES = {
    "yearly": "results_{data_id}_{run_id}.csv",
    "across_years": "results_{data_id}_{run_id}_across_years.csv",
    "harvest_event": "results_{data_id}_{run_id}_harvest_events.csv",
    "pulse_event": "results_{data_id}_{run_id}_rain_pulse_events.csv",
}
META_COLUMNS = {
    "Year",
    "Flux",
    "Method",
    "Scale",
    "Event",
    "N",
    "NaNs",
    "N_measured",
    "Site",
    "Lat",
    "Lon",
    "IGBP",
    "Model",
    "Challenge",
    "event_ID",
    "event_site_ID",
    "SITE_ID",
    "start_eval",
    "end_eval",
    "SOP",
    "EOP",
    "Duration",
    "start_of_season",
    "end_of_season",
    "peak_of_season",
    "season_harvest",
    "main_season",
    "reduced",
    "site_year",
    "period_key",
    "_group_sort_id",
    "Data_ID",
    "_source_file",
    "CorrelationWarningFlag",
}
METHOD_INFO = {
    "NT": (
        "Nighttime Method (NT) — Reichstein et al. (2005). "
        "Fits a parametric respiration model to nighttime NEE (typically an exponential / Lloyd & Taylor or Q10-type temperature response) "
        "and extrapolates the fitted Reco to daytime conditions. The approach assumes that nighttime measurements provide an unbiased estimate "
        "of the temperature sensitivity of ecosystem respiration and therefore can be used to reconstruct daytime Reco; GPP is then obtained as "
        "GPP = Reco - NEE. Widely implemented (e.g., REddyProc), robust for many sites but sensitive to nighttime data quality and to the assumption "
        "that night and day respiration follow the same temperature relationship. \n\n"
        "Reference: Reichstein, M., Falge, E., Baldocchi, D., Papale, D., Aubinet, M., Berbigier, P., Bernhofer, C., Buchmann, N., Gilmanov, T., Granier, A., Grünwald, T., Havránková, K., Ilvesniemi, H., Janous, D., Knohl, A., Laurila, T., Lohila, A., Loustau, D., Matteucci, G., Meyers, T., Miglietta, F., Ourcival, J.-M., Pumpanen, J., Rambal, S., Rotenberg, E., Sanz, M., Seufert, G., Vaccari, F., Vesala, T., Yakir, D., & Valentini, R. (2005). On the separation of net ecosystem exchange into assimilation and ecosystem respiration: review and improved algorithm. Global Change Biology, 11(9), 1424–1439. doi:10.1111/j.1365-2486.2005.001002.x."
    ),

    "DT": (
        "Daytime Method (DT) — Lasslop et al. (2010). "
        "Fits a light-response curve (a rectangular-hyperbola / Michaelis–Menten style formulation) to daytime NEE together with a temperature-dependent "
        "respiration term (often Lloyd & Taylor), optionally including drought/stress scalars. GPP is represented explicitly by the light-driven term, "
        "and Reco is estimated from the temperature term plus the residual. By using daytime information directly, DT avoids relying solely on night extrapolation "
        "but introduces dependence on the chosen light-response formulation and on daytime meteorological covariates. \n\n"
        "Reference: Lasslop, G., Reichstein, M., Papale, D., Richardson, A. D., Arneth, A., Barr, A., Stoy, P. C., Wohlfahrt, G., et al. (2010). Separation of net ecosystem exchange into assimilation and respiration using a light response curve approach: critical issues and global evaluation. Global Change Biology, 16(1), 187–208. doi:10.1111/j.1365-2486.2009.02041.x."
    ),

    "TK": (
        "Keenan correction for the Kok effect (TK) — Keenan et al. (2019). "
        "Accounts for the observed inhibition (suppression) of daytime respiration by light (the ‘Kok effect’) that can bias partitioning methods which "
        "assume identical day/night respiration behavior. The correction implements an empirical or mechanistic light-dependent suppression of Reco (or an "
        "explicit light-sensitive term) so that daytime Reco is reduced relative to a straight night-to-day extrapolation; this reduces positive bias in GPP "
        "that arises when daytime respiration is lower than nighttime-based estimates. \n\n"
        "Reference: Keenan, T. F., Migliavacca, M., Papale, D., Baldocchi, D., Reichstein, M., Torn, M. S., Wutzler, T., et al. (2019). Widespread inhibition of daytime ecosystem respiration. Nature Ecology & Evolution, 3, 407–415. doi:10.1038/s41559-019-0809-2."
    ),

    "DML2": (
        "Double Machine Learning (DML) hybrid approach — Cohrs et al. (2024). "
        "Uses a causal inference / double-machine-learning framework to estimate parametric (process) components while letting flexible ML models fit the "
        "remaining signal. DML orthogonalizes nuisance/ML parts (via cross-fitting) so that causal/parametric parameters (e.g., temperature sensitivity, Q10) "
        "are estimated with reduced bias from regularization and equifinality. Applied to flux partitioning, DML provides a principled way to combine "
        "scientific process models and data-driven components while improving interpretability and parameter identifiability. \n\n"
        "Reference: Reference: Cohrs, K.-H., Varando, G., Carvalhais, N., Reichstein, M., & Camps-Valls, G. (2024). Causal hybrid modeling with double machine learning—applications in carbon flux modeling. Machine Learning: Science and Technology, 5(3), 035021. doi:10.1088/2632-2153/ad5a60."
    ),

    "NN": (
        "Neural Network approach (NN) — Tramontana et al. (2020). "
        "Data-driven neural networks (NN C-part) are trained on both day and night half-hourly NEE and auxiliary drivers (radiation, temperature, VPD, soil moisture, etc.) "
        "to simultaneously predict GPP and Reco under a consistency constraint (GPP − Reco = −NEE). NNs capture complex, nonlinear dependencies and can outperform simple "
        "parametric fits where relationships are nonlinear or site-specific, but they require ample training data and careful design to enforce physical consistency, "
        "avoid overfitting, and ensure generalization. \n\n"
        "Reference: Tramontana, G., Migliavacca, M., Jung, M., Reichstein, M., Keenan, T. F., Camps-Valls, G., Ogee, J., Verrelst, J., Papale, D., et al. (2020). Partitioning net carbon dioxide fluxes into photosynthesis and respiration using neural networks. Global Change Biology, 26(9), 5235–5253. doi:10.1111/gcb.15203."
    ),

    "NNWW": (
        "Neural Network (NNWW) — Zhan et al. (2022) style approach. "
        "ML/NN frameworks that extend Tramontana et al. by including additional data streams (e.g., solar-induced chlorophyll fluorescence, SIF) or multi-task objectives "
        "to jointly retrieve GPP and Reco and to constrain the GPP–SIF relationship to improve GPP estimates and reduce ambiguity in partitioning, especially across ecosystems with differing light-use characteristics. "
        "Here we only use the architecture with SIF which is not available at all sites; it does not assume an explicit LUE model for GPP. \n\n"
        "Reference: Zhan, W., Yang, X., Ryu, Y., Dechant, B., Huang, Y., Goulas, Y., Kang, M., & Gentine, P. (2022). Two for one: Partitioning CO2 fluxes and understanding the relationship between solar-induced chlorophyll fluorescence and gross primary productivity using machine learning. Agricultural and Forest Meteorology, 321, 108980. doi:10.1016/j.agrformet.2022.108980."
    ),

    # short description for common NN modulations you mentioned
    "NN3": (
        "Neural Network (NN3) — Tramontana et al. (2020) style approach but trained longer / more thoroughly. \n\n"
        "Reference: - "
    ),
    "NNKGREGL1": (
        "Neural Network (NNKGREGL1) — Tramontana et al. (2020) style approach with a knowledge-guided regularization loss that encourages RECO to stay close to nighttime-method estimates (helps preserve a realistic daily cycle). Uses an L1 loss and extended training. \n\n"
        "Reference: - "
    ),
    "NNL1": (
        "Neural Network (NNL1) — Tramontana et al. (2020) style approach trained longer and using L1 loss for robust residual behaviour. \n\n"
        "Reference: - "
    ),
    "NNPREL1": (
        "Neural Network (NNPREL1) — Tramontana et al. (2020) style approach with pretraining on the five-year window centred on the year being partitioned, then fine-tuning on the target year; uses L1 loss and longer training. Pretraining can improve generalization and stability for short target-year data. \n\n"
        "Reference: - "
    ),
    "NNMRL1": (
        "Neural Network (NNMRL1) — Tramontana et al. (2020) style approach augmented with engineered features (e.g., NEE_SWin_lowlight_lm_slope, NEE_LUE_at_max_SW, NEE_light_night_diff, NEE_night_movmean, TS_night_movmean, TA_night_movmean) and L1 loss; trained longer. Feature engineering can help the NN capture diel/seasonal patterns and reduce overfitting. \n\n"
        "Reference: - "
    ),
    "NNFULL": (
        "Neural Network (NNFULL) — Tramontana et al. (2020) style approach that combines pretraining and knowledge-guided regularization; uses L1 loss and extended training. Intended to leverage both data-driven flexibility and process constraints to improve realism. \n\n"
        "Reference: - "
    ),
}



MINIMIZE_METRICS = {"RMSE", "cRMSE", "Centered RMSE", "MAE", "Absolute Bias", "Bias"}
TARGET_ZERO_METRICS = {"Bias"}
MAXIMIZE_METRICS = {"NSE", "robust_NSE", "Correlation", "KGE"}
# Display-only NSE / Robust NSE axis. Statistics stay on the unclipped values.
NSE_DISPLAY_METRICS = frozenset({"NSE", "robust_NSE"})
NSE_DISPLAY_FLOOR = -0.5
NSE_DISPLAY_CEILING = 1.0


def is_nse_metric(metric: str) -> bool:
    return metric in NSE_DISPLAY_METRICS


def apply_nse_display_axis(ax: plt.Axes, floor: float = NSE_DISPLAY_FLOOR) -> None:
    """Clip the NSE / Robust NSE *view* to [floor, 1], shade below 0, mark 0 and 1.

    *floor* is −0.5 by default (use 0.0 for a [0, 1] view). Does not
    change the samples used for medians, boxes, or other statistics.
    """
    ax.axhspan(floor, 0.0, facecolor="lightgray", alpha=0.45, linewidth=0, zorder=0)
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="-", alpha=0.70, zorder=1)
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.70, zorder=1)
    ax.set_ylim(bottom=floor, top=NSE_DISPLAY_CEILING)


ERROR_DISPLAY_METRICS = frozenset({
    "RMSE", "cRMSE", "Centered RMSE", "MAE", "Absolute Bias", "Bias",
})
DEFAULT_DISPLAY_QUANTILES = (0.10, 0.90)


def _finite_values(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        return arr
    return arr[np.isfinite(arr)]


def _display_quantile_band(display_mode: Optional[str] = None) -> Tuple[float, float]:
    """Quantile band the y-axis must cover. Defaults to 10th–90th."""
    if display_mode:
        band = OVERVIEW_QUANTILE_OPTIONS.get(display_mode)
        if band:
            return band
    return DEFAULT_DISPLAY_QUANTILES


def _finalize_error_limits(lo: float, hi: float, metric: str) -> Optional[Tuple[float, float]]:
    span = hi - lo
    pad = max(0.06 * span, 0.02 * max(abs(hi), 1.0))
    lo, hi = lo - pad, hi + pad
    if metric == "Bias":
        lo = min(lo, 0.0)
        hi = max(hi, 0.0)
    else:
        lo = max(0.0, lo)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    return lo, hi


def grouped_display_limits(
    groups: Iterable[Any],
    metric: str,
    q_low: float = 0.10,
    q_high: float = 0.90,
) -> Optional[Tuple[float, float]]:
    """Display range covering *q_low*–*q_high* of every method × scale cell.

    Each *group* is one depicted cell (one method at one scale / IGBP / …).
    Only points outside that envelope are clipped. Statistics stay on the
    full data.
    """
    lows: List[float] = []
    highs: List[float] = []
    for group in groups:
        vals = _finite_values(group)
        if vals.size == 0:
            continue
        if vals.size == 1:
            lows.append(float(vals[0]))
            highs.append(float(vals[0]))
            continue
        lows.append(float(np.quantile(vals, q_low)))
        highs.append(float(np.quantile(vals, q_high)))
    if not highs:
        return None
    return _finalize_error_limits(min(lows), max(highs), metric)


def coverage_display_limits(
    groups: Iterable[Any],
    metric: str,
    q_low: float = 0.10,
    q_high: float = 0.90,
) -> Optional[Tuple[float, float]]:
    """Robust y-limits from cell-level quantiles across all IGBP × bin × method.

    Each group is one coverage cell. The 10th/90th of every cell is collected,
    then a Tukey inner fence on those cell edges sets the axis so one wild
    bin cannot stretch the view. Cells whose spread exceeds the fence are
    clipped.
    """
    lows: List[float] = []
    highs: List[float] = []
    meds: List[float] = []
    for group in groups:
        vals = _finite_values(group)
        if vals.size == 0:
            continue
        if vals.size == 1:
            v = float(vals[0])
            lows.append(v)
            highs.append(v)
            meds.append(v)
            continue
        lows.append(float(np.quantile(vals, q_low)))
        highs.append(float(np.quantile(vals, q_high)))
        meds.append(float(np.median(vals)))
    if not highs:
        return None

    def _fence(arr: Sequence[float], upper: bool) -> float:
        data = np.asarray(arr, dtype=float)
        q1, q3 = np.percentile(data, [25.0, 75.0])
        iqr = float(q3 - q1)
        if iqr < 1e-12:
            return float(np.median(data))
        return float(q3 + 1.5 * iqr) if upper else float(q1 - 1.5 * iqr)

    lo = min(_fence(lows, False), _fence(meds, False))
    hi = max(_fence(highs, True), _fence(meds, True))
    return _finalize_error_limits(lo, hi, metric)


def robust_display_limits(
    values: Any,
    metric: str,
    q_low: float = 0.10,
    q_high: float = 0.90,
) -> Optional[Tuple[float, float]]:
    """Fallback when values are not already split by method × scale."""
    return grouped_display_limits([values], metric, q_low, q_high)


def count_display_cutouts(
    values_by_method: Dict[str, Any], lo: float, hi: float,
) -> Dict[str, int]:
    """Count finite values of each method that fall outside [lo, hi]."""
    counts: Dict[str, int] = {}
    for method, raw in values_by_method.items():
        vals = _finite_values(raw)
        if vals.size == 0:
            continue
        n_cut = int(np.sum((vals < lo) | (vals > hi)))
        if n_cut:
            counts[method] = n_cut
    return counts


def format_cutout_caption(
    metric: str,
    lo: float,
    hi: float,
    cutouts: Dict[str, int],
    q_low: float = 0.10,
    q_high: float = 0.90,
    cell_label: str = "method × scale",
    robust_cells: bool = False,
) -> str:
    range_txt = f"[{lo:.3g}, {hi:.3g}]"
    q_txt = f"{int(round(q_low * 100))}th–{int(round(q_high * 100))}th"
    envelope = (
        f"Tukey fence on the {q_txt} of each {cell_label}"
        if robust_cells
        else f"covers the {q_txt} percentile of every {cell_label}"
    )
    prefix = (
        f"{metric_with_unit(metric)} display clipped to {range_txt} "
        f"({envelope}; statistics use the full data)."
    )
    if not cutouts:
        return prefix + " No points cut out."
    parts = [
        f"{method_display_name(method)}: {n}"
        for method, n in cutouts.items()
    ]
    return prefix + " Cut out per method: " + ", ".join(parts) + "."


def apply_error_display_axis(ax: plt.Axes, lo: float, hi: float, metric: str) -> None:
    """Clip the *view* for RMSE / cRMSE / MAE / Bias. Does not alter samples."""
    ax.set_ylim(bottom=lo, top=hi)
    if metric in TARGET_ZERO_METRICS or metric == "Bias":
        ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="-", alpha=0.70, zorder=1)


def _attach_error_display_limits(
    axes: Sequence[Any],
    values_by_method: Dict[str, Any],
    metric: str,
    groups: Optional[Sequence[Any]] = None,
    q_low: float = 0.10,
    q_high: float = 0.90,
    cell_label: str = "method × scale",
) -> Optional[str]:
    """Apply a shared robust y-limit to *axes* and return the cut-out caption."""
    if metric not in ERROR_DISPLAY_METRICS:
        return None
    if groups is None:
        groups = [_finite_values(v) for v in values_by_method.values()]
    limits = grouped_display_limits(groups, metric, q_low, q_high)
    if limits is None:
        return None
    lo, hi = limits
    for ax in axes:
        if ax is None:
            continue
        if hasattr(ax, "get_visible") and not ax.get_visible():
            continue
        apply_error_display_axis(ax, lo, hi, metric)
    return format_cutout_caption(
        metric, lo, hi, count_display_cutouts(values_by_method, lo, hi),
        q_low=q_low, q_high=q_high, cell_label=cell_label,
    )


def _coverage_bin_groups(
    work: pd.DataFrame,
    metric: str,
    methods: Sequence[str],
    bins: Sequence[Tuple[float, float]],
    last_inclusive: bool = True,
) -> List[np.ndarray]:
    """One sample array per (IGBP, coverage bin, method) cell."""
    groups: List[np.ndarray] = []
    if work.empty or metric not in work.columns or "_pct" not in work.columns:
        return groups
    n_bins = len(bins)
    igbps = (
        work["IGBP"].dropna().unique().tolist()
        if "IGBP" in work.columns
        else [None]
    )
    for igbp in igbps:
        df_ig = work[work["IGBP"] == igbp] if igbp is not None else work
        for method in methods:
            df_m = df_ig[df_ig["Method"] == method] if "Method" in df_ig.columns else df_ig
            if df_m.empty:
                continue
            for b_idx, (lo, hi) in enumerate(bins):
                if last_inclusive:
                    mask = _in_coverage_bin(df_m["_pct"], lo, hi, b_idx == n_bins - 1)
                else:
                    mask = (df_m["_pct"] >= lo) & (df_m["_pct"] < hi)
                vals = _finite_values(df_m.loc[mask, metric])
                if vals.size:
                    groups.append(vals)
    return groups


def _values_by_method_from_grouped(
    samples: Dict[str, Dict[str, np.ndarray]],
    methods: Sequence[str],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    per_method: Dict[str, List[np.ndarray]] = {m: [] for m in methods}
    chunks: List[np.ndarray] = []
    for per_group in samples.values():
        for method in methods:
            arr = _finite_values(per_group.get(method, []))
            if arr.size:
                per_method[method].append(arr)
                chunks.append(arr)
    stacked = np.concatenate(chunks) if chunks else np.array([])
    merged = {
        method: np.concatenate(parts) if parts else np.array([])
        for method, parts in per_method.items()
    }
    return stacked, merged


# Present in evaluation CSVs but not shown as dashboard metrics.
# R2 is algebraically identical to NSE; CorrelationWarningFlag is a diagnostic bit.
HIDDEN_METRICS = frozenset({"R2", "CorrelationWarningFlag"})
FLUX_METRIC_UNIT = "µmol m⁻² s⁻¹"
DIMENSIONLESS_UNIT = "−"
FLUX_UNIT_METRICS = {
    "RMSE",
    "cRMSE",
    "Centered RMSE",
    "MAE",
    "Absolute Bias",
    "Bias",
    "Predicted Std",
    "Reference Std",
}
DIMENSIONLESS_METRICS = {
    "NSE",
    "robust_NSE",
    "KGE",
    "Correlation",
}


METRIC_DISPLAY_NAMES: Dict[str, str] = {
    "Correlation": "Pearson r²",
}
METRIC_ORDER = (
    "Correlation",
    "NSE",
    "Bias",
    "RMSE",
    "cRMSE",
    "MAE",
    "robust_NSE",
    "KGE",
)
DEFAULT_METRIC = "Correlation"

DEFAULT_EVENTS_ORDER = [
    "full_day",
    "day",
    "night",
    "measured",
    "unmeasured",
    "peak_temperature",
]
EVENT_DISPLAY_NAMES: Dict[str, str] = {
    "full_day": "Full Day",
    "day": "Day",
    "night": "Night",
    "measured": "Measured",
    "unmeasured": "Unmeasured",
    "peak_temperature": "Peak Temperature",
    "rain_pulse": "Rain Pulse",
    "harvest": "Harvest",
}


def metric_display_name(metric: str) -> str:
    return METRIC_DISPLAY_NAMES.get(metric, metric)


def order_metrics(available: Sequence[str]) -> List[str]:
    avail = [m for m in available if m]
    ordered = [m for m in METRIC_ORDER if m in avail]
    ordered.extend(m for m in avail if m not in ordered)
    return ordered


def default_metric_index(metrics: Sequence[str]) -> int:
    if DEFAULT_METRIC in metrics:
        return list(metrics).index(DEFAULT_METRIC)
    return 0


def event_display_name(event: str) -> str:
    return EVENT_DISPLAY_NAMES.get(event, str(event).replace("_", " ").title())


def order_events(events: Iterable[str]) -> List[str]:
    present = [e for e in events if pd.notna(e)]
    ordered = [e for e in DEFAULT_EVENTS_ORDER if e in present]
    ordered.extend(sorted(e for e in present if e not in DEFAULT_EVENTS_ORDER))
    return ordered


def default_event_index(events: Sequence[str]) -> int:
    if "full_day" in events:
        return list(events).index("full_day")
    return 0


def order_scales(scales: Iterable[str]) -> List[str]:
    present = [s for s in scales if pd.notna(s)]
    ordered = [s for s in DEFAULT_TIMESCALES_ORDER if s in present]
    extras = [s for s in present if s not in DEFAULT_TIMESCALES_ORDER]
    extra_priority = list(CROSS_SCALE_EXTRA_HH_SUBSETS)
    ordered.extend(s for s in extra_priority if s in extras)
    ordered.extend(sorted(s for s in extras if s not in extra_priority))
    return ordered


def default_scale_index(scales: Sequence[str]) -> int:
    if "Half-Hourly" in scales:
        return list(scales).index("Half-Hourly")
    return 0


def metric_unit(metric: str) -> Optional[str]:
    if metric in FLUX_UNIT_METRICS:
        return FLUX_METRIC_UNIT
    if metric in DIMENSIONLESS_METRICS or metric.startswith("Improvement_"):
        return DIMENSIONLESS_UNIT
    return None


def metric_with_unit(metric: str) -> str:
    name = metric_display_name(metric)
    unit = metric_unit(metric)
    return name if unit is None else f"{name} ({unit})"


# Gold, Silver, Bronze colors for top 3 rankings (matching heatmap highlighting)
RANK_COLORS = [
    "rgba(255, 215, 0, 0.4)",      # Gold (best) - #FFD700 with transparency
    "rgba(192, 192, 192, 0.4)",    # Silver (second best) - #C0C0C0 with transparency
    "rgba(205, 127, 50, 0.4)"      # Bronze (third best) - #CD7F32 with transparency
]
DEFAULT_TIMESCALES_ORDER = [
    "Half-Hourly",
    "Daily",
    "Monthly Diurnal Cycle",
    "Seasonal Cycle",
    "Daily Anomalies",
    "Inter-Annual Variability",
]
CROSS_SCALE_TIMESCALES = DEFAULT_TIMESCALES_ORDER[:-1]
# Virtual "scales" that derive samples from special subsets of the Half-Hourly data.
# Displayed after the standard timescales on the cross-scale x-axis.
CROSS_SCALE_EXTRA_HH_SUBSETS = [
    "HH Peak-Temp",
    "HH Unmeasured",
    "HH Harvest",
    "HH Rain Pulse",
    "HH Tropics (EBF)",
]
# Summary 2×2 extras — harvest is intentionally omitted.
OVERVIEW_EXTRA_SUBSETS = [
    "HH Unmeasured",
    "HH Peak-Temp",
    "HH Rain Pulse",
]
CROSS_SCALE_DISPLAY_OPTIONS = {
    "10th-90th percentile band": (0.10, 0.90),
    "25th-75th percentile band": (0.25, 0.75),
    "5th-95th percentile band": (0.05, 0.95),
    "Boxplot": "boxplot",
    "Violin": "violin",
}
OVERVIEW_QUANTILE_OPTIONS: Dict[str, Optional[Tuple[float, float]]] = {
    "None": None,
    "5th-95th": (0.05, 0.95),
    "10th-90th": (0.10, 0.90),
    "25th-75th": (0.25, 0.75),
}
DIST_GROUP_OPTIONS = ("Temporal scale", "Stress regime", "IGBP")
DIST_STRESS_KEYS = (
    "HH Peak-Temp",
    "HH Unmeasured",
    "HH Harvest",
    "HH Rain Pulse",
)
DIST_STRESS_LABELS = {
    "HH Peak-Temp": "Peak-Temp",
    "HH Unmeasured": "Gap-filled",
    "HH Harvest": "Harvest",
    "HH Rain Pulse": "Rain Pulse",
}
DIST_METRIC_PRIORITY = METRIC_ORDER
DIST_IGBP_ORDER = (
    "EBF", "ENF", "DNF", "DBF", "MF",
    "SAV", "WSA", "OSH", "CSH",
    "GRA", "CRO", "WET", "WAT", "SNO",
)
METHOD_ORDER_PRIORITY = ["NT", "DT", "TK", "DML", "NN"]

# Short x-axis tick labels for the dense 2×2 overview grid.
_SCALE_SHORT_LABELS: Dict[str, str] = {
    "Half-Hourly":              "HH",
    "Daily":                    "Daily",
    "Monthly Diurnal Cycle":    "MDC",
    "Seasonal Cycle":           "Seas.",
    "Daily Anomalies":          "Anom.",
    "Inter-Annual Variability": "IAV",
    "HH Peak-Temp":             "Peak-T",
    "HH Unmeasured":            "Unmeas.",
    "HH Harvest":               "Harvest",
    "HH Rain Pulse":            "Rain P.",
    "HH Tropics (EBF)":         "Tropics",
}

# Per-method markers for publication-quality plots.
_METHOD_MARKERS: Dict[str, str] = {
    "NT":         "s",   # square
    "DT":         "D",   # diamond
    "TK":         "^",   # triangle up
    "DML":        "v",   # triangle down
    "DML2":       "p",   # pentagon
    "NN":         "o",   # circle
    "NN3":        "P",   # plus-filled
    "NNL1":       "X",   # x-filled
    "NNWW":       "*",   # star
    "NNMRL1":     "H",   # hexagon2
    "NNPREL1":    "h",   # hexagon
    "NNKGREGL1":  "8",   # octagon
    "NNFULL":     "<",   # triangle left
    "all":        ">",   # triangle right
    "all_old":    "1",   # tri-down
    "all_ML":     "2",   # tri-up
    "measured":   "d",   # thin diamond
}

# Parametric methods get dashed lines and warm colours.
PARAMETRIC_METHODS = {"NT", "DT", "TK"}

# Fixed warm colours for the three parametric methods (order: NT, DT, TK).
_PARAMETRIC_COLORS: Dict[str, str] = {
    "NT": "#C0392B",   # deep red
    "DT": "#E67E22",   # burnt orange
    "TK": "#C9A800",   # golden yellow
}

# Stable cold-colour roster for ML / data-driven methods.
# New/unknown methods fall back to the palette by index.
_ML_METHOD_COLOR_ORDER = [
    "DML", "DML2",
    "NN", "NN3", "NNL1", "NNWW", "NNMRL1", "NNPREL1", "NNKGREGL1", "NNFULL",
    "all", "all_old", "all_ML",
    "measured",
]
_COLD_PALETTE: List[str] = [
    "#1565C0",  # dark blue
    "#0097A7",  # teal / cyan
    "#2E7D32",  # dark green
    "#6A1B9A",  # deep purple
    "#0277BD",  # medium blue
    "#00695C",  # dark teal
    "#558B2F",  # olive green
    "#4527A0",  # indigo
    "#01579B",  # steel blue
    "#004D40",  # very dark teal
    "#1B5E20",  # forest green
    "#311B92",  # dark indigo
    "#006064",  # dark cyan
    "#33691E",  # dark lime
    "#1A237E",  # navy
]
_ML_COLORS: Dict[str, str] = {
    m: _COLD_PALETTE[i % len(_COLD_PALETTE)]
    for i, m in enumerate(_ML_METHOD_COLOR_ORDER)
}


def get_method_color(method: str, fallback_idx: int = 0) -> str:
    """Return a stable colour for *method*: warm for parametric, cold for ML."""
    if method in _PARAMETRIC_COLORS:
        return _PARAMETRIC_COLORS[method]
    if method in _ML_COLORS:
        return _ML_COLORS[method]
    return _COLD_PALETTE[fallback_idx % len(_COLD_PALETTE)]


def get_method_marker(method: str) -> str:
    """Return a stable matplotlib marker character for *method*."""
    return _METHOD_MARKERS.get(method, "o")


def scale_short_label(scale: str) -> str:
    """Return an abbreviated tick label for *scale* (for dense x-axes)."""
    return _SCALE_SHORT_LABELS.get(scale, scale)


def sort_methods(methods: Iterable[str]) -> List[str]:
    def priority(method: str) -> Tuple[int, str]:
        if method in METHOD_ORDER_PRIORITY:
            return (METHOD_ORDER_PRIORITY.index(method), method)
        return (len(METHOD_ORDER_PRIORITY), method)

    return sorted(methods, key=priority)


def _get_method_style(method: str, idx: int) -> Dict[str, Any]:
    color = get_method_color(method, idx)
    linestyle = "--" if method in PARAMETRIC_METHODS else "-"
    return {"color": color, "linestyle": linestyle}

DEFAULT_METHODS = {
    "DML2", "DT", "NT", "NN3", "NNKGREGL1", "NNL1",
    "NNMRL1", "NNPREL1", "NNWW", "TK",
}


def paper_methods() -> List[str]:
    """Canonical method set used by every dashboard view."""
    return sort_methods(DEFAULT_METHODS)


def filter_to_paper_methods(df: pd.DataFrame) -> pd.DataFrame:
    """Drop ensemble / extra-NN rows so only DEFAULT_METHODS remain."""
    if df.empty or "Method" not in df.columns:
        return df
    mask = df["Method"].isin(DEFAULT_METHODS)
    if bool(mask.all()):
        return df
    return df.loc[mask]


def available_paper_methods(df: pd.DataFrame) -> List[str]:
    """Paper methods that are actually present in *df*, in canonical order."""
    if df.empty or "Method" not in df.columns:
        return []
    present = set(df["Method"].dropna().unique())
    return [m for m in paper_methods() if m in present]

# ── Display names ──────────────────────────────────────────────────────────────
# Edit this dict to rename methods in all plots and tables.  Internal filtering,
# sorting, and colour assignment always use the raw (key) names, so changes here
# are purely cosmetic and never break any logic.
# Methods not listed fall back to their raw name unchanged.
METHOD_DISPLAY_NAMES: Dict[str, str] = {
    "NT": "NT", "DT": "DT", "TK": "TK",
    "DML": "DML", "DML2": "DML",
    "NN": "NN", "NN3": "NN_{base}",
    "NNL1": r"NN_{L1}", "NNWW": r"NN_{night}", "NNMRL1": r"NN_{FE}",
    "NNPREL1": r"NN_{PT}", "NNKGREGL1": r"NN_{KG}", "NNFULL": r"NN_{Full}",
    "all": "Ensemble (all)", "all_old": "Ensemble (old)", "all_ML": "Ensemble (ML)",
    "measured": "Measured",
}

# LaTeX command strings matching the paper's \newcommand definitions.
METHOD_LATEX_NAMES: Dict[str, str] = {
    "NT":        r"\NT",
    "DT":        r"\DT",
    "TK":        r"\TK",
    "DML":       r"\DMLmethod",
    "DML2":      r"\DMLmethod",
    "NN3":       r"\NNbase",
    "NNL1":      r"\NNL{1}",
    "NNWW":      r"\NNnight",
    "NNMRL1":    r"\NNFE",
    "NNPREL1":   r"\NNPT",
    "NNKGREGL1": r"\NNKGREG",
}


def method_display_name(method: str) -> str:
    """Plain-text display name for tables and non-math contexts.

    LaTeX math notation (``_{…}`` / ``^{…}``) in the dictionary is flattened
    to plain subscripts so that Streamlit DataFrames and LaTeX index columns
    render cleanly without literal braces.
    """
    name = METHOD_DISPLAY_NAMES.get(method, method)
    name = re.sub(r"_\{([^}]+)\}", r"_\1", name)
    name = re.sub(r"\^\{([^}]+)\}", r"^\1", name)
    return name


def method_html_name(method: str) -> str:
    """HTML display name with real ``<sub>`` / ``<sup>`` tags."""
    name = METHOD_DISPLAY_NAMES.get(method, method)
    name = re.sub(r"_\{([^}]+)\}", r"<sub>\1</sub>", name)
    name = re.sub(r"\^\{([^}]+)\}", r"<sup>\1</sup>", name)
    return name


def method_plot_label(method: str) -> str:
    """Matplotlib-ready label for *method*.

    If the dictionary entry contains LaTeX math syntax (``_{…}`` or ``^{…}``)
    the whole string is wrapped in ``$…$`` so matplotlib renders it via its
    built-in math-text engine (subscripts, superscripts, etc.).
    Plain names (no math syntax) are returned unchanged.
    """
    name = METHOD_DISPLAY_NAMES.get(method, method)
    if "_{" in name or "^{" in name:
        return f"${name}$"
    return name


def method_latex_label(method: str) -> str:
    """Return the LaTeX command string for a method row in a LaTeX table.

    Falls back to method_plot_label for any method not in METHOD_LATEX_NAMES.
    Callers must pass escape_index=False to the LaTeX exporter so backslashes
    are written verbatim.
    """
    return METHOD_LATEX_NAMES.get(method, method_plot_label(method))


def _show_fig(fig: plt.Figure, stretch: bool = False) -> None:
    """Render a matplotlib figure and release it immediately."""
    st.pyplot(fig, width="stretch" if stretch else "content")
    plt.close(fig)


try:
    _st_fragment = st.fragment
except AttributeError:  # Streamlit < 1.33
    def _st_fragment(func):  # type: ignore[misc]
        return func


# Columns kept from evaluation CSVs (everything else is dropped on load).
_RESULT_ID_COLUMNS = frozenset({
    "Year", "Flux", "Method", "Scale", "Event", "N", "NaNs", "N_measured",
    "Site", "Lat", "Lon", "IGBP", "event_ID", "event_site_ID",
})


def _csv_usecols(path: Path) -> Optional[List[str]]:
    try:
        header = pd.read_csv(path, nrows=0).columns.tolist()
    except Exception:
        return None
    keep = [c for c in header if c in _RESULT_ID_COLUMNS or c in METRIC_ORDER]
    return keep or None


def _compact_results_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast dtypes so cached evaluation tables fit in Streamlit Cloud RAM."""
    if df.empty:
        return df
    drop = [c for c in ("_source_file", "Data_ID", "Challenge") if c in df.columns]
    if drop:
        df = df.drop(columns=drop, errors="ignore")
    for col in ("Method", "Site", "IGBP", "Scale", "Event", "Flux", "Model"):
        if col in df.columns:
            df[col] = df[col].astype("category")
    float_cols = [c for c in df.columns if c in METRIC_ORDER or c in {"Lat", "Lon", "N_measured"}]
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], downcast="float")
    if "Year" in df.columns:
        converted = pd.to_numeric(df["Year"], errors="coerce")
        if converted.notna().mean() >= 0.5:
            df["Year"] = pd.to_numeric(converted, downcast="integer")
        else:
            df["Year"] = df["Year"].astype("category")
    return df


@functools.lru_cache(maxsize=None)
def _configuration_paths() -> Dict[str, str]:
    config = get_configuration()
    return config["paths"]


def is_event_period(period: str) -> bool:
    return period in EVENT_PERIODS


def period_unit_label(period: str) -> str:
    return "events" if is_event_period(period) else "site-years"


def model_data_id(model: str) -> str:
    """Full evaluation data id including LSM prefix (e.g. quincy_216medium2)."""
    return f"{model}_{DATA_ID}"


def evaluation_file_token(model: str) -> str:
    """Substring that identifies evaluation CSVs for one LSM run."""
    return f"_{model_data_id(model)}_{RUN_ID}"


def _prepare_grouping_columns(df: pd.DataFrame, period: str) -> Optional[pd.DataFrame]:
    """Add ``site_year`` row identifier for heatmaps and ranking summaries."""
    grouped = df.copy()
    if "IGBP" not in grouped.columns or "Site" not in grouped.columns:
        return None

    if is_event_period(period):
        if "event_site_ID" not in grouped.columns:
            return None
        grouped["_group_sort_id"] = grouped["event_site_ID"]
        grouped["site_year"] = (
            grouped["Site"].astype(str)
            + "-E"
            + grouped["event_site_ID"].astype(str)
            + " ("
            + grouped["IGBP"].astype(str)
            + ")"
        )
    else:
        if "Year" not in grouped.columns:
            grouped["Year"] = 1
        grouped["_group_sort_id"] = grouped["Year"]
        grouped["site_year"] = (
            grouped["Site"].astype(str)
            + "-"
            + grouped["Year"].astype(str)
            + " ("
            + grouped["IGBP"].astype(str)
            + ")"
        )
    return grouped


def _deduplicate_metric_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per site-year/event/method/flux when duplicate sources were loaded."""
    base_cols = ["Method", "Site", "Flux", "Scale", "Event"]
    # For event-period data use event_site_ID as the row identifier instead of Year,
    # otherwise all events at a site are collapsed to the last one.
    if "event_site_ID" in df.columns:
        key_cols = base_cols + ["event_site_ID"]
    elif "Year" in df.columns:
        key_cols = base_cols + ["Year"]
    else:
        key_cols = base_cols
    present = [c for c in key_cols if c in df.columns]
    if not present:
        return df
    return df.drop_duplicates(subset=present, keep="last")


def _summarize_lsm_load(model: str, period: str) -> Dict[str, Any]:
    """Diagnostics for which evaluation files were loaded for one LSM."""
    if model not in MODELS or period not in PERIOD_CONFIG:
        return {}

    period_cfg = PERIOD_CONFIG[period]
    paths = _configuration_paths()
    eval_tables_root = Path(paths["results"])
    file_token = evaluation_file_token(model)
    sources: List[str] = []

    if eval_tables_root.exists():
        for method_dir in sorted(p for p in eval_tables_root.iterdir() if p.is_dir()):
            candidate_dir = method_dir / period_cfg["subdir"]
            if not candidate_dir.exists():
                continue
            for csv_file in candidate_dir.glob(f"*{period_cfg['suffix']}"):
                if file_token in csv_file.stem:
                    sources.append(csv_file.name)

    return {
        "model": model,
        "data_id": model_data_id(model),
        "file_token": file_token,
        "results_root": str(eval_tables_root),
        "source_files": sources,
        "source_count": len(sources),
    }


_LFS_POINTER_HEADER = "version https://git-lfs.github.com/spec/"


def _is_lfs_pointer(path: Path) -> bool:
    """Return True if *path* is a Git LFS pointer instead of real file contents."""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return handle.readline().startswith(_LFS_POINTER_HEADER)
    except OSError:
        return False


def lfs_pointer_result_files() -> List[str]:
    """Relative paths of evaluation CSVs that are still Git LFS pointers."""
    try:
        results_root = Path(_configuration_paths()["results"])
    except Exception:
        return []
    if not results_root.is_dir():
        return []
    hits: List[str] = []
    for csv_file in sorted(results_root.rglob("*.csv")):
        if _is_lfs_pointer(csv_file):
            try:
                hits.append(str(csv_file.relative_to(results_root)))
            except ValueError:
                hits.append(str(csv_file))
    return hits


@st.cache_data(show_spinner=False, max_entries=4)
def load_results(model: str, period: str, _version: int = METRICS_LOAD_VERSION) -> pd.DataFrame:
    if model not in MODELS:
        return pd.DataFrame()
    if period not in PERIOD_CONFIG:
        return pd.DataFrame()

    period_cfg = PERIOD_CONFIG[period]
    paths = _configuration_paths()
    eval_tables_root = Path(paths["results"])
    file_token = evaluation_file_token(model)

    frames: List[pd.DataFrame] = []
    if eval_tables_root.exists():
        for method_dir in sorted(p for p in eval_tables_root.iterdir() if p.is_dir()):
            if method_dir.name not in DEFAULT_METHODS:
                continue
            candidate_dir = method_dir / period_cfg["subdir"]
            if not candidate_dir.exists():
                continue

            for csv_file in candidate_dir.glob(f"*{period_cfg['suffix']}"):
                if file_token not in csv_file.stem:
                    continue
                if _is_lfs_pointer(csv_file):
                    continue
                try:
                    df = pd.read_csv(csv_file, usecols=_csv_usecols(csv_file))
                except Exception:
                    continue
                if df.empty:
                    continue
                if "Method" not in df.columns:
                    df["Method"] = method_dir.name
                else:
                    df["Method"] = df["Method"].fillna(method_dir.name)
                df["Model"] = model
                if "Event" not in df.columns:
                    df["Event"] = "full_day"
                if is_event_period(period) and "Scale" in df.columns:
                    df = df.loc[df["Scale"] == "Half-Hourly"]
                frames.append(df)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        _base = ["Method", "Site", "Flux", "Scale", "Event"]
        if "event_site_ID" in combined.columns:
            _key = _base + ["event_site_ID"]
        elif "Year" in combined.columns:
            _key = _base + ["Year"]
        else:
            _key = _base
        _key = [c for c in _key if c in combined.columns]
        if _key:
            combined = combined.drop_duplicates(subset=_key, keep="last")
        return _compact_results_frame(filter_to_paper_methods(combined))

    return pd.DataFrame()


def _lsm_metrics_differ(
    df: pd.DataFrame,
    timescale: str,
    event: str,
    flux: str = "NEE",
    metric: str = "NSE",
) -> Tuple[bool, float]:
    """Return whether Quincy and Cable-Pop raw metrics differ for the same site-years."""
    if "Model" not in df.columns or metric not in df.columns:
        return False, 0.0

    subset = df[
        (df["Scale"] == timescale)
        & (df["Event"] == event)
        & (df["Flux"] == flux)
    ]
    quincy = subset[subset["Model"] == "quincy"]
    cablepop = subset[subset["Model"] == "cablepop"]
    if quincy.empty or cablepop.empty:
        return False, 0.0

    merge_keys = ["Site", "Year", "Method"]
    if "Year" not in quincy.columns:
        merge_keys = ["Site", "Method"]
    merged = quincy.merge(cablepop, on=merge_keys, suffixes=("_q", "_c"), how="inner")
    if merged.empty:
        return False, 0.0

    col_q, col_c = f"{metric}_q", f"{metric}_c"
    if col_q not in merged.columns or col_c not in merged.columns:
        return False, 0.0

    identical_share = float((merged[col_q].round(3) == merged[col_c].round(3)).mean())
    return identical_share < 1.0, identical_share


def load_combined_results(period: str, _version: int = METRICS_LOAD_VERSION) -> pd.DataFrame:
    """Concatenate per-LSM tables. Not cached: `load_results` already is."""
    frames = [load_results(model, period) for model in MODELS]
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _compute_best_share_for_flux_metric(
    df: pd.DataFrame,
    flux: str,
    metric: str,
    period: str,
    tie_decimals: int = 3,
) -> Dict[str, float]:
    """Return the share of site-years/sites where each method ranks first."""
    flux_df = df[df["Flux"] == flux].copy()
    if flux_df.empty or metric not in flux_df.columns:
        return {}

    grouped = _prepare_grouping_columns(flux_df, period)
    if grouped is None:
        return {}

    try:
        matrix = grouped.pivot_table(
            index="site_year",
            columns="Method",
            values=metric,
            aggfunc="first",
        )
    except Exception:
        return {}

    if matrix.empty:
        return {}

    rank_matrix = _identify_top_methods(matrix, metric, n_top=1, tie_decimals=tie_decimals)
    total = len(matrix.index)
    if total == 0:
        return {}

    shares: Dict[str, float] = {}
    for method in matrix.columns:
        best_count = (rank_matrix[method] == 1).sum()
        shares[method] = 100.0 * best_count / total
    return shares


def _leaderboard_frame(
    df: pd.DataFrame,
    timescale: str,
    event: str,
    methods: Optional[Sequence[str]],
    sites: Optional[Sequence[str]] = None,
    igbps: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Filter *df* and return (filtered, method list, LSM display labels)."""
    filtered = df[(df["Scale"] == timescale) & (df["Event"] == event)].copy()
    if methods:
        filtered = filtered[filtered["Method"].isin(methods)]
    if sites:
        filtered = filtered[filtered["Site"].isin(sites)]
    if igbps:
        filtered = filtered[filtered["IGBP"].isin(igbps)]
    table_methods = available_paper_methods(filtered)
    lsm_labels = [MODEL_DISPLAY_NAMES[model] for model in MODELS]
    return filtered, table_methods, lsm_labels


def _build_best_method_share_table(
    df: pd.DataFrame,
    period: str,
    timescale: str,
    event: str,
    metric: str,
    tie_decimals: int = 3,
    methods: Optional[Sequence[str]] = None,
    sites: Optional[Sequence[str]] = None,
    igbps: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Best-method share (%) for one metric: rows = methods, cols = Flux × LSM."""
    filtered, table_methods, lsm_labels = _leaderboard_frame(
        df, timescale, event, methods, sites=sites, igbps=igbps,
    )
    if filtered.empty or not table_methods:
        return pd.DataFrame()

    columns = pd.MultiIndex.from_product(
        [list(LEADERBOARD_FLUXES), lsm_labels],
        names=["Flux", "LSM"],
    )
    table = pd.DataFrame(index=table_methods, columns=columns, dtype=float)

    for model in MODELS:
        model_df = filtered[filtered["Model"] == model]
        if model_df.empty or metric not in model_df.columns:
            continue
        lsm_label = MODEL_DISPLAY_NAMES[model]
        for flux in LEADERBOARD_FLUXES:
            shares = _compute_best_share_for_flux_metric(
                model_df, flux, metric, period, tie_decimals=tie_decimals
            )
            for method, share in shares.items():
                if method in table.index:
                    table.loc[method, (flux, lsm_label)] = share

    table = table.dropna(how="all")
    if table.empty:
        return table
    return table.loc[sort_methods(table.index)]


def _build_metric_value_table(
    df: pd.DataFrame,
    timescale: str,
    event: str,
    metric: str,
    methods: Optional[Sequence[str]] = None,
    sites: Optional[Sequence[str]] = None,
    igbps: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Median (25th/75th) for one metric: rows = methods, cols = Flux × LSM."""
    filtered, table_methods, lsm_labels = _leaderboard_frame(
        df, timescale, event, methods, sites=sites, igbps=igbps,
    )
    if filtered.empty or not table_methods or metric not in filtered.columns:
        return pd.DataFrame(), pd.DataFrame()

    columns = pd.MultiIndex.from_product(
        [list(LEADERBOARD_FLUXES), lsm_labels],
        names=["Flux", "LSM"],
    )
    display = pd.DataFrame(index=table_methods, columns=columns, dtype=object)
    numeric = pd.DataFrame(index=table_methods, columns=columns, dtype=float)

    for model in MODELS:
        model_df = filtered[filtered["Model"] == model]
        if model_df.empty:
            continue
        lsm_label = MODEL_DISPLAY_NAMES[model]
        for flux in LEADERBOARD_FLUXES:
            flux_df = model_df[model_df["Flux"] == flux]
            for method in table_methods:
                values = flux_df.loc[flux_df["Method"] == method, metric].dropna()
                if values.empty:
                    display.loc[method, (flux, lsm_label)] = "—"
                    numeric.loc[method, (flux, lsm_label)] = float("nan")
                else:
                    median = float(values.median())
                    p25 = float(values.quantile(0.25))
                    p75 = float(values.quantile(0.75))
                    display.loc[method, (flux, lsm_label)] = f"{median:.3f} ({p25:.3f}/{p75:.3f})"
                    numeric.loc[method, (flux, lsm_label)] = median

    display = display.dropna(how="all")
    if display.empty:
        return pd.DataFrame(), pd.DataFrame()
    order = sort_methods(display.index)
    return display.loc[order], numeric.loc[order]


def _column_rank_colors(
    values: pd.Series,
    *,
    higher_is_better: bool,
    target_zero: bool,
    abs_tol: float,
) -> Dict[Any, str]:
    """Assign gold / silver / bronze to the top 3 distinct ranks in *values*."""
    valid = values.dropna()
    if valid.empty:
        return {}

    if target_zero:
        ordered = valid.reindex(valid.abs().sort_values().index)
    elif higher_is_better:
        ordered = valid.sort_values(ascending=False)
    else:
        ordered = valid.sort_values()

    colors: Dict[Any, str] = {}
    color_idx = 0
    prev: Optional[float] = None
    for idx, val in ordered.items():
        cmp_val = float(abs(val) if target_zero else val)
        if prev is not None and not math.isclose(cmp_val, prev, abs_tol=abs_tol):
            color_idx += 1
        if color_idx >= len(RANK_COLORS):
            break
        colors[idx] = RANK_COLORS[color_idx]
        prev = cmp_val
    return colors


def _style_rank_table(
    display: pd.DataFrame,
    numeric: pd.DataFrame,
    metric: str,
    *,
    ranking_mode: bool,
) -> Any:
    if ranking_mode:
        higher_is_better = True
        target_zero = False
    else:
        target_zero = metric in TARGET_ZERO_METRICS
        higher_is_better = metric in MAXIMIZE_METRICS
    abs_tol = 0.05 if ranking_mode else 0.001

    def highlight_column(col: pd.Series) -> List[str]:
        if col.name not in numeric.columns:
            return [""] * len(col)
        colors = _column_rank_colors(
            numeric[col.name],
            higher_is_better=higher_is_better,
            target_zero=target_zero,
            abs_tol=abs_tol,
        )
        return [
            f"background-color: {colors[idx]}; font-weight: 600;" if idx in colors else ""
            for idx in col.index
        ]

    styler = display.style.apply(highlight_column, axis=0)
    if ranking_mode:
        styler = styler.format("{:.1f}%", na_rep="—")
    else:
        styler = styler.format(na_rep="—")
    return styler


def _render_rank_table_html(
    display: pd.DataFrame,
    numeric: pd.DataFrame,
    metric: str,
    *,
    ranking_mode: bool,
) -> None:
    """Render the comparison table as HTML so method subscripts display."""
    if ranking_mode:
        higher_is_better = True
        target_zero = False
        abs_tol = 0.05
    else:
        target_zero = metric in TARGET_ZERO_METRICS
        higher_is_better = metric in MAXIMIZE_METRICS
        abs_tol = 0.001

    col_colors: Dict[Any, Dict[Any, str]] = {}
    for col in numeric.columns:
        col_colors[col] = _column_rank_colors(
            numeric[col],
            higher_is_better=higher_is_better,
            target_zero=target_zero,
            abs_tol=abs_tol,
        )

    def _fmt_cell(row: Any, col: Any) -> str:
        val = display.loc[row, col] if col in display.columns else None
        if val is None or (isinstance(val, float) and (math.isnan(val) or pd.isna(val))):
            return "—"
        if ranking_mode and isinstance(val, (int, float)):
            return f"{val:.1f}%"
        return str(val)

    fluxes = []
    if isinstance(display.columns, pd.MultiIndex):
        for flux, _lsm in display.columns:
            if flux not in fluxes:
                fluxes.append(flux)
        n_lsm = max(1, display.columns.levshape[1] if display.columns.nlevels > 1 else 1)
    else:
        fluxes = list(display.columns)
        n_lsm = 1

    header1 = ['<th rowspan="2" style="text-align:left;vertical-align:bottom;">Method</th>']
    header2: List[str] = []
    if isinstance(display.columns, pd.MultiIndex):
        for flux in fluxes:
            header1.append(
                f'<th colspan="{n_lsm}" style="text-align:center;">{flux}</th>'
            )
        for flux, lsm in display.columns:
            header2.append(f'<th style="text-align:center;">{lsm}</th>')
    else:
        for col in display.columns:
            header1.append(f'<th style="text-align:center;">{col}</th>')

    body_rows: List[str] = []
    for method in display.index:
        cells = [
            f'<td style="text-align:left;white-space:nowrap;">{method_html_name(method)}</td>'
        ]
        for col in display.columns:
            color = col_colors.get(col, {}).get(method)
            style = "text-align:right;"
            if color:
                style += f"background-color:{color};font-weight:600;"
            cells.append(f'<td style="{style}">{_fmt_cell(method, col)}</td>')
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    thead = "<tr>" + "".join(header1) + "</tr>"
    if header2:
        thead += "<tr>" + "".join(header2) + "</tr>"

    html = (
        '<table style="width:100%;border-collapse:collapse;font-size:0.95rem;">'
        f"<thead>{thead}</thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
    )
    st.markdown(html, unsafe_allow_html=True)


@_st_fragment
def render_best_method_leaderboard(
    methods: Optional[Sequence[str]] = None,
) -> Optional[Tuple[str, str, str, pd.DataFrame]]:
    """Method-comparison table with its own period / timescale / metric / filters.

    Returns ``(period, timescale, event, combined_df)`` for later sections, or
    ``None`` when no data is available.
    """
    st.subheader("Aggregate Method Ranking")

    period = st.radio(
        "Period",
        PERIOD_OPTIONS,
        index=PERIOD_OPTIONS.index("yearly") if "yearly" in PERIOD_OPTIONS else 0,
        horizontal=True,
        key="leaderboard_period",
        format_func=lambda p: {
            "yearly": "yearly",
            "across_years": "across_years",
            "pulse_event": "pulse_event",
        }.get(p, p),
    )

    show_plot_info("method_ranking", period, header=False)

    df = load_combined_results(period)
    if df.empty:
        st.warning("No results found for the selected period.")
        return None
    if methods:
        df = df[df["Method"].isin(methods)]

    timescales = available_timescales(df)
    if is_event_period(period) or not timescales:
        timescale = "Half-Hourly"
        st.caption("Event evaluations use half-hourly metrics only.")
    else:
        timescale = st.selectbox(
            "Time-scale",
            options=timescales,
            index=default_scale_index(timescales),
            format_func=scale_short_label,
            key="leaderboard_timescale",
        )

    events = available_events(df)
    event = "full_day" if "full_day" in events else (events[0] if events else "full_day")

    avail_metrics = [m for m in metric_columns(df) if m in df.columns]
    if not avail_metrics:
        st.info("No metrics available for the comparison table.")
        return period, timescale, event, df
    default_metric = DEFAULT_METRIC if DEFAULT_METRIC in avail_metrics else avail_metrics[0]

    col_v, col_m, col_d = st.columns([2, 3, 2])
    with col_v:
        table_view = st.radio(
            "Table view",
            options=("Ranking", "Values"),
            index=0,
            horizontal=True,
            key="leaderboard_view",
            help=(
                "Ranking: share of site-years / events where a method ranks first. "
                "Values: median (25th/75th percentile) of the selected metric."
            ),
        )
    with col_m:
        selected_metric = st.selectbox(
            "Metric",
            options=avail_metrics,
            index=default_metric_index(avail_metrics),
            format_func=metric_with_unit,
            key="leaderboard_metric",
        )
    with col_d:
        tie_decimals = 3
        if table_view == "Ranking":
            tie_decimals = st.selectbox(
                "Tie precision (decimals)",
                options=[1, 2, 3, 4, 5, 6],
                index=2,
                key="leaderboard_tie_decimals",
                help=(
                    "Decimal places used when deciding whether two methods "
                    "are tied for rank 1."
                ),
            )

    site_options = sorted(df["Site"].dropna().unique()) if "Site" in df.columns else []
    igbp_options = sorted(df["IGBP"].dropna().unique()) if "IGBP" in df.columns else []
    filt_c1, filt_c2 = st.columns(2)
    with filt_c1:
        selected_sites = st.multiselect(
            "Sites (optional)",
            options=site_options,
            default=[],
            key="leaderboard_sites",
            help="Leave empty to include all sites.",
        )
    with filt_c2:
        selected_igbps = st.multiselect(
            "IGBP (optional)",
            options=igbp_options,
            default=[],
            key="leaderboard_igbps",
            help="Leave empty to include all IGBP types.",
        )
    sites = selected_sites or None
    igbps = selected_igbps or None

    if st.button("Generate comparison table", key="leaderboard_button"):
        unit = period_unit_label(period)
        ranking_mode = table_view == "Ranking"
        if ranking_mode:
            st.caption(
                f"Share of {unit} where a method ranks first on {metric_with_unit(selected_metric)} "
                f"({timescale}). Gold / silver / bronze mark 1st / 2nd / 3rd. Ties share a rank."
            )
            table = _build_best_method_share_table(
                df, period, timescale, event,
                metric=selected_metric,
                tie_decimals=tie_decimals,
                methods=methods,
                sites=sites,
                igbps=igbps,
            )
            numeric = table
            display = table
        else:
            st.caption(
                f"Median (25th/75th percentile) of {metric_with_unit(selected_metric)} "
                f"({timescale}). Gold / silver / bronze mark 1st / 2nd / 3rd."
            )
            display, numeric = _build_metric_value_table(
                df, timescale, event, selected_metric,
                methods=methods, sites=sites, igbps=igbps,
            )
            table = numeric

        if display.empty:
            st.info("No data available for the selected period, time-scale, metric, and filters.")
        else:
            _render_rank_table_html(
                display, numeric, selected_metric, ranking_mode=ranking_mode,
            )

    st.markdown("---")
    render_grouped_method_distributions(
        period=period,
        methods=list(methods) if methods else available_paper_methods(df),
        sites=sites,
        igbps=igbps,
        default_metric=selected_metric,
    )

    return period, timescale, event, df


def available_timescales(df: pd.DataFrame) -> List[str]:
    unique = df["Scale"].dropna().unique().tolist()
    ordered = order_scales(unique)
    return ordered if ordered else sorted(unique)


def available_events(df: pd.DataFrame) -> List[str]:
    return order_events(df["Event"].dropna().unique().tolist())


def metric_columns(df: pd.DataFrame) -> List[str]:
    excluded = META_COLUMNS.union(HIDDEN_METRICS).union({"Predicted Std", "Reference Std"})
    found = [
        col for col in df.columns
        if col not in excluded and df[col].dtype.kind in ("f", "i")
    ]
    return order_metrics(found)


def summarise_metrics(df: pd.DataFrame, metrics: Iterable[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    filtered_metrics = [m for m in metrics if m not in HIDDEN_METRICS and m not in {"Predicted Std", "Reference Std"}]
    records_display: Dict[str, Dict[str, str]] = {}
    records_median: Dict[str, Dict[str, float]] = {}
    methods = sort_methods(df["Method"].dropna().unique())
    for method in methods:
        sub = df[df["Method"] == method]
        if sub.empty:
            continue
        row_display: Dict[str, str] = {}
        row_median: Dict[str, float] = {}
        for metric in filtered_metrics:
            values = sub[metric].dropna()
            if values.empty:
                row_display[metric] = "—"
                row_median[metric] = float("nan")
            else:
                median = values.median()
                p10 = values.quantile(0.10)
                p90 = values.quantile(0.90)
                row_display[metric] = f"{median:.3f} ({p10:.3f}/{p90:.3f})"
                row_median[metric] = float(median)
        records_display[method] = row_display
        records_median[method] = row_median

    summary_display = pd.DataFrame.from_dict(records_display, orient="index")
    summary_display.index.name = "Method"
    summary_display = summary_display.loc[sort_methods(summary_display.index)]

    summary_median = pd.DataFrame.from_dict(records_median, orient="index")
    summary_median.index.name = "Method"
    summary_median = summary_median.loc[summary_display.index]
    return summary_display, summary_median


def apply_filters(
    df: pd.DataFrame,
    timescale: str,
    event: str,
    methods: List[str],
    sites: Optional[List[str]],
    igbps: Optional[List[str]],
) -> pd.DataFrame:
    filtered = df[(df["Scale"] == timescale) & (df["Event"] == event)]
    if methods:
        filtered = filtered[filtered["Method"].isin(methods)]
    if sites:
        filtered = filtered[filtered["Site"].isin(sites)]
    if igbps:
        filtered = filtered[filtered["IGBP"].isin(igbps)]
    return filtered


def apply_filters_without_timescale(
    df: pd.DataFrame,
    event: str,
    methods: List[str],
    sites: Optional[List[str]],
    igbps: Optional[List[str]],
) -> pd.DataFrame:
    filtered = df[df["Event"] == event]
    if methods:
        filtered = filtered[filtered["Method"].isin(methods)]
    if sites:
        filtered = filtered[filtered["Site"].isin(sites)]
    if igbps:
        filtered = filtered[filtered["IGBP"].isin(igbps)]
    return filtered


def available_cross_scale_timescales(period: str, df: pd.DataFrame) -> List[str]:
    """All timescales present in *df* that can appear on the cross-scale x-axis.

    Inter-Annual Variability is included only when *period* is ``across_years``.
    """
    in_data = set(df["Scale"].dropna().unique())
    return [
        s for s in DEFAULT_TIMESCALES_ORDER
        if s in in_data and (s != "Inter-Annual Variability" or period == "across_years")
    ]


def cross_scale_timescales(
    period: str,
    df: pd.DataFrame,
    selected_scales: Optional[List[str]],
) -> List[str]:
    """Return the ordered list of base timescales to show on the cross-scale plot.

    *selected_scales* is the subset chosen by the user (multiselect).  When
    ``None`` all available timescales are returned (backward-compatible).
    """
    available = available_cross_scale_timescales(period, df)
    if selected_scales is None:
        return available
    sel_set = set(selected_scales)
    return [s for s in available if s in sel_set]


def _build_extra_subset_dfs(
    df: pd.DataFrame,
    model: str,
    period: str,
    extra_labels: List[str],
    methods: List[str],
    sites: Optional[List[str]],
    igbps: Optional[List[str]],
) -> Dict[str, pd.DataFrame]:
    """Build a per-label DataFrame for each selected extra HH subset.

    The returned DataFrames carry the same columns as a regular evaluation
    CSV, so they can be fed straight into ``_collect_cross_scale_samples``.

    Data sources
    ------------
    * "HH Peak-Temp"   — Scale=="Half-Hourly", Event=="peak_temperature" in *df*.
    * "HH Unmeasured"  — Scale=="Half-Hourly", Event=="unmeasured" in *df*.
    * "HH Harvest"     — ``load_results(model, "harvest_event")``, Scale=="Half-Hourly".
    * "HH Rain Pulse"  — ``load_results(model, "pulse_event")``,  Scale=="Half-Hourly".
    * "HH Tropics (EBF)" — Scale=="Half-Hourly", IGBP=="EBF" (ignores *igbps* filter).
    """

    def _filter(src: pd.DataFrame, override_igbp: Optional[str] = None) -> pd.DataFrame:
        out = src
        if methods:
            out = out[out["Method"].isin(methods)]
        if sites:
            out = out[out["Site"].isin(sites)]
        if override_igbp is not None:
            if "IGBP" in out.columns:
                out = out[out["IGBP"] == override_igbp]
        elif igbps:
            out = out[out["IGBP"].isin(igbps)]
        return out

    result: Dict[str, pd.DataFrame] = {}

    for label in extra_labels:
        if label == "HH Peak-Temp":
            sub = _filter(df)
            sub = sub[(sub["Scale"] == "Half-Hourly") & (sub["Event"] == "peak_temperature")]
            result[label] = sub.copy()

        elif label == "HH Unmeasured":
            sub = _filter(df)
            sub = sub[(sub["Scale"] == "Half-Hourly") & (sub["Event"] == "unmeasured")]
            result[label] = sub.copy()

        elif label == "HH Harvest":
            harvest_df = load_results(model, "harvest_event")
            if not harvest_df.empty:
                sub = _filter(harvest_df)
                sub = sub[sub["Scale"] == "Half-Hourly"]
                result[label] = sub.copy()
            else:
                result[label] = pd.DataFrame()

        elif label == "HH Rain Pulse":
            pulse_df = load_results(model, "pulse_event")
            if not pulse_df.empty:
                sub = _filter(pulse_df)
                sub = sub[sub["Scale"] == "Half-Hourly"]
                result[label] = sub.copy()
            else:
                result[label] = pd.DataFrame()

        elif label == "HH Tropics (EBF)":
            sub = _filter(df, override_igbp="EBF")
            sub = sub[sub["Scale"] == "Half-Hourly"]
            result[label] = sub.copy()

    return result


def _clip_for_cross_scale(metric: str, values: pd.Series, skip_lower_clip: bool) -> pd.Series:
    """Optional lower clip for maximize metrics on the cross-scale plot.

    RMSE / cRMSE / MAE / Bias are left unclipped so medians and bands use
    the full data. Display limits are applied later via a robust fence.
    """
    if metric in MAXIMIZE_METRICS and not skip_lower_clip:
        values = values.clip(lower=0.0)
    return values


def _collect_cross_scale_samples(
    df: pd.DataFrame,
    metric: str,
    flux: str,
    methods: Sequence[str],
    timescales: Sequence[str],
    extra_scale_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    skip_lower_clip: bool = False,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Collect metric arrays for each (method, scale) combination.

    *extra_scale_dfs* maps a virtual scale label (e.g. "HH Peak-Temp") to a
    pre-filtered DataFrame.  Their samples are appended after the regular
    timescales under the corresponding label key.

    *skip_lower_clip* — when True the lower=0 floor is not applied for
    MAXIMIZE_METRICS (e.g. NSE), so real negative values reach the caller.
    """
    samples: Dict[str, Dict[str, np.ndarray]] = {method: {} for method in methods}
    subset = df[(df["Flux"] == flux) & df[metric].notna()].copy()
    subset[metric] = _clip_for_cross_scale(metric, subset[metric], skip_lower_clip)

    for method in methods:
        method_df = subset[subset["Method"] == method]
        for scale in timescales:
            values = method_df.loc[method_df["Scale"] == scale, metric].dropna().to_numpy(dtype=float)
            samples[method][scale] = values

        if extra_scale_dfs:
            for label, extra_df in extra_scale_dfs.items():
                if extra_df.empty or metric not in extra_df.columns or "Flux" not in extra_df.columns:
                    samples[method][label] = np.array([])
                    continue
                edf = extra_df[(extra_df["Flux"] == flux) & extra_df[metric].notna()].copy()
                edf[metric] = _clip_for_cross_scale(metric, edf[metric], skip_lower_clip)
                method_edf = edf[edf["Method"] == method] if "Method" in edf.columns else edf
                samples[method][label] = method_edf[metric].dropna().to_numpy(dtype=float)

    return samples


def _plot_cross_scale_performance(
    df: pd.DataFrame,
    metric: str,
    flux: str,
    methods: Sequence[str],
    timescales: Sequence[str],
    display_mode: str,
    period: str,
    event: str,
    extra_scale_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    n_base_timescales: int = 0,
    show_uncertainty: bool = True,
    suptitle: str = "",
) -> Optional[plt.Figure]:
    """Plot metric distributions across timescales.

    *extra_scale_dfs* maps virtual scale labels (e.g. "HH Peak-Temp") to
    pre-filtered DataFrames; *n_base_timescales* is the number of standard
    timescale columns before the extra subsets start (used to draw a visual
    separator).  When *show_uncertainty* is ``False`` only the median/central
    statistic is rendered (no shaded bands, dotted lines, whiskers, or violin
    bodies).
    """
    if not methods or not timescales:
        return None

    # For metrics where negative values are meaningful (e.g. NSE < 0 = below
    # climatology skill), lift the lower=0 clip so the plot shows real values.
    skip_lower = metric in MAXIMIZE_METRICS
    samples = _collect_cross_scale_samples(
        df, metric, flux, methods, timescales, extra_scale_dfs,
        skip_lower_clip=skip_lower,
    )
    methods = sort_methods(methods)
    x_positions = np.arange(len(timescales))
    fig, ax = plt.subplots(figsize=(max(10, len(timescales) * 1.8), 10))
    unit = period_unit_label(period)

    if display_mode in {"Boxplot", "Violin"}:
        n_methods = len(methods)
        group_width = 0.8
        slot_width = group_width / max(n_methods, 1)
        for s_idx, scale in enumerate(timescales):
            for m_idx, method in enumerate(methods):
                values = samples[method].get(scale, np.array([]))
                if values.size == 0:
                    continue
                position = s_idx + (m_idx - (n_methods - 1) / 2) * slot_width
                color = get_method_color(method, m_idx)
                if display_mode == "Boxplot":
                    if show_uncertainty:
                        ax.boxplot(
                            values,
                            positions=[position],
                            widths=slot_width * 0.85,
                            patch_artist=True,
                            boxprops={"facecolor": color, "alpha": 0.55, "linewidth": 1.0},
                            medianprops={"color": "black", "linewidth": 1.2},
                            whiskerprops={"color": color, "linewidth": 1.0},
                            capprops={"color": color, "linewidth": 1.0},
                            flierprops={"marker": "o", "markersize": 2, "alpha": 0.4, "color": color},
                            showfliers=False,
                        )
                    else:
                        median_val = float(np.median(values))
                        ax.plot(
                            [position],
                            [median_val],
                            marker="o",
                            markersize=6,
                            color=color,
                            linestyle="none",
                        )
                else:
                    parts = ax.violinplot(
                        [values],
                        positions=[position],
                        widths=slot_width * 0.9,
                        showmeans=False,
                        showmedians=True,
                        showextrema=False,
                    )
                    if show_uncertainty:
                        for body in parts["bodies"]:
                            body.set_facecolor(color)
                            body.set_edgecolor(color)
                            body.set_alpha(0.55)
                    else:
                        for body in parts["bodies"]:
                            body.set_visible(False)
                    if "cmedians" in parts:
                        parts["cmedians"].set_color("black")
                        parts["cmedians"].set_linewidth(1.2)
        for m_idx, method in enumerate(methods):
            ax.plot([], [], color=get_method_color(method, m_idx), linewidth=2, label=method_plot_label(method))
    else:
        q_low, q_high = CROSS_SCALE_DISPLAY_OPTIONS[display_mode]
        for m_idx, method in enumerate(methods):
            style = _get_method_style(method, m_idx)
            color = style["color"]
            medians: List[float] = []
            lowers: List[float] = []
            uppers: List[float] = []
            valid_x: List[float] = []
            for s_idx, scale in enumerate(timescales):
                values = samples[method].get(scale, np.array([]))
                if values.size == 0:
                    continue
                valid_x.append(float(s_idx))
                medians.append(float(np.median(values)))
                lowers.append(float(np.quantile(values, q_low)))
                uppers.append(float(np.quantile(values, q_high)))
            if not valid_x:
                continue
            x_arr = np.array(valid_x)
            medians_arr = np.array(medians)
            lowers_arr = np.array(lowers)
            uppers_arr = np.array(uppers)
            if show_uncertainty:
                ax.fill_between(
                    x_arr,
                    lowers_arr,
                    uppers_arr,
                    color=color,
                    alpha=0.12,
                    linewidth=0,
                )
            ax.plot(
                x_arr,
                medians_arr,
                label=method_plot_label(method),
                marker="o",
                markersize=5,
                linewidth=1.8,
                **style,
            )
            if show_uncertainty:
                ax.plot(
                    x_arr,
                    lowers_arr,
                    color=color,
                    linewidth=1.0,
                    alpha=0.85,
                    linestyle=":",
                )
                ax.plot(
                    x_arr,
                    uppers_arr,
                    color=color,
                    linewidth=1.0,
                    alpha=0.85,
                    linestyle=":",
                )

    if is_nse_metric(metric):
        apply_nse_display_axis(ax)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(list(timescales), rotation=20, ha="right")
    ax.set_ylabel(metric_with_unit(metric))
    ax.set_xlabel("Timescale")
    aggregation_label = "site-years" if period == "yearly" else period_unit_label(period)
    ax.set_title(
        f"{flux} {metric_with_unit(metric)} across timescales ({aggregation_label}, event={event})"
    )
    ax.grid(True, axis="y", alpha=0.3, linestyle=":")

    # Draw a vertical separator between base timescales and extra HH subsets when
    # both groups are present.
    if extra_scale_dfs and n_base_timescales > 0 and n_base_timescales < len(timescales):
        sep_x = n_base_timescales - 0.5
        ax.axvline(sep_x, color="gray", linewidth=1.0, linestyle="--", alpha=0.5)
        # Italicise x-tick labels for the extra subsets
        tick_labels = ax.get_xticklabels()
        for i, lbl in enumerate(tick_labels):
            if i >= n_base_timescales:
                lbl.set_fontstyle("italic")
                lbl.set_color("#444444")

    ax.legend(loc="best", fontsize=9)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def _apply_metric_clipping(metric: str, values: pd.Series) -> pd.Series:
    """
    Apply appropriate clipping to metric values based on metric type.
    This ensures consistent handling across all plots.
    """
    if metric in TARGET_ZERO_METRICS:
        # For target zero metrics, use absolute values
        values = values.abs()
    
    # For minimize metrics, apply appropriate clipping
    if metric == "RMSE":
        values = values.clip(lower=0.0, upper=4.0)
    elif metric in {"MAE", "cRMSE", "Centered RMSE", "Absolute Bias", "Bias"}:
        # Clip to reasonable range, but allow data-driven max
        values = values.clip(lower=0.0)
        # Cap at 4.0 for consistency with RMSE
        if values.max() > 4.0:
            values = values.clip(upper=4.0)
    
    # For maximize metrics, clip negative values to 0 (start from 0)
    if metric in MAXIMIZE_METRICS:
        values = values.clip(lower=0.0)
    
    return values


def _default_thresholds(metric: str, values: pd.Series) -> np.ndarray:
    clean_values = values.dropna()
    if metric in TARGET_ZERO_METRICS:
        clean_values = clean_values.abs()
    
    # For minimize metrics, set appropriate max thresholds
    if metric == "RMSE":
        clean_values = clean_values.clip(lower=0.0, upper=4.0)
        vmin = 1e-3
        vmax = 4.0
        return np.linspace(vmin, vmax, 101)
    elif metric in {"MAE", "cRMSE", "Centered RMSE", "Absolute Bias", "Bias"}:
        # For other minimize metrics, use a reasonable max based on data or default
        if clean_values.empty:
            vmax = 4.0
        else:
            vmax = max(4.0, clean_values.max() * 0.25)
        vmin = 1e-3
        vmax = min(vmax, 4.0)
        return np.linspace(vmin, vmax, 101)
    
    if clean_values.empty:
        # Default for maximize metrics: 0 to 1
        if metric in MAXIMIZE_METRICS:
            return np.linspace(0.0, 1.0, 101)
        return np.linspace(0.0, 1.0, 101)

    # For maximize metrics (NSE, robust_NSE, Correlation, KGE), start from 0
    if metric in MAXIMIZE_METRICS:
        if metric == "Correlation":
            # Correlation can be -1 to 1, but we only show from 0 onwards
            lower = 0.0
            upper = min(1.0, max(0.0, clean_values.max()))
            if math.isclose(lower, upper):
                upper = min(upper + 1e-3, 1.0)
            return np.linspace(lower, upper, 101)
        elif metric in {"NSE", "robust_NSE", "KGE"}:
            # These metrics can be negative, but we only show from 0 onwards
            lower = 0.0
            upper = max(0.0, clean_values.max())
            if math.isclose(lower, upper):
                upper = max(upper + 1e-3, 1.0)
            else:
                upper = max(upper, 1.0)  # Cap at 1 for these metrics
            return np.linspace(lower, upper, 101)

    if metric in TARGET_ZERO_METRICS:
        lower = clean_values.min()
        upper = clean_values.max()
        if math.isclose(lower, upper):
            upper += 1e-3
        return np.linspace(lower, upper, 101)

    # Default case: use data range
    lower = clean_values.min()
    upper = clean_values.max()
    if math.isclose(lower, upper):
        upper += 1e-3
    return np.linspace(lower, upper, 101)


def _compute_fraction_above_thresholds(
    df: pd.DataFrame,
    metric: str,
    thresholds: np.ndarray,
) -> Dict[str, List[float]]:
    fractions: Dict[str, List[float]] = {}
    minimize = metric in MINIMIZE_METRICS
    target_zero = metric in TARGET_ZERO_METRICS

    for method in sorted(df["Method"].dropna().unique()):
        method_data = df[df["Method"] == method]
        metric_values = method_data[metric].dropna()
        # Apply consistent clipping for all metrics
        metric_values = _apply_metric_clipping(metric, metric_values)
        if metric_values.empty:
            fractions[method] = [np.nan] * len(thresholds)
            continue

        method_fractions = []
        total = len(metric_values)
        for threshold in thresholds:
            if minimize or target_zero:
                fraction = float((metric_values <= threshold).sum() / total)
            else:
                fraction = float((metric_values >= threshold).sum() / total)
            method_fractions.append(fraction)
        fractions[method] = method_fractions
    return fractions


def _plot_fraction_curves(
    thresholds: np.ndarray,
    fractions: Dict[str, List[float]],
    metric: str,
    suptitle: str = "",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, method in enumerate(sort_methods(fractions.keys())):
        values = fractions[method]
        numeric = np.asarray(values, dtype=float)
        if np.isnan(numeric).all():
            continue
        style = _get_method_style(method, idx)
        ax.plot(thresholds, numeric, label=method_plot_label(method), **style)

    x_label_metric = (
        f"|{metric}|" if metric in TARGET_ZERO_METRICS else metric_with_unit(metric)
    )
    ax.set_xlabel(f"{x_label_metric} threshold")
    
    # Update y-axis label based on metric type
    minimize = metric in MINIMIZE_METRICS
    target_zero = metric in TARGET_ZERO_METRICS
    if minimize or target_zero:
        ax.set_ylabel("Fraction Below Threshold")
    else:
        ax.set_ylabel("Fraction Above Threshold")
    
    ax.set_ylim(0, 1)

    # Set x-axis limits for minimize metrics
    if metric in MINIMIZE_METRICS:
        if metric == "RMSE":
            ax.set_xlim(0, 4)
        else:
            # For other minimize metrics, use the threshold range
            ax.set_xlim(thresholds.min(), thresholds.max())
    elif metric in MAXIMIZE_METRICS:
        # For maximize metrics, ensure we show from 0 onwards
        ax.set_xlim(max(0, thresholds.min()), thresholds.max())
    
    ax.grid(alpha=0.3)
    ax.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left")
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def _compute_auc(fractions: Dict[str, List[float]], thresholds: np.ndarray) -> pd.DataFrame:
    records = []
    for method, values in fractions.items():
        numeric_values = np.asarray(values, dtype=float)
        if np.isnan(numeric_values).all():
            auc_value = np.nan
        else:
            auc_value = float(np.trapz(numeric_values, thresholds))
        records.append({"Method": method, "AUC": auc_value})
    auc_df = pd.DataFrame(records).sort_values("AUC", ascending=False)
    return auc_df


class TaylorDiagram:
    """Taylor diagram implementation (adapted from Yannick Copin's example)."""

    def __init__(self, ref_std: float, ax: plt.Axes, max_std: Optional[float] = None):
        self.ref_std = ref_std
        self.ax = ax

        self.ax.set_theta_direction(-1)
        self.ax.set_theta_zero_location("E")
        self.ax.set_thetamin(0)
        self.ax.set_thetamax(90)

        self.max_std = max_std or (self.ref_std * 1.5)
        if not np.isfinite(self.max_std) or self.max_std <= 0:
            self.max_std = 1.0
        self.ax.set_rlim(0, self.max_std)

        corr_ticks = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0])
        for c in corr_ticks:
            theta = np.arccos(np.clip(c, -1, 1))
            self.ax.plot([theta, theta], [0, self.max_std], color="0.85", linewidth=0.7, zorder=1)
            self.ax.text(theta, self.max_std * 1.02, f"{c:.2f}", ha="center", va="bottom", fontsize=8, color="0.5")

        radial_ticks = np.linspace(0, self.max_std, 5)[1:]
        angles = np.linspace(0, np.pi / 2, 256)
        for r in radial_ticks:
            self.ax.plot(angles, np.full_like(angles, r), color="0.9", linewidth=0.7, zorder=1)
        self.ax.set_yticks(radial_ticks)
        self.ax.set_yticklabels([f"{tick:.2f}" for tick in radial_ticks])
        self.ax.set_ylabel(f"Standard Deviation ({FLUX_METRIC_UNIT})", labelpad=20)

        self.ax.set_xticks(np.linspace(0, np.pi / 2, 7))
        self.ax.set_xticklabels(["" for _ in range(7)])
        self.ax.set_xlabel("Angular Axis: Correlation Coefficient", labelpad=20)

        self.ax.plot(angles, np.full_like(angles, self.ref_std), "--", color="black", linewidth=1.2, label="Reference Std")

    def add_sample(
        self,
        stddev: float,
        corrcoef: float,
        label: str,
        marker: str = "o",
        color: str = "C0",
        annotate: bool = False,
        **plot_kwargs,
    ) -> None:
        corrcoef = np.clip(corrcoef, -1.0, 1.0)
        theta = np.arccos(corrcoef)
        plot_kwargs.setdefault("markersize", 8)
        plot_kwargs.setdefault("markeredgecolor", "black")
        plot_kwargs.setdefault("markerfacecolor", color)
        self.ax.plot(
            theta,
            stddev,
            marker=marker,
            color=color,
            label=label,
            **plot_kwargs,
        )


def _prepare_taylor_stats(df: pd.DataFrame, flux: str) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    required = {"Correlation", "Predicted Std", "Reference Std"}
    if not required.issubset(df.columns):
        return None

    flux_df = df[df["Flux"] == flux].copy()
    if flux_df.empty:
        return None

    optional_cols = [col for col in ["Site", "Year"] if col in flux_df.columns]
    raw = (
        flux_df[["Method", "Correlation", "Predicted Std", "Reference Std"] + optional_cols]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["Method", "Correlation", "Predicted Std", "Reference Std"])
    )
    if raw.empty:
        return None

    grouped = (
        raw.groupby("Method")[["Correlation", "Predicted Std", "Reference Std"]]
        .median()
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if grouped.empty:
        return None

    return grouped, raw


def _plot_taylor_diagram(stats: pd.DataFrame, raw: pd.DataFrame, flux: str, ax: plt.Axes) -> None:
    corr_coeff = np.sqrt(np.clip(stats["Correlation"], 0.0, 1.0))
    ref_std = stats["Reference Std"].median()
    if not np.isfinite(ref_std) or ref_std <= 0:
        ref_std = stats["Predicted Std"].median()
        if not np.isfinite(ref_std) or ref_std <= 0:
            ref_std = 1.0

    max_std = float(max(stats["Predicted Std"].max(), ref_std)) * 1.2
    diagram = TaylorDiagram(ref_std, ax=ax, max_std=max_std)

    markers = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "*"]

    for idx, method in enumerate(sort_methods(stats.index)):
        color = get_method_color(method, idx)
        marker = markers[idx % len(markers)]
        method_raw = raw[raw["Method"] == method]
        if method_raw.empty:
            continue

        corr_samples = np.sqrt(np.clip(method_raw["Correlation"].values, 0.0, 1.0))
        std_samples = np.clip(method_raw["Predicted Std"].values, 0.0, 4.0)
        theta_samples = np.arccos(np.clip(corr_samples, -1.0, 1.0))

        x_samples = std_samples * corr_samples
        y_samples = std_samples * np.sin(theta_samples)

        mean_xy = np.array([x_samples.mean(), y_samples.mean()])
        mean_std = float(np.linalg.norm(mean_xy))
        mean_corr = 0.0 if mean_std == 0 else float(np.clip(mean_xy[0] / mean_std, -1.0, 1.0))

        if len(x_samples) >= 2:
            cov = np.cov(np.vstack([x_samples, y_samples]), bias=False)
            if np.all(np.isfinite(cov)) and np.linalg.det(cov) >= 0:
                t = np.linspace(0, 2 * np.pi, 256)
                unit_circle = np.stack([np.cos(t), np.sin(t)], axis=0)
                try:
                    ellipse_pts = mean_xy[:, None] + np.linalg.cholesky(cov) @ unit_circle
                except np.linalg.LinAlgError:
                    ellipse_pts = None
                if ellipse_pts is None:
                    vals, vecs = np.linalg.eigh(cov)
                    vals = np.clip(vals, 0, None)
                    widths = np.sqrt(vals)
                    ellipse_pts = mean_xy[:, None] + (vecs @ (widths[:, None] * unit_circle))
                ellipse_pts = ellipse_pts.T
                ellipse_theta = np.arctan2(ellipse_pts[:, 1], ellipse_pts[:, 0])
                ellipse_radius = np.linalg.norm(ellipse_pts, axis=1)
                ax.plot(
                    ellipse_theta,
                    ellipse_radius,
                    color=color,
                    linestyle="--" if method in {"NT", "DT", "TK", "DML"} else "-",
                    linewidth=1.5,
                    alpha=0.8,
                )

        diagram.add_sample(
            mean_std,
            mean_corr,
            label=method_plot_label(method),
            marker=marker,
            color=color,
            annotate=False,
            linestyle="--" if method in PARAMETRIC_METHODS else "-",
        )

    diagram.ax.set_title(f"{flux}", pad=25)
    if raw.empty:
        return
    if stats.index.empty:
        return


def _plot_violin_by_igbp(
    df: pd.DataFrame,
    flux: str,
    metric: str,
    methods: Sequence[str],
    suptitle: str = "",
) -> Optional[plt.Figure]:
    subset = df[(df["Flux"] == flux) & (df["Method"].isin(methods))].copy()
    if subset.empty:
        return None

    methods = sort_methods(methods)
    igbps = sorted(subset["IGBP"].dropna().unique())
    if not igbps:
        return None

    # Apply consistent clipping for all metrics
    subset[metric] = _apply_metric_clipping(metric, subset[metric])

    ncols = min(3, len(igbps))
    nrows = int(math.ceil(len(igbps) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.2 * nrows), squeeze=False)

    method_colors = {method: get_method_color(method, idx) for idx, method in enumerate(methods)}

    for idx, igbp in enumerate(igbps):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        igbp_data = subset[subset["IGBP"] == igbp]

        data_per_method = [
            igbp_data[igbp_data["Method"] == method][metric].dropna()
            for method in methods
        ]
        valid_pairs = [(method, values) for method, values in zip(methods, data_per_method) if len(values) > 0]

        if not valid_pairs:
            ax.set_axis_off()
            continue

        method_labels = [pair[0] for pair in valid_pairs]
        distributions = [pair[1] for pair in valid_pairs]

        parts = ax.violinplot(
            distributions,
            positions=np.arange(1, len(method_labels) + 1),
            showmeans=False,
            showmedians=True,
            showextrema=False,
        )

        for pc, method in zip(parts["bodies"], method_labels):
            color = method_colors.get(method, "C0")
            pc.set_facecolor(color)
            pc.set_edgecolor("black")
            pc.set_alpha(0.6)

        median_color = "black"
        if "cmedians" in parts:
            parts["cmedians"].set_color(median_color)
            parts["cmedians"].set_linewidth(1.4)

        ax.set_xticks(np.arange(1, len(method_labels) + 1))
        ax.set_xticklabels(method_labels, rotation=35, ha="right")
        ax.set_ylabel(metric_with_unit(metric))
        ax.set_title(f"{igbp}", pad=14)
        ax.grid(axis="y", alpha=0.3)
        
        # Set y-axis limits based on metric type to match clipping
        if metric == "RMSE":
            ax.set_ylim(0, 4)
        elif metric in {"MAE", "cRMSE", "Centered RMSE", "Absolute Bias", "Bias"}:
            # For other minimize metrics, use same range as RMSE for consistency
            ax.set_ylim(0, 4)
        elif metric in MAXIMIZE_METRICS:
            # For maximize metrics, ensure we start from 0
            # Let matplotlib auto-scale the upper limit based on clipped data
            ax.set_ylim(bottom=0)

    total_axes = nrows * ncols
    for idx in range(len(igbps), total_axes):
        row, col = divmod(idx, ncols)
        axes[row][col].set_axis_off()

    fig.suptitle(
        f"{metric_with_unit(metric)} distribution by IGBP for {flux}",
        y=0.97,
        fontsize=14,
    )
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    return fig


WORLD_GEOJSON_CACHE: Optional[List[np.ndarray]] = None


def _world_geojson_path() -> Optional[Path]:
    candidate = _geo_data_dir() / "ne_110m_admin_0_countries.geojson"
    return candidate if candidate.exists() else None


def _iter_geojson_rings(geometry: Dict[str, Any]) -> Iterable[np.ndarray]:
    if not geometry:
        return
    kind = geometry.get("type")
    coords = geometry.get("coordinates") or []
    if kind == "Polygon":
        rings = coords[:1]
    elif kind == "MultiPolygon":
        rings = [poly[0] for poly in coords if poly]
    else:
        return
    for ring in rings:
        arr = np.asarray(ring, dtype=float)
        if arr.ndim == 2 and arr.shape[0] >= 2:
            yield arr


def _load_world_rings() -> List[np.ndarray]:
    global WORLD_GEOJSON_CACHE
    if WORLD_GEOJSON_CACHE is not None:
        return WORLD_GEOJSON_CACHE
    path = _world_geojson_path()
    if path is None:
        WORLD_GEOJSON_CACHE = []
        return WORLD_GEOJSON_CACHE
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        WORLD_GEOJSON_CACHE = []
        return WORLD_GEOJSON_CACHE
    rings: List[np.ndarray] = []
    for feature in payload.get("features", []):
        rings.extend(_iter_geojson_rings(feature.get("geometry") or {}))
    WORLD_GEOJSON_CACHE = rings
    return rings


def _draw_world_basemap(ax: plt.Axes) -> bool:
    """Draw land outlines on a regular Matplotlib axes (no Cartopy required)."""
    rings = _load_world_rings()
    if not rings:
        return False
    for ring in rings:
        ax.fill(ring[:, 0], ring[:, 1], facecolor="#e8e4d9", edgecolor="#8a8a8a",
                linewidth=0.35, zorder=0)
    ax.set_facecolor("#d7e6f0")
    return True


def add_world_outline(ax):
    """Coastlines and borders from bundled Natural Earth files; no runtime download."""
    _configure_cartopy_offline()
    try:
        if cfeature is None:
            raise RuntimeError("cartopy unavailable")
        ax.coastlines(linewidth=0.5, zorder=1)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, zorder=1)
    except Exception:
        _draw_world_basemap(ax)

# Continent boundaries: [lon_min, lon_max, lat_min, lat_max]
CONTINENT_BOUNDS = {
    "World": [-180, 180, -90, 90],
    "North America": [-170, -50, 10, 85],
    "South America": [-85, -30, -60, 15],
    "Europe": [-25, 45, 35, 75],
    "Africa": [-20, 55, -35, 40],
    "Asia": [40, 180, -10, 80],
    "Oceania": [110, 180, -50, 10],
    "Australia": [110, 155, -45, -10],
}


def _filter_sites_by_continent(site_stats: pd.DataFrame, continent: str) -> Optional[Tuple[pd.DataFrame, Tuple[float, float, float, float]]]:
    """Filter sites by continent and return filtered dataframe with adjusted bounds.
    
    Returns None if no sites are found in the selected continent.
    """
    if continent == "World" or continent not in CONTINENT_BOUNDS:
        bounds = CONTINENT_BOUNDS["World"]
        return site_stats, tuple(bounds)
    
    bounds = CONTINENT_BOUNDS[continent]
    lon_min, lon_max, lat_min, lat_max = bounds
    
    # Filter sites within continent bounds
    filtered = site_stats[
        (site_stats["Lon"] >= lon_min)
        & (site_stats["Lon"] <= lon_max)
        & (site_stats["Lat"] >= lat_min)
        & (site_stats["Lat"] <= lat_max)
    ].copy()
    
    # If no sites in bounds, return None
    if filtered.empty:
        return None
    
    # Adjust bounds to include a small margin around the actual sites
    lon_range = filtered["Lon"].max() - filtered["Lon"].min()
    lat_range = filtered["Lat"].max() - filtered["Lat"].min()
    margin_lon = max(5.0, lon_range * 0.1)
    margin_lat = max(5.0, lat_range * 0.1)
    
    adjusted_bounds = (
        max(lon_min, filtered["Lon"].min() - margin_lon),
        min(lon_max, filtered["Lon"].max() + margin_lon),
        max(lat_min, filtered["Lat"].min() - margin_lat),
        min(lat_max, filtered["Lat"].max() + margin_lat),
    )
    
    return filtered, adjusted_bounds


def _identify_top_methods(
    matrix: pd.DataFrame,
    metric: str,
    n_top: int = 3,
    tie_decimals: int = 3,
) -> pd.DataFrame:
    """Identify top N methods for each site-year.

    Two methods are considered tied when their rounded values match at
    *tie_decimals* decimal places.  After rounding, ``math.isclose`` uses
    ``abs_tol = 0.5 * 10 ** -tie_decimals`` so that values whose rounded
    representations are identical are always treated as tied.

    Returns a DataFrame of the same shape as *matrix* whose values are the
    assigned rank (1 = best … n_top = n-th best; NaN = outside top N).
    """
    is_minimize = metric in MINIMIZE_METRICS
    is_target_zero = metric in TARGET_ZERO_METRICS
    abs_tol = 0.5 * 10 ** (-tie_decimals)

    rank_matrix = pd.DataFrame(index=matrix.index, columns=matrix.columns, dtype=float)

    for site_year in matrix.index:
        row = matrix.loc[site_year].copy()
        row_rounded = row.round(tie_decimals)
        valid_values = row_rounded.dropna()
        if len(valid_values) < n_top:
            continue

        if is_target_zero:
            sorted_indices = valid_values.abs().sort_values().index
        elif is_minimize:
            sorted_indices = valid_values.sort_values().index
        else:
            sorted_indices = valid_values.sort_values(ascending=False).index

        current_rank = 1
        prev_value = None

        for method in sorted_indices:
            value = row_rounded[method]

            if prev_value is not None:
                if is_target_zero:
                    if not math.isclose(abs(value), abs(prev_value), abs_tol=abs_tol):
                        current_rank += 1
                else:
                    if not math.isclose(value, prev_value, abs_tol=abs_tol):
                        current_rank += 1

            if current_rank <= n_top:
                rank_matrix.loc[site_year, method] = current_rank

            prev_value = value

            if current_rank > n_top:
                break

    return rank_matrix


def _count_top_performances(df: pd.DataFrame, metric: str, period: str = "yearly") -> pd.DataFrame:
    """
    Count how many times each method appears as best, second best, or third best
    across all site-years/events and fluxes.
    """
    fluxes = ['NEE', 'RECO', 'GPP']
    counts = {'Method': [], 'Best': [], 'Second': [], 'Third': []}
    
    for flux in fluxes:
        flux_df = df[df['Flux'] == flux].copy()
        if flux_df.empty:
            continue
        
        flux_df = _prepare_grouping_columns(flux_df, period)
        if flux_df is None:
            continue
        
        # Pivot to get matrix
        try:
            matrix = flux_df.pivot_table(
                index='site_year',
                columns='Method',
                values=metric,
                aggfunc='first'
            )
        except Exception:
            continue
        
        if matrix.empty:
            continue
        
        # Get top methods
        rank_matrix = _identify_top_methods(matrix, metric, n_top=3)
        
        # Count occurrences
        for method in matrix.columns:
            if method not in counts['Method']:
                counts['Method'].append(method)
                counts['Best'].append(0)
                counts['Second'].append(0)
                counts['Third'].append(0)
            
            method_idx = counts['Method'].index(method)
            best_count = (rank_matrix[method] == 1).sum()
            second_count = (rank_matrix[method] == 2).sum()
            third_count = (rank_matrix[method] == 3).sum()
            
            counts['Best'][method_idx] += best_count
            counts['Second'][method_idx] += second_count
            counts['Third'][method_idx] += third_count
    
    # Create DataFrame and sort by total performance
    summary_df = pd.DataFrame(counts)
    if not summary_df.empty:
        summary_df['Total'] = summary_df['Best'] + summary_df['Second'] + summary_df['Third']
        summary_df = summary_df.sort_values(['Best', 'Second', 'Third'], ascending=[False, False, False])
        summary_df = summary_df[['Method', 'Best', 'Second', 'Third', 'Total']]
    
    return summary_df


def _count_top_performances_by_flux(df: pd.DataFrame, metric: str, period: str = "yearly") -> pd.DataFrame:
    """
    Count how many times each method appears as best, second best, or third best, separated by flux.
    Returns a DataFrame with columns: Method, NEE_Best, NEE_Second, NEE_Third, GPP_Best, GPP_Second, GPP_Third, RECO_Best, RECO_Second, RECO_Third
    """
    fluxes = ['NEE', 'RECO', 'GPP']
    all_methods = set()
    
    # Collect all methods first
    for flux in fluxes:
        flux_df = df[df['Flux'] == flux].copy()
        if not flux_df.empty and 'Method' in flux_df.columns:
            all_methods.update(flux_df['Method'].dropna().unique())
    
    if not all_methods:
        return pd.DataFrame()
    
    # Initialize counts dictionary
    counts = {'Method': list(all_methods)}
    for flux in fluxes:
        counts[f'{flux}_Best'] = [0] * len(all_methods)
        counts[f'{flux}_Second'] = [0] * len(all_methods)
        counts[f'{flux}_Third'] = [0] * len(all_methods)
    
    # Count for each flux
    for flux in fluxes:
        flux_df = df[df['Flux'] == flux].copy()
        if flux_df.empty:
            continue
        
        flux_df = _prepare_grouping_columns(flux_df, period)
        if flux_df is None:
            continue
        
        # Pivot to get matrix
        try:
            matrix = flux_df.pivot_table(
                index='site_year',
                columns='Method',
                values=metric,
                aggfunc='first'
            )
        except Exception:
            continue
        
        if matrix.empty:
            continue
        
        # Get top methods
        rank_matrix = _identify_top_methods(matrix, metric, n_top=3)
        
        # Count occurrences for each method
        for method in matrix.columns:
            if method not in all_methods:
                continue
            
            method_idx = counts['Method'].index(method)
            best_count = (rank_matrix[method] == 1).sum()
            second_count = (rank_matrix[method] == 2).sum()
            third_count = (rank_matrix[method] == 3).sum()
            
            counts[f'{flux}_Best'][method_idx] = best_count
            counts[f'{flux}_Second'][method_idx] = second_count
            counts[f'{flux}_Third'][method_idx] = third_count
    
    # Create DataFrame
    summary_df = pd.DataFrame(counts)
    if not summary_df.empty:
        # Calculate totals for sorting
        summary_df['Total_Best'] = summary_df['NEE_Best'] + summary_df['GPP_Best'] + summary_df['RECO_Best']
        summary_df['Total_Second'] = summary_df['NEE_Second'] + summary_df['GPP_Second'] + summary_df['RECO_Second']
        summary_df['Total_Third'] = summary_df['NEE_Third'] + summary_df['GPP_Third'] + summary_df['RECO_Third']
        summary_df['Total'] = summary_df['Total_Best'] + summary_df['Total_Second'] + summary_df['Total_Third']
        
        # Sort by total performance
        summary_df = summary_df.sort_values(['Total_Best', 'Total_Second', 'Total_Third'], ascending=[False, False, False])
        
        # Reorder columns for better display
        column_order = ['Method'] + \
                      [f'{flux}_{rank}' for flux in fluxes for rank in ['Best', 'Second', 'Third']] + \
                      ['Total_Best', 'Total_Second', 'Total_Third', 'Total']
        summary_df = summary_df[[col for col in column_order if col in summary_df.columns]]
    
    return summary_df


def _plot_performance_heatmaps(
    df: pd.DataFrame,
    metric: str,
    timescale: str,
    cmap: str = 'RdYlGn',
    period: str = "yearly",
) -> Optional[Tuple[plt.Figure, pd.DataFrame]]:
    """
    Create three aligned heatmaps showing model performance for NEE, RECO, and GPP,
    with sites ordered by IGBP plant type, plus N_measured ratio.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing all results (already filtered for timescale)
    metric : str
        Performance metric column name
    timescale : str
        Time scale label for title
    cmap : str
        Colormap to use for the heatmap
    
    Returns:
    --------
    fig : matplotlib figure object or None if no data
    """
    # Filter for specific timescale
    df_filtered = df[df['Scale'] == timescale].copy()
    if df_filtered.empty:
        return None
    
    # Check if required columns exist
    if is_event_period(period):
        required_cols = {'Site', 'event_site_ID', 'IGBP', 'Method', 'Flux', metric}
    else:
        required_cols = {'Site', 'Year', 'IGBP', 'Method', 'Flux', metric}
    if not required_cols.issubset(df_filtered.columns):
        return None
    
    df_filtered = _prepare_grouping_columns(df_filtered, period)
    if df_filtered is None:
        return None
    
    # Get unique grouped rows and sort by IGBP
    site_years = (
        df_filtered[['site_year', 'IGBP', 'Site', '_group_sort_id']]
        .drop_duplicates()
        .sort_values(['IGBP', 'Site', '_group_sort_id'])
    )
    
    if site_years.empty:
        return None
    
    # Get all available methods and sort them according to preference
    available_methods = df_filtered['Method'].dropna().unique()
    ordered_methods = sort_methods(available_methods)
    
    if not ordered_methods:
        return None
    
    # Create heatmap matrices for each flux
    fluxes = ['NEE', 'RECO', 'GPP']
    matrices = {}
    for flux in fluxes:
        flux_df = df_filtered[df_filtered['Flux'] == flux]
        if flux_df.empty:
            continue
        matrix = flux_df.pivot_table(
            index='site_year',
            columns='Method',
            values=metric,
            aggfunc='first'  # Use first value if duplicates exist
        )
        # Reorder columns according to preferred order (only include available methods)
        available_in_matrix = [m for m in ordered_methods if m in matrix.columns]
        matrix = matrix[available_in_matrix]
        # Reorder rows according to IGBP ordering
        matrix = matrix.reindex(site_years['site_year'])
        matrices[flux] = matrix
    
    if not matrices:
        return None
    
    # Get N_measured data (take first occurrence for each site-year as it's constant)
    n_measured_df = (df_filtered[['site_year', 'N_measured']]
                    .drop_duplicates()
                    .set_index('site_year')
                    .reindex(site_years['site_year']))
    
    # Calculate figure size
    n_rows = len(site_years)
    n_cols = len(ordered_methods)

    base_cell_height = 0.8
    base_cell_width = 0.6
    
    if n_rows <= 10:
        # Very few rows: use much larger cells
        cell_height = base_cell_height * 3.0
    elif n_rows <= 20:
        # Few rows: use larger cells
        cell_height = base_cell_height * 2.5
    elif n_rows <= 35:
        # GRA range: use the perfect size (this was the reference)
        cell_height = base_cell_height * 1.5
    elif n_rows <= 50:
        # Slightly more than GRA: keep good size
        cell_height = base_cell_height * 1.4
    elif n_rows <= 80:
        # Medium-large: maintain reasonable size
        cell_height = base_cell_height * 1.3
    elif n_rows <= 120:
        # Large: still maintain good visibility
        cell_height = base_cell_height * 1.2
    else:
        # Very many rows: use minimum but still visible size
        # This ensures plot grows with more site-years
        cell_height = base_cell_height * 1.1
    
    # Keep cell width constant for consistency
    cell_width = base_cell_width
    
    # Calculate individual heatmap width and total figure width
    heatmap_width = cell_width * n_cols
    total_width = heatmap_width * 3 + cell_width + 4.0  # Added width for N_measured and spacing
    height = cell_height * n_rows + 1.0  # Add minimal space for margins (no colorbars)
    
    # Ensure minimum sizes for readability
    total_width = max(total_width, 12)
    height = max(height, 6)
    
    # Limit maximum size to prevent PIL decompression bomb errors
    # At 72 DPI: max_pixels = 150M, so max_area = 150M / (72^2) ≈ 28,935 square inches
    # Be very conservative: limit to ~8000 square inches (gives ~41M pixels at 72 DPI)
    max_area = 8000  # square inches
    current_area = total_width * height
    
    if current_area > max_area:
        # Scale down proportionally
        scale = np.sqrt(max_area / current_area)
        total_width = total_width * scale
        height = height * scale
    
    # Also set absolute maximums as safety (at 72 DPI)
    # Allow taller plots for more site-years, but keep width reasonable
    total_width = min(total_width, 40)  # Max 40 inches width
    # Increase max height to accommodate more site-years while preventing decompression bomb
    # At 72 DPI: 50 inches = ~18M pixels, 60 inches = ~26M pixels (both safe)
    max_height = min(60, height * 1.2)  # Allow up to 60 inches or 20% more than calculated
    height = min(height, max_height)
    
    # Create figure with four subplots (no colorbars)
    # Use lower DPI to reduce file size and prevent decompression bomb errors
    fig = plt.figure(figsize=(total_width, height), dpi=72, constrained_layout=False)  # Lower DPI (default is 100)
    
    # Create gridspec with space for both colorbars
    # Calculate width ratios so N_measured has same cell width as flux heatmaps
    # If flux heatmaps have n_cols columns and ratio 1, N_measured (1 column) needs ratio 1/n_cols
    # But we want to keep it simple: use proportional ratios based on number of columns
    n_measured_ratio = 1.0 / n_cols if n_cols > 0 else 0.2
    # Normalize so flux heatmaps are 1.0 each
    width_ratios = [1.0, 1.0, 1.0, n_measured_ratio]
    
    # Create gridspec without colorbars - just the heatmaps
    # Single row gridspec for just the heatmaps (no colorbar row)
    gs = fig.add_gridspec(1, 4, width_ratios=width_ratios, wspace=0.05)
    
    # Create subplot axes directly
    axs = [fig.add_subplot(gs[0, i]) for i in range(4)]
    
    # Create performance heatmaps
    for idx, flux in enumerate(fluxes):
        if flux not in matrices:
            axs[idx].set_axis_off()
            axs[idx].set_title(f'{flux} (no data)', pad=5)
            continue
            
        matrix = matrices[flux]
        # Create a mask for missing values
        mask = matrix.isna()
        
        # Create a custom colormap that includes grey for missing values
        colors = plt.colormaps[cmap](np.linspace(0, 1, 256))
        colors = np.vstack((colors, [0.8, 0.8, 0.8, 1]))  # Add grey color for missing values
        custom_cmap = ListedColormap(colors)
        
        # Adjust annotation font size based on number of rows
        # Increased font sizes for better readability
        annot_fontsize = max(8, min(14, 250 // n_rows)) if n_rows > 0 else 12
        
        # Identify top 3 methods for highlighting
        # Ensure rank_matrix has same index and columns as matrix
        rank_matrix = _identify_top_methods(matrix, metric, n_top=3)
        # Reindex to match matrix exactly
        rank_matrix = rank_matrix.reindex(index=matrix.index, columns=matrix.columns)
        
        # Create the heatmap
        sns.heatmap(
            matrix,
            cmap=custom_cmap,
            center=0.5,
            vmin=0,
            vmax=1,
            annot=True,
            fmt='.2f',
            cbar=False,
            ax=axs[idx],
            square=True,  # Keep square cells for consistency
            mask=mask,
            linewidths=0.2,  # Slightly thicker lines for larger cells
            linecolor='white',
            annot_kws={'size': annot_fontsize}  # Adjust annotation font size
        )
        
        # Add highlighting for top 3 methods
        for i in range(len(matrix.index)):
            for j in range(len(matrix.columns)):
                if mask.iloc[i, j]:  # Skip missing values
                    continue
                
                rank = rank_matrix.iloc[i, j]
                if pd.notna(rank):
                    # Add border/outline for top performers
                    if rank == 1:
                        # Best: thick gold border
                        rect = plt.Rectangle((j, i), 1, 1, fill=False, 
                                            edgecolor='gold', linewidth=3, zorder=10)
                        axs[idx].add_patch(rect)
                    elif rank == 2:
                        # Second: medium silver border
                        rect = plt.Rectangle((j, i), 1, 1, fill=False, 
                                            edgecolor='silver', linewidth=2.5, zorder=10)
                        axs[idx].add_patch(rect)
                    elif rank == 3:
                        # Third: thin bronze border
                        rect = plt.Rectangle((j, i), 1, 1, fill=False, 
                                            edgecolor='#CD7F32', linewidth=2, zorder=10)
                        axs[idx].add_patch(rect)
        
        # Add "NA" text for missing values
        na_fontsize = max(10, min(14, 300 // n_rows)) if n_rows > 0 else 12
        for i in range(len(matrix.index)):
            for j in range(len(matrix.columns)):
                if mask.iloc[i, j]:  # Explicitly check for True
                    axs[idx].text(j + 0.5, i + 0.5, 'NA',
                                ha='center', va='center',
                                color='black', fontsize=na_fontsize)

        # Customize each subplot
        axs[idx].set_title(f'{flux}', pad=5)
        axs[idx].set_xlabel('Model')
        if idx == 0:
            row_ylabel = 'Site-Event (IGBP)' if is_event_period(period) else 'Site-Year (IGBP)'
            axs[idx].set_ylabel(row_ylabel)
        else:
            axs[idx].set_ylabel('')
            axs[idx].set_yticklabels([])
        
        axs[idx].set_xticklabels(axs[idx].get_xticklabels(), rotation=45, ha='right')
    
    # Create N_measured heatmap with missing value handling
    if not n_measured_df.empty:
        mask = n_measured_df.isna()
        colors = plt.colormaps["YlOrBr"](np.linspace(0, 1, 256))
        colors = np.vstack((colors, [0.8, 0.8, 0.8, 1]))  # Add grey color for missing values
        custom_cmap = ListedColormap(colors)
        
        # Adjust annotation font size based on number of rows
        # Increased font sizes for better readability
        annot_fontsize = max(10, min(16, 350 // n_rows)) if n_rows > 0 else 14
        
        sns.heatmap(
            n_measured_df,
            cmap=custom_cmap,
            vmin=0,
            vmax=1,
            annot=True,
            fmt='.2f',
            cbar=False,
            ax=axs[3],
            square=True,
            mask=mask,
            linewidths=0.2,  # Slightly thicker lines for larger cells
            linecolor='white',
            annot_kws={'size': annot_fontsize}  # Adjust annotation font size
        )

        # Add "NA" text for missing values in N_measured
        na_fontsize = max(10, min(14, 300 // n_rows)) if n_rows > 0 else 12
        for i in range(len(n_measured_df.index)):
            try:
                if mask.iloc[i, 0]:  # Explicitly check for True
                    axs[3].text(0.5, i + 0.5, 'NA',
                                ha='center', va='center',
                                color='black', fontsize=na_fontsize)
            except Exception:
                pass

        # Customize N_measured subplot
        axs[3].set_title('Measured\nRatio', pad=5)
        axs[3].set_xlabel('')
        axs[3].set_ylabel('')
        axs[3].set_yticklabels([])
        axs[3].set_xticks([])
    else:
        axs[3].set_axis_off()
        axs[3].set_title('Measured\nRatio (no data)', pad=5)
    
    # Title removed to minimize white space
    # fig.suptitle(f'Model Performance Comparison - {metric.upper()} at {timescale} scale',
    #              y=0.995, fontsize=10)
    
    # Add legend for highlighting (positioned very close to bottom of heatmap)
    from matplotlib.patches import Patch
    # legend_elements = [
    #     Patch(facecolor='none', edgecolor='gold', linewidth=3, label='Best'),
    #     Patch(facecolor='none', edgecolor='silver', linewidth=2.5, label='Second Best'),
    #     Patch(facecolor='none', edgecolor='#CD7F32', linewidth=2, label='Third Best')
    # ]

    # fig.legend(
    #     handles=legend_elements,
    #     loc='upper center',
    #     ncol=3,
    #     bbox_to_anchor=(0.5, 1.005),
    #     frameon=True,
    #     borderpad=0.2
    # )

    #fig.legend(handles=legend_elements, loc='lower center', ncol=3, 
    #          bbox_to_anchor=(0.5, -0.005), fontsize=9, frameon=True)
    
    # Use subplots_adjust for precise control - minimize space at top and bottom
    # Since we removed colorbars and title, we can be even more aggressive with spacing
    plt.subplots_adjust(top=0.99, bottom=0.02, left=0.05, right=0.95)
    
    # Create summary table of top performances
    summary_df = _count_top_performances(df, metric)
    
    return fig, summary_df


def _plot_site_performance_map(
    df,
    flux,
    metric,
    method,
    timescale,
    event,
    continent="World",
    model_label: str = "",
    sites: Optional[Sequence[str]] = None,
    igbps: Optional[Sequence[str]] = None,
):
    required_cols = {"Site", "Lat", "Lon", "Method", "Scale", "Event", "Flux", metric}
    if not required_cols.issubset(df.columns):
        return None

    subset = df[
        (df["Flux"] == flux)
        & (df["Method"] == method)
        & (df["Scale"] == timescale)
        & (df["Event"] == event)
        & df[metric].notna()
    ].copy()
    if subset.empty:
        return None

    site_stats = subset.groupby("Site").agg({metric: "median", "Lat": "first", "Lon": "first"}).reset_index()
    if site_stats.empty:
        return None

    # Filter by continent and get adjusted bounds
    result = _filter_sites_by_continent(site_stats, continent)
    if result is None:
        return None
    
    site_stats, bounds = result
    lon_min, lon_max, lat_min, lat_max = bounds
    
    if site_stats.empty:
        return None

    metric_values = site_stats[metric].to_numpy()

    # Calculate geographic aspect ratio to adjust plot dimensions
    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min
    geo_aspect = lon_range / lat_range if lat_range > 0 else 1.0
    
    # Base figure dimensions
    base_width = 11
    base_height = 6
    
    # Adjust figure size to better match geographic aspect ratio
    # This ensures the map and density plots align properly
    # For wide regions (high geo_aspect), make figure wider
    # For tall regions (low geo_aspect), make figure taller
    if geo_aspect > 1.5:
        # Wide region (e.g., continents spanning many longitudes)
        fig_width = base_width * min(geo_aspect / 1.5, 2.0)  # Cap at 2x
        fig_height = base_height
    elif geo_aspect < 0.67:
        # Tall region (e.g., narrow longitude range)
        fig_width = base_width
        fig_height = base_height * min(0.67 / geo_aspect, 1.5)  # Cap at 1.5x
    else:
        # Balanced region
        fig_width = base_width
        fig_height = base_height
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Adjust width_ratios to ensure density plots align with map
    # The key is that longitude plot should match map width visually
    # and latitude plot should match map height visually
    map_width_ratio = 6.0
    
    # Adjust latitude plot width ratio based on geographic aspect
    # For wide regions, we can make latitude plot narrower relative to map
    # For tall regions, we need latitude plot wider relative to map to maintain alignment
    if geo_aspect > 1.5:
        # Wide region: latitude plot can be narrower
        lat_plot_ratio = 1.5 * (1.5 / geo_aspect)
    elif geo_aspect < 0.67:
        # Tall region: latitude plot should be wider to match visual height
        lat_plot_ratio = 1.5 * (geo_aspect / 0.67)
    else:
        # Balanced: use default
        lat_plot_ratio = 1.5
    
    # Ensure reasonable bounds for lat_plot_ratio
    lat_plot_ratio = max(0.8, min(lat_plot_ratio, 3.0))
    
    gs = gridspec.GridSpec(2, 2, width_ratios=[map_width_ratio, lat_plot_ratio], height_ratios=[1.5, 6], hspace=0.05, wspace=0.05)

    ax_map = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_map)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_map)

    _draw_world_basemap(ax_map)
    sc = ax_map.scatter(
        site_stats["Lon"],
        site_stats["Lat"],
        c=metric_values,
        cmap="viridis",
        s=28,
        edgecolor="black",
        linewidth=0.25,
        zorder=5,
    )
    ax_map.set_xlim(lon_min, lon_max)
    ax_map.set_ylim(lat_min, lat_max)

    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    ax_map.grid(alpha=0.3, linestyle=":")

    # Site name tags clutter the global view; keep them on zoomed regional maps.
    if continent != "World":
        for _, row in site_stats.iterrows():
            ax_map.text(row["Lon"], row["Lat"], f" {row['Site']}", fontsize=4.5, va="center", zorder=6)

    fig.colorbar(sc, ax=ax_map, label=metric_with_unit(metric), fraction=0.046, pad=0.04)

    # Adjust bins to match continent extent
    n_bins = 19
    lon_bins = np.linspace(lon_min, lon_max, n_bins + 1)
    lat_bins = np.linspace(lat_min, lat_max, n_bins + 1)

    site_stats["LonBin"] = pd.cut(site_stats["Lon"], bins=lon_bins, include_lowest=True)
    site_stats["LatBin"] = pd.cut(site_stats["Lat"], bins=lat_bins, include_lowest=True)

    lon_groups = site_stats.groupby("LonBin")[metric]
    lon_median = lon_groups.median()
    lon_p10 = lon_groups.quantile(0.10)
    lon_p90 = lon_groups.quantile(0.90)
    lon_centers = [interval.mid for interval in lon_median.index]

    lat_groups = site_stats.groupby("LatBin")[metric]
    lat_median = lat_groups.median()
    lat_p10 = lat_groups.quantile(0.10)
    lat_p90 = lat_groups.quantile(0.90)
    lat_centers = [interval.mid for interval in lat_median.index]

    ax_top.fill_between(
        lon_centers,
        lon_p10,
        lon_p90,
        color="lightblue",
        alpha=0.4,
    )
    ax_top.plot(lon_centers, lon_median, color="tab:blue", linewidth=1.3)
    ax_top.set_ylabel(metric_with_unit(metric))
    ax_top.set_xticklabels([])
    ax_top.grid(alpha=0.2)
    ax_top.set_xlim(lon_min, lon_max)

    ax_right.fill_betweenx(
        lat_centers,
        lat_p10,
        lat_p90,
        color="lightgreen",
        alpha=0.4,
    )
    ax_right.plot(lat_median, lat_centers, color="tab:green", linewidth=1.3)
    ax_right.set_xlabel(metric_with_unit(metric))
    ax_right.set_yticklabels([])
    ax_right.grid(alpha=0.2)
    ax_right.set_ylim(lat_min, lat_max)

    fig.tight_layout(rect=[0, 0, 1, 0.86])

    # After tight_layout, ensure longitude plot has exactly the same width as map
    map_pos = ax_map.get_position()
    top_pos = ax_top.get_position()
    ax_top.set_position([map_pos.x0, top_pos.y0, map_pos.width, top_pos.height])

    line1 = " — ".join(
        part for part in (model_label, flux, metric_with_unit(metric), timescale) if part
    )
    line2_parts = [method_plot_label(method), event_display_name(event)]
    if sites:
        site_list = list(sites)
        line2_parts.append(
            ", ".join(site_list) if len(site_list) <= 4 else f"{len(site_list)} sites"
        )
    if igbps:
        line2_parts.append(", ".join(igbps))
    title = f"{line1}\n{' · '.join(line2_parts)}\n{continent}"
    fig.suptitle(title, fontsize=11, fontweight="bold", y=0.99)
    return fig

ADDITIONAL_COLOR_CMAP = plt.colormaps["tab20c"].resampled(20)


def _find_truth_file(paths: Dict[str, str], site: str, model: str, data_id: str) -> Optional[Path]:
    candidates: set[Path] = set()
    for value in paths.values():
        if not value:
            continue
        base = Path(value)
        candidates.add(base)
        candidates.add(base / "all")
        candidates.add(base / "processed" / "all")
        candidates.add(base / "data" / "processed" / "all")
        candidates.add(base.parent)
        candidates.add(base.parent / "all")
        candidates.add(base.parent / "processed" / "all")
    if "partitioned" in paths:
        part = Path(paths["partitioned"])
        candidates.add(part.parent / "all")
        candidates.add(part.parent / "processed" / "all")
    if "processed" in paths:
        proc = Path(paths["processed"])
        candidates.add(proc / "all")
        candidates.add(proc)

    # Try pattern with model first: {site}_{model}_{data_id}.csv
    for cand in sorted(candidates):
        if not cand or not cand.exists():
            continue
        file = cand / f"{site}_{model}_{data_id}.csv"
        if file.exists():
            return file
    
    # Fallback to pattern without model: {site}_{data_id}.csv
    for cand in sorted(candidates):
        if not cand or not cand.exists():
            continue
        file = cand / f"{site}_{data_id}.csv"
        if file.exists():
            return file
    return None


def _numeric_year_bounds(year_series: pd.Series) -> Optional[Tuple[int, int]]:
    numeric = pd.to_numeric(year_series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return int(numeric.min()), int(numeric.max())


def _get_site_datetime_bounds(
    paths_config: Dict[str, str],
    site: str,
    model: str,
    data_id: str,
    run_id: str,
    base_path: str,
    methods: Sequence[str],
) -> Tuple[date, date]:
    default_min = date(2000, 1, 1)
    default_max = date(2025, 12, 31)

    truth_path = _find_truth_file(paths_config, site, model, data_id)
    if truth_path and truth_path.exists():
        try:
            dt_df = pd.read_csv(truth_path, usecols=["DateTime"])
            dt_df["DateTime"] = pd.to_datetime(dt_df["DateTime"], errors="coerce")
            dt_df = dt_df.dropna(subset=["DateTime"])
            if not dt_df.empty:
                return dt_df["DateTime"].min().date(), dt_df["DateTime"].max().date()
        except (ValueError, KeyError, pd.errors.EmptyDataError):
            pass

    for method in methods:
        csv_path, _ = _resolve_partitioned_file(base_path, site, model, data_id, run_id, method)
        if csv_path is None:
            continue
        try:
            dt_df = pd.read_csv(csv_path, usecols=["DateTime"])
            dt_df["DateTime"] = pd.to_datetime(dt_df["DateTime"], errors="coerce")
            dt_df = dt_df.dropna(subset=["DateTime"])
            if not dt_df.empty:
                return dt_df["DateTime"].min().date(), dt_df["DateTime"].max().date()
        except (ValueError, KeyError, pd.errors.EmptyDataError):
            continue

    return default_min, default_max


def _partitioned_file_candidates(
    base_path: str,
    site: str,
    model: str,
    data_id: str,
    run_id: str,
    method: str,
) -> List[Path]:
    base = Path(base_path)
    return [
        base / f"{site}_{model}_{data_id}_{run_id}_{method}.csv",
        base / f"{site}_{data_id}_{run_id}_{method}.csv",
        base / f"{site}_{run_id}_{method}.csv",
    ]


def _resolve_partitioned_file(
    base_path: str,
    site: str,
    model: str,
    data_id: str,
    run_id: str,
    method: str,
) -> Tuple[Optional[Path], List[str]]:
    candidates = _partitioned_file_candidates(base_path, site, model, data_id, run_id, method)
    tried = [str(path) for path in candidates]
    for path in candidates:
        if path.exists():
            return path, tried
    return None, tried


def _render_timeseries_diagnostics(diagnostics: Dict[str, Any]) -> None:
    with st.expander("Time series diagnostics", expanded=diagnostics.get("empty_plot", False)):
        st.markdown(
            f"**Window:** `{diagnostics.get('start_date')}` to `{diagnostics.get('end_date')}`  \n"
            f"**Site:** `{diagnostics.get('site')}`  \n"
            f"**Partitioned data directory:** `{diagnostics.get('base_path')}`"
        )

        truth = diagnostics.get("truth", {})
        st.markdown("**Ground truth file**")
        if truth.get("found"):
            date_range = ""
            if truth.get("datetime_min") and truth.get("datetime_max"):
                date_range = f" Available range: `{truth['datetime_min']}` to `{truth['datetime_max']}`."
            st.success(
                f"Found `{truth.get('path')}` with "
                f"{truth.get('rows_in_window', 0)} rows in window "
                f"(of {truth.get('rows_total', 0)} total).{date_range}"
            )
        else:
            st.error(
                "Ground truth file not found. "
                f"Tried: `{truth.get('path')}`"
                if truth.get("path")
                else "No candidate ground-truth path could be resolved."
            )
            if truth.get("candidates"):
                st.caption("Ground-truth search paths: " + ", ".join(f"`{p}`" for p in truth["candidates"]))

        method_rows = diagnostics.get("methods", [])
        if method_rows:
            st.markdown("**Partitioned method files**")
            st.dataframe(pd.DataFrame(method_rows), width="stretch", hide_index=True)

        if diagnostics.get("empty_plot"):
            st.warning(diagnostics.get("empty_reason", "No data was plotted in the selected window."))
        elif diagnostics.get("warnings"):
            for warning in diagnostics["warnings"]:
                st.warning(warning)


def _contiguous_intervals(times: np.ndarray, mask: np.ndarray) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    intervals = []
    start = None
    prev = None
    for t, m in zip(times, mask):
        if m and start is None:
            start = t
        elif not m and start is not None:
            intervals.append((start, prev))
            start = None
        prev = t
    if start is not None and prev is not None:
        intervals.append((start, prev))
    return intervals


def _peak_temperature_mask(df: pd.DataFrame) -> Optional[np.ndarray]:
    """Match evaluation ``peak_temperature`` event: TA above max night-time TA per calendar year."""
    if df.empty or "TA" not in df.columns or "DateTime" not in df.columns:
        return None

    night_mask = df["NIGHT"].astype(bool) if "NIGHT" in df.columns else pd.Series(False, index=df.index)
    peak_mask = pd.Series(False, index=df.index)
    years = df["DateTime"].dt.year
    for year in years.dropna().unique():
        year_mask = years == year
        year_night = night_mask & year_mask
        if not year_night.any():
            continue
        max_temp = df.loc[year_night, "TA"].max()
        if pd.isna(max_temp):
            continue
        peak_mask |= year_mask & (df["TA"] > max_temp)
    return peak_mask.to_numpy()


def _plot_site_timeseries(
    site: str,
    start_date: str,
    end_date: str,
    methods: Sequence[str],
    base_path: str,
    model: str,
    data_id: str,
    run_id: str,
    paths_config: Dict[str, str],
    additional_vars: Sequence[str],
    show_measured: bool = True,
    show_peak_temperature: bool = False,
) -> Tuple[Optional[plt.Figure], Dict[str, Any]]:
    diagnostics: Dict[str, Any] = {
        "site": site,
        "start_date": start_date,
        "end_date": end_date,
        "base_path": base_path,
        "truth": {},
        "methods": [],
        "warnings": [],
        "empty_plot": False,
        "empty_reason": "",
    }

    if not methods:
        diagnostics["empty_plot"] = True
        diagnostics["empty_reason"] = "No methods were selected."
        return None, diagnostics

    truth_path = _find_truth_file(paths_config, site, model, data_id)
    truth_df = pd.DataFrame()
    truth_rows_total = 0
    diagnostics["truth"] = {
        "path": str(truth_path) if truth_path else None,
        "found": bool(truth_path and truth_path.exists()),
        "rows_total": 0,
        "rows_in_window": 0,
    }
    if truth_path and truth_path.exists():
        try:
            truth_df = pd.read_csv(truth_path)
            if "DateTime" in truth_df.columns:
                truth_df["DateTime"] = pd.to_datetime(truth_df["DateTime"], errors="coerce")
                truth_df = truth_df.dropna(subset=["DateTime"])
                truth_rows_total = len(truth_df)
                diagnostics["truth"]["datetime_min"] = str(truth_df["DateTime"].min())
                diagnostics["truth"]["datetime_max"] = str(truth_df["DateTime"].max())
                mask = (truth_df["DateTime"] >= start_date) & (truth_df["DateTime"] <= end_date)
                truth_df = truth_df.loc[mask]
                diagnostics["truth"]["rows_total"] = truth_rows_total
                diagnostics["truth"]["rows_in_window"] = len(truth_df)
            else:
                truth_df = pd.DataFrame()
                diagnostics["warnings"].append(
                    f"Ground truth file `{truth_path}` does not contain a DateTime column."
                )
        except Exception as exc:
            truth_df = pd.DataFrame()
            diagnostics["warnings"].append(f"Failed to read ground truth file `{truth_path}`: {exc}")
    elif truth_path:
        diagnostics["warnings"].append(f"Ground truth file not found at `{truth_path}`.")
    else:
        diagnostics["warnings"].append(
            f"No ground truth file found for site `{site}` with model `{model}` and data id `{data_id}`."
        )

    measured_intervals: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    if show_measured and not truth_df.empty and "NEE_QC" in truth_df.columns:
        measured_mask = truth_df["NEE_QC"].eq(0).values
        measured_intervals = _contiguous_intervals(truth_df["DateTime"].values, measured_mask)

    peak_temperature_intervals: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    if show_peak_temperature and not truth_df.empty:
        peak_mask = _peak_temperature_mask(truth_df)
        if peak_mask is None:
            diagnostics["warnings"].append(
                "Peak temperature shading requires TA and DateTime columns in the ground truth file."
            )
        elif peak_mask.any():
            peak_temperature_intervals = _contiguous_intervals(truth_df["DateTime"].values, peak_mask)
            diagnostics["peak_temperature_intervals"] = len(peak_temperature_intervals)
        else:
            diagnostics["peak_temperature_intervals"] = 0
            diagnostics["warnings"].append(
                "No peak temperature periods found in the selected window "
                "(TA above max night-time TA per calendar year)."
            )

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fluxes = ["NEE", "GPP", "RECO"]
    plotted_methods: set[str] = set()

    for ax, flux in zip(axes, fluxes):
        ax.set_ylabel(f"{flux} (µmol m⁻² s⁻¹)")
        ax.grid(alpha=0.3, linestyle=":")

        if measured_intervals:
            for start, end in measured_intervals:
                ax.axvspan(start, end, color="lightgrey", alpha=0.8, zorder=0)

        if peak_temperature_intervals:
            for start, end in peak_temperature_intervals:
                ax.axvspan(start, end, color="red", alpha=0.25, zorder=1)

        if not truth_df.empty and flux in truth_df.columns:
            ax.plot(
                truth_df["DateTime"],
                truth_df[flux],
                color="black",
                linewidth=0.8,
                label="Ground Truth",
                alpha=1.0,
                zorder=3,
            )
            if flux == "NEE" and "NEE_obs" in truth_df.columns:
                ax.scatter(
                    truth_df["DateTime"],
                    truth_df["NEE_obs"],
                    color="black",
                    s=3,
                    alpha=0.3,
                    label="Observations",
                    zorder=4,
                )

    twin_ax = None

    for idx, method in enumerate(sort_methods(methods)):
        style = _get_method_style(method, idx)
        method_info: Dict[str, Any] = {
            "method": method,
            "found": False,
            "path": None,
            "candidate_paths": [],
            "rows_total": 0,
            "rows_in_window": 0,
            "plotted_fluxes": [],
            "status": "",
        }
        csv_path, candidate_paths = _resolve_partitioned_file(
            base_path, site, model, data_id, run_id, method
        )
        method_info["candidate_paths"] = candidate_paths
        method_info["path"] = str(csv_path) if csv_path else candidate_paths[0]

        if csv_path is None:
            method_info["status"] = "file not found"
            diagnostics["methods"].append(method_info)
            continue

        method_info["found"] = True
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            method_info["status"] = f"read error: {exc}"
            diagnostics["methods"].append(method_info)
            continue

        if "DateTime" not in df.columns:
            method_info["status"] = "missing DateTime column"
            diagnostics["methods"].append(method_info)
            continue

        df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
        df = df.dropna(subset=["DateTime"])
        method_info["rows_total"] = len(df)
        if not df.empty:
            method_info["datetime_min"] = str(df["DateTime"].min())
            method_info["datetime_max"] = str(df["DateTime"].max())
        mask = (df["DateTime"] >= start_date) & (df["DateTime"] <= end_date)
        df = df.loc[mask]
        method_info["rows_in_window"] = len(df)
        if df.empty:
            method_info["status"] = "no rows in selected window"
            diagnostics["methods"].append(method_info)
            continue

        plotted_any = False
        for ax, flux in zip(axes, fluxes):
            if flux not in df.columns:
                continue
            ax.plot(df["DateTime"], df[flux], label=method_plot_label(method), linewidth=0.8, **style, zorder=2)
            method_info["plotted_fluxes"].append(flux)
            plotted_any = True

        if plotted_any:
            method_info["status"] = "plotted"
            plotted_methods.add(method)
        else:
            method_info["status"] = "rows found, but NEE/GPP/RECO columns missing"
        diagnostics["methods"].append(method_info)

    available_vars = []
    if additional_vars and not truth_df.empty:
        available_vars = [var for var in additional_vars if var in truth_df.columns]
    if available_vars:
        twin_ax = axes[0].twinx()
        twin_ax.set_ylabel("Normalized Additional Variables", color="gray")
        twin_ax.tick_params(axis="y", labelcolor="gray")
        for idx, var in enumerate(available_vars):
            series = truth_df[var].astype(float)
            if series.empty:
                continue
            var_min = series.min()
            var_max = series.max()
            if not np.isfinite(var_min) or not np.isfinite(var_max) or var_min == var_max:
                continue
            normalized = (series - var_min) / (var_max - var_min)
            color = ADDITIONAL_COLOR_CMAP(idx % ADDITIONAL_COLOR_CMAP.N)
            twin_ax.plot(
                truth_df["DateTime"],
                normalized,
                linestyle=":",
                linewidth=0.5,
                color=color,
                label=f"{var} ({var_min:.1f}-{var_max:.1f})",
            )
        twin_ax.set_ylim(0, 1)

    axes[-1].set_xlabel("Date")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()

    # Collect legend handles from all axes to ensure ground truth is included
    handles, labels = [], []
    seen_labels = set()
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in seen_labels:
                handles.append(handle)
                labels.append(label)
                seen_labels.add(label)
    
    if twin_ax is not None:
        h2, l2 = twin_ax.get_legend_handles_labels()
        for handle, label in zip(h2, l2):
            if label not in seen_labels:
                handles.append(handle)
                labels.append(label)
                seen_labels.add(label)
    
    if handles or measured_intervals or peak_temperature_intervals:
        overlay_handles: List[Any] = []
        if measured_intervals:
            overlay_handles.append(
                mpatches.Patch(facecolor="lightgrey", edgecolor="none", alpha=0.8, label="Measured periods")
            )
        if peak_temperature_intervals:
            overlay_handles.append(
                mpatches.Patch(facecolor="red", edgecolor="none", alpha=0.25, label="Peak temperature periods")
            )
        handles = overlay_handles + handles
        labels = [h.get_label() for h in handles]
        fig.legend(handles, labels, loc="upper right")

    fig.suptitle(f"{site}: {start_date} to {end_date}")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    has_truth = not truth_df.empty
    if not plotted_methods and not has_truth:
        diagnostics["empty_plot"] = True
        missing_files = [m for m in diagnostics["methods"] if not m["found"]]
        empty_window = [m for m in diagnostics["methods"] if m["found"] and m["rows_in_window"] == 0]
        if missing_files:
            diagnostics["empty_reason"] = (
                "No partitioned method files were found for the selected methods. "
                "See candidate paths in the diagnostics table."
            )
        elif empty_window:
            diagnostics["empty_reason"] = (
                "Partitioned files exist, but none contain rows in the selected evaluation window."
            )
        else:
            diagnostics["empty_reason"] = "No plottable flux columns were found in the selected window."

    return fig, diagnostics


def _highlight_best(
    display_df: pd.DataFrame,
    median_df: pd.DataFrame,
) -> "pd.io.formats.style.Styler":
    def rank_colors(metric: str, medians: pd.Series) -> Dict[str, str]:
        """
        Rank methods and assign gold, silver, bronze colors to top 3.
        Uses rounding to 3 decimal places for consistency with heatmap highlighting.
        """
        colors: Dict[str, str] = {}
        series = medians.dropna()
        if series.empty:
            return colors

        # Round to 3 decimal places for comparison (matching heatmap behavior)
        series_rounded = series.round(3)

        if metric in TARGET_ZERO_METRICS:
            # For target zero metrics, rank by absolute value (closer to zero is better)
            abs_values = series_rounded.abs()
            order_index = abs_values.sort_values().index
            ordered_series = series_rounded.loc[order_index]
            compare_series = abs_values.loc[order_index]
        elif metric in MINIMIZE_METRICS:
            # For minimize metrics, lower is better
            ordered_series = series_rounded.sort_values()
            compare_series = ordered_series
        else:
            # For maximize metrics, higher is better
            ordered_series = series_rounded.sort_values(ascending=False)
            compare_series = ordered_series

        color_idx = 0
        prev_cmp_value: Optional[float] = None

        for method, cmp_value in compare_series.items():
            if color_idx >= len(RANK_COLORS):
                break

            if prev_cmp_value is None:
                # First method gets gold (best)
                colors[method] = RANK_COLORS[color_idx]
                prev_cmp_value = cmp_value
                continue

            # Use abs_tol=0.001 to match heatmap behavior (3 decimal places)
            if math.isfinite(cmp_value) and math.isfinite(prev_cmp_value) and math.isclose(
                cmp_value, prev_cmp_value, abs_tol=0.001
            ):
                # Same value after rounding - same rank (tie)
                colors[method] = RANK_COLORS[color_idx]
            else:
                # Different value - move to next rank
                color_idx += 1
                if color_idx >= len(RANK_COLORS):
                    break
                colors[method] = RANK_COLORS[color_idx]
                prev_cmp_value = cmp_value

        return colors

    def highlight_column(col: pd.Series) -> List[str]:
        metric_with_arrow = col.name
        # Strip arrow to get original metric name for comparison
        metric = _strip_metric_arrow(metric_with_arrow)
        if metric_with_arrow not in median_df.columns:
            return [""] * len(col)
        medians = median_df[metric_with_arrow]
        if medians.dropna().empty:
            return [""] * len(col)

        colors = rank_colors(metric, medians)
        styles = [""] * len(col)
        for idx, method in enumerate(median_df.index):
            color = colors.get(method)
            if color:
                styles[idx] = f"background-color: {color}; font-weight: 600;"
        return styles

    styler = display_df.style.apply(highlight_column, axis=0)
    return styler


def get_metric_formula(metric: str) -> str:
    """Get the mathematical formula for a given metric."""
    formulas = {
        "RMSE": r"RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_{true,i} - y_{pred,i})^2}",
        "cRMSE": r"cRMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}((y_{true,i} - \bar{y}_{true}) - (y_{pred,i} - \bar{y}_{pred}))^2}",
        "Centered RMSE": r"Centered\ RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}((y_{true,i} - \bar{y}_{true}) - (y_{pred,i} - \bar{y}_{pred}))^2}",
        "MAE": r"MAE = \frac{1}{n}\sum_{i=1}^{n}|y_{true,i} - y_{pred,i}|",
        "Bias": r"Bias = \bar{y}_{pred} - \bar{y}_{true} = \frac{1}{n}\sum_{i=1}^{n}(y_{pred,i} - y_{true,i})",
        "Correlation": r"r^2 = \left(\frac{\sum_{i=1}^{n}(y_{true,i} - \bar{y}_{true})(y_{pred,i} - \bar{y}_{pred})}{\sigma_{true} \times \sigma_{pred}}\right)^2",
        "NSE": r"NSE = 1 - \frac{\sum_{i=1}^{n}(y_{true,i} - y_{pred,i})^2}{\sum_{i=1}^{n}(y_{true,i} - \bar{y}_{true})^2}",
        "KGE": r"KGE = 1 - \sqrt{(r - 1)^2 + (\alpha - 1)^2 + (\beta - 1)^2},\ where\ \alpha = \frac{\sigma_{pred}}{\sigma_{true}},\ \beta = \frac{\bar{y}_{pred}}{\bar{y}_{true}}",
        "robust_NSE": r"Robust\ NSE = Similar\ to\ NSE\ but\ using\ robust\ statistics\ (median\ instead\ of\ mean)",
        "Predicted Std": r"\sigma_{pred} = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(y_{pred,i} - \bar{y}_{pred})^2}",
        "Reference Std": r"\sigma_{true} = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(y_{true,i} - \bar{y}_{true})^2}",
    }
    return formulas.get(metric, f"Formula not available for {metric}")


def get_metric_description(metric: str) -> str:
    """Get a description of what the metric measures and what values are better/worse."""
    descriptions = {
        "RMSE": (
            "**Root Mean Squared Error (RMSE)**: Measures the average magnitude of prediction errors. "
            f"**Lower is better** (0 is perfect). Sensitive to outliers. Units: {FLUX_METRIC_UNIT}."
        ),
        "cRMSE": (
            "**Centered RMSE**: Measures prediction error after removing mean bias. "
            f"**Lower is better** (0 is perfect). Focuses on pattern accuracy rather than absolute values. "
            f"Units: {FLUX_METRIC_UNIT}."
        ),
        "Centered RMSE": (
            "**Centered RMSE**: Measures prediction error after removing mean bias. "
            f"**Lower is better** (0 is perfect). Focuses on pattern accuracy rather than absolute values. "
            f"Units: {FLUX_METRIC_UNIT}."
        ),
        "MAE": (
            "**Mean Absolute Error (MAE)**: Average absolute difference between predictions and observations. "
            f"**Lower is better** (0 is perfect). Less sensitive to outliers than RMSE. Units: {FLUX_METRIC_UNIT}."
        ),
        "Bias": (
            "**Bias**: Systematic over- or under-prediction. Positive values indicate over-prediction, "
            f"negative values indicate under-prediction. **Closer to zero is better** (0 is perfect). "
            f"Units: {FLUX_METRIC_UNIT}."
        ),
        "Correlation": (
            "**Pearson r²**: Squared Pearson correlation coefficient. Measures linear relationship strength. "
            "**Higher is better** (1 is perfect, 0 indicates no linear relationship). Range: 0 to 1."
        ),
        "NSE": (
            "**Nash-Sutcliffe Efficiency (NSE)**: Measures model efficiency relative to mean baseline. "
            "**Higher is better** (1 is perfect, 0 equals the mean predictor, negative values are worse than the mean). "
            "Range: −∞ to 1. Algebraically identical to the coefficient of determination (R²). "
            "Plots clip the *view* to [−0.5, 1] and shade the region below 0; "
            "medians, boxes, and tables still use the unclipped values."
        ),
        "KGE": (
            "**Kling-Gupta Efficiency (KGE)**: Composite metric combining correlation, variability, and bias. "
            "**Higher is better** (1 is perfect). Range: -∞ to 1."
        ),
        "robust_NSE": (
            "**Robust NSE**: Nash-Sutcliffe Efficiency using robust statistics (less sensitive to outliers). "
            "**Higher is better** (1 is perfect, 0 equals the mean predictor, negative values are worse than the mean). "
            "Plots clip the *view* to [−0.5, 1] and shade the region below 0; "
            "medians, boxes, and tables still use the unclipped values."
        ),
        "Predicted Std": (
            f"**Predicted Standard Deviation**: Variability in predicted values ({FLUX_METRIC_UNIT}). "
            "Useful for comparing with Reference Std to assess variance matching."
        ),
        "Reference Std": (
            f"**Reference Standard Deviation**: Variability in observed values ({FLUX_METRIC_UNIT}). "
            "Used as reference for comparing predicted variability."
        ),
    }
    return descriptions.get(metric, f"Description not available for {metric}")


def show_metric_info(metrics: List[str]) -> None:
    """Display expandable metric information with formulas and descriptions."""
    with st.expander("📊 Metric Information & Formulas", expanded=False):
        st.markdown("### Metric Definitions")
        for metric in order_metrics(metrics):
            st.markdown(f"#### {metric_with_unit(metric)}")
            st.markdown(get_metric_description(metric))
            st.latex(get_metric_formula(metric))
            st.markdown("---")


def show_method_info(methods: List[str]) -> None:
    """Display expandable method information from METHOD_INFO."""
    # Filter to only show methods that have information available
    available_methods = [m for m in methods if m in METHOD_INFO]
    if not available_methods:
        return
    
    with st.expander("🔬 Method Information", expanded=False):
        st.markdown("### Method Descriptions")
        # Sort methods according to priority order
        sorted_methods = sort_methods(available_methods)
        for method in sorted_methods:
            st.markdown(f"#### {method}")
            st.markdown(METHOD_INFO[method])
            st.markdown("---")


# Dashboard scale name → (formula, description). Order follows DEFAULT_TIMESCALES_ORDER.
SCALE_INFO: Dict[str, Tuple[str, str]] = {
    "Half-Hourly": (
        r"x_{d,h}",
        "**Half-Hourly**: Raw half-hourly flux values. "
        "No temporal aggregation is applied. "
        r"$h \in \{1,\dots,48\}$ is the half-hour slot within calendar date $d$.",
    ),
    "Daily": (
        r"\bar{x}_d = \frac{1}{48}\sum_{h=1}^{48} x_{d,h}",
        "**Daily**: Arithmetic mean over all half-hourly timesteps within each calendar date. "
        "Each day is represented by a single daily mean.",
    ),
    "Monthly Diurnal Cycle": (
        r"\mu_{m,h} = \underset{d:\,m(d)=m}{\mathrm{mean}}\bigl(x_{d,h}\bigr)",
        "**Monthly Diurnal Cycle** (mean diurnal cycle): "
        "Mean flux at each half-hour slot $h$ within each calendar month $m$, "
        "averaged across all days of that month. "
        "Captures the typical within-day pattern for that month.",
    ),
    "Seasonal Cycle": (
        r"S_j = \underset{d:\,j(d)=j}{\mathrm{mean}}(\bar{x}_d)",
        "**Seasonal Cycle**: Mean of daily means across all years that share the same "
        "day-of-year $j$. Captures the climatological seasonal pattern without additional smoothing.",
    ),
    "Daily Anomalies": (
        r"a_d = \bar{x}_d - S_{j(d)}",
        "**Daily Anomalies**: Deviation of each daily mean from the multi-year "
        "day-of-year climatology $S_{j(d)}$. "
        "Requires more than 3 years of data per site.",
    ),
    "Inter-Annual Variability": (
        r"v_y = \bar{x}_y - \frac{1}{Y}\sum_{y'=1}^{Y}\bar{x}_{y'},"
        r"\quad \bar{x}_y = \underset{d:\,y(d)=y}{\mathrm{mean}}(\bar{x}_d)",
        "**Inter-Annual Variability**: Deviation of each annual mean from the grand mean "
        "across all $Y$ years. Requires $Y > 5$ years of continuous data per site. "
        "Only available for the across-years period.",
    ),
}


def get_scale_formula(scale: str) -> str:
    entry = SCALE_INFO.get(scale)
    return entry[0] if entry else rf"\text{{Formula not available for {scale}}}"


def get_scale_description(scale: str) -> str:
    entry = SCALE_INFO.get(scale)
    return entry[1] if entry else f"Description not available for {scale}."


def show_scale_info() -> None:
    """Display expandable temporal-scale information with formulas and descriptions."""
    with st.expander("⏱️ Temporal Scale Information & Formulas", expanded=False):
        st.markdown("### Temporal Aggregation Scales")
        st.markdown(
            r"Let $x_{d,h}$ denote the half-hourly flux value on calendar date $d$ "
            r"at half-hour slot $h \in \{1,\dots,48\}$, with $j(d)$ the day-of-year, "
            r"$m(d)$ the calendar month, and $y(d)$ the calendar year. "
            r"The daily mean is $\bar{x}_d = \frac{1}{48}\sum_h x_{d,h}$."
        )
        st.markdown("---")
        for scale in DEFAULT_TIMESCALES_ORDER:
            st.markdown(f"#### {scale}")
            st.markdown(get_scale_description(scale))
            st.latex(get_scale_formula(scale))
            st.markdown("---")


# Special evaluation subsets (dashboard names). Harvest is omitted.
EVENT_INFO_ORDER = (
    "Day",
    "Night",
    "Measured",
    "Unmeasured",
    "Peak Temperature",
    "Rain Pulse",
)
EVENT_INFO: Dict[str, str] = {
    "Day": (
        "**Day**: Half-hourly time steps classified as daytime by the FLUXNET **NIGHT** flag. "
        "NIGHT is a half-hourly binary variable that marks daytime versus nighttime from "
        "potential incoming solar radiation ($\\mathrm{SW\\_IN\\_POT}$). "
        "The Day subset keeps steps where $\\mathrm{NIGHT} = 0$ (daytime)."
    ),
    "Night": (
        "**Night**: Half-hourly time steps classified as nighttime by the FLUXNET **NIGHT** flag "
        "($\\mathrm{NIGHT} = 1$). "
        "The same $\\mathrm{SW\\_IN\\_POT}$-based flag is used as for the Day subset, so Day and "
        "Night form a complementary split of the half-hourly record."
    ),
    "Measured": (
        "**Measured**: Half-hourly time steps where NEE was directly observed "
        r"($\mathrm{NEE\_QC} = 0$). "
        "This is the complement of Unmeasured and isolates performance where a direct "
        "observational constraint is available."
    ),
    "Unmeasured": (
        "**Unmeasured** (gap-filled): Half-hourly time steps where NEE was not directly measured "
        r"($\mathrm{NEE\_QC} \neq 0$). "
        "Assesses method performance under conditions where no direct observational "
        "constraint is available."
    ),
    "Peak Temperature": (
        "**Peak Temperature**: Half-hourly time steps where air temperature exceeds the maximum "
        "nighttime temperature recorded in the respective site-year. "
        "Targets periods of anomalous heat stress that are not represented in the nighttime "
        "temperature envelope used by traditional respiration models."
    ),
    "Rain Pulse": (
        "**Rain Pulse**: A ±2-day window centred on each identified rain pulse event, "
        "evaluated at half-hourly resolution. Multiple events within a site-year are "
        "treated independently. Targets the rapid ecosystem response to precipitation "
        "inputs, particularly the pulse respiration response in water-limited ecosystems."
    ),
}


def show_event_info() -> None:
    """Display expandable information on special evaluation subsets."""
    with st.expander("⚡ Event Subset Information", expanded=False):
        st.markdown("### Special Evaluation Subsets")
        st.markdown(
            "These subsets restrict the half-hourly record before metrics are computed. "
            "Day and Night split the record with the FLUXNET NIGHT flag; Measured and "
            "Unmeasured split it by NEE quality flag. All are evaluated at half-hourly resolution."
        )
        st.markdown("---")
        for name in EVENT_INFO_ORDER:
            st.markdown(f"#### {name}")
            st.markdown(EVENT_INFO[name])
            st.markdown("---")


def show_plot_info(plot_type: str, period: str = "yearly", *, header: bool = True) -> None:
    """Display the section heading and an expandable description of the plot."""
    unit = period_unit_label(period)
    unit_singular = "event" if is_event_period(period) else "site-year"
    plot_info = {
        "site_map": {
            "title": "Per-Site Performance Map",
            "description": (
                "**What it shows**: The geographic distribution of a method's skill — the median "
                "metric value at each benchmark site.\n\n"
                "**How it's rendered**: Each of the 188 sites is coloured by its median metric "
                "value for the selected method, flux and scale. Site coverage inherits the bias of "
                "the underlying FLUXNET network — dense in North America and Europe, sparse in "
                "Africa, South America and continental Asia.\n\n"
                "**What it can tell you**: Whether errors are spatially structured — regional "
                "clusters of weak performance or latitudinal gradients — and where the network is "
                "thin enough that per-site reads should be cautious."
            ),
        },
        "cross_scale": {
            "title": "Cross-Scale Performance",
            "description": (
                f"**What it shows**: One metric tracked across all available timescales for the selected methods. "
                f"Each point/box/violin summarizes performance over all {unit}.\n\n"
                "**Display modes**:\n"
                "- **Percentile bands**: Median connected across scales, with shaded symmetric quantile bands "
                "and dashed upper/lower quantile lines\n"
                "- **Boxplot / Violin**: Full distribution at each scale, shown side-by-side per method\n\n"
                "**Interpretation**: Reveals whether a method is consistently good across temporal aggregations "
                "or only at specific scales."
            ),
        },
        "correlation_overview": {
            "title": "Performance Across Conditions",
            "description": (
                "**What it shows**: How well each method's estimated fluxes track the ground "
                "truth across every evaluation condition at once — the most compact view of the "
                "benchmark.\n\n"
                "**How it's rendered**: Each point is the median metric value "
        f"across {unit}, with an optional shaded quantile band.\n\n"
                "**What it can tell you**: Reading left to right within a panel shows whether a "
                "method's skill survives the move from raw temporal scales to derived scales and "
                "stress regimes, or breaks down along the way. Comparing the two columns shows "
                "whether a difficulty is robust (both simulators drop together) or simulator-"
                "specific (they diverge)."
            ),
        },
        "method_ranking": {
            "title": "Aggregate Method Ranking",
            "description": (
                "**What it shows**: An overall comparison of methods on one metric, aggregated "
                f"over all {unit} that pass the current filters.\n\n"
                "**How it's rendered**: *Ranking* reports the share of "
                f"{unit} in which a method ranks first on the selected metric (ties, resolved at "
                "the chosen decimal precision, share a rank); *Values* reports the median with its "
                "25th/75th-percentile spread.\n\n"
                "**What it can tell you**: Whether any single method dominates, or whether the "
                "ranking flips across fluxes and across the two simulators. A method that leads on "
                "cRMSE (dynamics) but not on Bias (structural over-/under-estimation) captures "
                "shape without capturing level. Note that absolute magnitudes differ between "
                "simulators and real towers, so read RMSE, cRMSE and Bias within a simulator, not "
                "across them."
            ),
        },
        "method_distributions": {
            "title": "Performance Distributions",
            "description": (
                f"**What it shows**: The full spread of a metric across {unit}, not just its "
                "median — grouped by temporal scale, stress regime, or IGBP class.\n\n"
                "**How it's rendered**: Each box is the per-"
                f"{unit_singular} distribution for one method. Unbounded metrics use a robust "
                "axis; any site-years pushed past the clip are reported as a cut-out count beneath "
                "the plot rather than silently dropped.\n\n"
                "**What it can tell you**: How *consistent* a method is, not just how good on "
                "average. A high median with a long lower tail means the method is unreliable on a "
                "meaningful fraction of site-years; a tight box means stable behaviour."
            ),
        },
        "coverage_bins": {
            "title": "Performance vs. Data Coverage",
            "description": (
                "**What it shows**: How partitioning quality depends on how much *measured* "
                "(non-gap-filled) data a site-year contains, split by land-cover class.\n\n"
                "**How it's rendered**: Coverage is the percentage of directly measured "
                "half-hourly NEE steps in a site-year (NEE_QC = 0), binned up to a chosen threshold. Lines are "
                "the median metric per bin with IQR error bars; parametric methods are dashed. "
                "Bins with few site-years ($n<10$) should be read with caution.\n\n"
                "**What it can tell you**: Whether a method's skill depends on how much measured "
                "data a site-year contains. A line that stays flat across bins marks a method that "
                "is robust to data scarcity; a line that falls toward the low-coverage bins marks "
                "a data-hungry one. Comparing the dashed parametric methods against the ML methods "
                "shows which paradigm is more exposed, and comparing GPP against RECO or one biome "
                "against another shows where scarcity is more harmful."
            ),
        },
    }
    
    info = plot_info.get(plot_type)
    if info:
        if header:
            st.subheader(info["title"])
        with st.expander("Description", expanded=False):
            st.markdown(info["description"])


def _add_metric_arrows(metric: str) -> str:
    """Add up/down arrow to metric name to indicate if higher or lower is better."""
    labeled = metric_with_unit(metric)
    if metric in MINIMIZE_METRICS:
        return f"{labeled} ↓"
    elif metric in TARGET_ZERO_METRICS:
        return f"{labeled} →0"
    else:
        # Assume maximize for other metrics (Correlation, NSE, etc.)
        return f"{labeled} ↑"


def _strip_metric_arrow(metric_with_arrow: str) -> str:
    """Remove arrow and unit suffix to get original metric name."""
    metric = metric_with_arrow.replace(" ↓", "").replace(" ↑", "").replace(" →0", "")
    for unit_suffix in (f" ({FLUX_METRIC_UNIT})", f" ({DIMENSIONLESS_UNIT})"):
        if metric.endswith(unit_suffix):
            metric = metric[: -len(unit_suffix)]
            break
    reverse = {display: raw for raw, display in METRIC_DISPLAY_NAMES.items()}
    return reverse.get(metric, metric)


def _plot_nse_appendix_grid(
    dfs: Dict[str, pd.DataFrame],
    extra_scale_dfs: Dict[str, Dict[str, pd.DataFrame]],
    scales: List[str],
    methods: Sequence[str],
    flux: str,
    figsize: Optional[Tuple[float, float]] = None,
    fontsize: int = 11,
    suptitle: str = "",
) -> Optional[plt.Figure]:
    """Appendix NSE box-plot grid: rows = timescales or events, cols = models.

    Each subplot shows one box per method.  NSE is shown on [−0.5, 1]
    with a shaded below-zero band; box statistics use the unclipped values.
    """
    methods = sort_methods(methods)
    n_rows = len(scales)
    if n_rows == 0 or not methods:
        return None

    metric = "NSE"
    FSIZE_TICK = fontsize
    FSIZE_LABEL = fontsize + 2
    FSIZE_LEGEND = fontsize - 1
    positions = list(range(len(methods)))

    w = max(10.0, len(methods) * 0.85)
    h = n_rows * 2.3
    fig, axes = plt.subplots(
        nrows=n_rows, ncols=2,
        figsize=figsize or (w, h),
        sharex=True,
        sharey="row",
    )
    fig.subplots_adjust(
        left=0.12, right=0.98, top=0.93, bottom=0.08,
        hspace=0.15, wspace=0.06,
    )

    if n_rows == 1:
        axes = np.array(axes).reshape(1, 2)

    # ── Main plotting loop ──────────────────────────────────────────────────
    for row_idx, scale in enumerate(scales):
        for col_idx, model in enumerate(MODELS):
            ax = axes[row_idx][col_idx]
            df_m = dfs.get(model, pd.DataFrame())
            extra_m = extra_scale_dfs.get(model, {})

            if scale in extra_m:
                df_scale = extra_m[scale]
            elif not df_m.empty:
                df_scale = df_m[df_m["Scale"] == scale]
            else:
                df_scale = pd.DataFrame()

            for m_idx, method in enumerate(methods):
                if df_scale.empty:
                    continue
                rows = df_scale[
                    (df_scale["Method"] == method) & (df_scale["Flux"] == flux)
                ]
                vals = rows[metric].dropna().values
                if vals.size < 2:
                    continue
                color = get_method_color(method, m_idx)
                ax.boxplot(
                    vals,
                    positions=[m_idx],
                    widths=0.55,
                    patch_artist=True,
                    boxprops={"facecolor": color, "alpha": 0.60, "linewidth": 0.8},
                    medianprops={"color": "black", "linewidth": 1.6},
                    whiskerprops={"color": color, "linewidth": 0.9},
                    capprops={"color": color, "linewidth": 0.9},
                    flierprops={
                        "marker": "o", "markersize": 2.5, "alpha": 0.45,
                        "markerfacecolor": color, "markeredgecolor": color,
                        "linestyle": "none",
                    },
                    showfliers=True,
                )

            # Column titles – top row only.
            if row_idx == 0:
                ax.set_title(MODEL_DISPLAY_NAMES[model], fontsize=FSIZE_LABEL,
                             fontweight="bold", pad=5)
            # Row labels – left column only.
            if col_idx == 0:
                ax.set_ylabel(scale, fontsize=FSIZE_LABEL - 1, labelpad=5)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, axis="y", alpha=0.25, linestyle=":", linewidth=0.8, zorder=0)
            ax.tick_params(axis="y", labelsize=FSIZE_TICK)

            # X-ticks on bottom row only (sharex propagates elsewhere).
            ax.set_xticks(positions)
            if row_idx == n_rows - 1:
                ax.set_xticklabels(
                    [method_plot_label(m) for m in methods],
                    fontsize=FSIZE_TICK, rotation=40, ha="right",
                )
            else:
                ax.set_xticklabels([])

    for row_idx in range(n_rows):
        for col_idx in range(2):
            apply_nse_display_axis(axes[row_idx][col_idx])

    # ── Shared legend at the top ─────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(
            facecolor=get_method_color(m, i), alpha=0.7,
            label=method_plot_label(m),
            edgecolor="gray", linewidth=0.5,
        )
        for i, m in enumerate(methods)
    ]
    fig.legend(
        legend_handles,
        [method_plot_label(m) for m in methods],
        ncol=min(5, len(methods)),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        fontsize=FSIZE_LEGEND,
        frameon=False,
        handlelength=1.5,
        handletextpad=0.5,
        columnspacing=1.0,
    )
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, fontweight="bold",
                     y=1.0 + 0.012 * (1 + len(methods) // 10))

    return fig


def _plot_metric_distribution_grid(
    dfs: Dict[str, pd.DataFrame],
    methods: Sequence[str],
    metric: str,
    fluxes: Sequence[str] = ("NEE", "GPP", "RECO"),
    fontsize: int = 11,
) -> Optional[plt.Figure]:
    """3×2 box-plot grid: rows = fluxes, cols = Quincy / Cable-Pop.

    Each panel shows one box per method for *metric*. NSE is shown on
    [−0.5, 1] with a shaded below-zero band; other metrics use a
    data-driven y-axis.
    """
    methods = sort_methods(methods)
    n_rows = len(fluxes)
    if n_rows == 0 or not methods:
        return None
    if metric not in set().union(*(set(df.columns) for df in dfs.values() if not df.empty)):
        return None

    FSIZE_TICK = fontsize
    FSIZE_LABEL = fontsize + 2
    FSIZE_LEGEND = fontsize - 1
    positions = list(range(len(methods)))
    is_nse = is_nse_metric(metric)

    w = max(10.0, len(methods) * 0.85)
    h = n_rows * 2.4
    fig, axes = plt.subplots(
        nrows=n_rows, ncols=2,
        figsize=(w, h),
        sharex=True,
        sharey="row",
    )
    fig.subplots_adjust(
        left=0.12, right=0.98, top=0.90, bottom=0.10,
        hspace=0.16, wspace=0.06,
    )
    if n_rows == 1:
        axes = np.array(axes).reshape(1, 2)

    plotted = False
    for row_idx, flux in enumerate(fluxes):
        for col_idx, model in enumerate(MODELS):
            ax = axes[row_idx][col_idx]
            df_m = dfs.get(model, pd.DataFrame())
            for m_idx, method in enumerate(methods):
                if df_m.empty or metric not in df_m.columns:
                    continue
                vals = df_m.loc[
                    (df_m["Method"] == method) & (df_m["Flux"] == flux),
                    metric,
                ].dropna().values
                if vals.size < 2:
                    continue
                color = get_method_color(method, m_idx)
                ax.boxplot(
                    vals,
                    positions=[m_idx],
                    widths=0.55,
                    patch_artist=True,
                    boxprops={"facecolor": color, "alpha": 0.60, "linewidth": 0.8},
                    medianprops={"color": "black", "linewidth": 1.6},
                    whiskerprops={"color": color, "linewidth": 0.9},
                    capprops={"color": color, "linewidth": 0.9},
                    flierprops={
                        "marker": "o", "markersize": 2.5, "alpha": 0.45,
                        "markerfacecolor": color, "markeredgecolor": color,
                        "linestyle": "none",
                    },
                    showfliers=True,
                )
                plotted = True

            if row_idx == 0:
                ax.set_title(MODEL_DISPLAY_NAMES[model], fontsize=FSIZE_LABEL,
                             fontweight="bold", pad=5)
            if col_idx == 0:
                ax.set_ylabel(
                    f"{flux}\n{metric_with_unit(metric)}",
                    fontsize=FSIZE_LABEL - 1, labelpad=5,
                )
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, axis="y", alpha=0.25, linestyle=":", linewidth=0.8, zorder=0)
            ax.tick_params(axis="y", labelsize=FSIZE_TICK)
            ax.set_xticks(positions)
            if row_idx == n_rows - 1:
                ax.set_xticklabels(
                    [method_plot_label(m) for m in methods],
                    fontsize=FSIZE_TICK, rotation=40, ha="right",
                )
            else:
                ax.set_xticklabels([])

    if not plotted:
        plt.close(fig)
        return None

    for row_idx in range(n_rows):
        for col_idx in range(2):
            ax = axes[row_idx][col_idx]
            if is_nse:
                apply_nse_display_axis(ax)
            elif metric in TARGET_ZERO_METRICS and metric not in ERROR_DISPLAY_METRICS:
                ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="-",
                           alpha=0.6, zorder=1)

    if metric in ERROR_DISPLAY_METRICS:
        by_method: Dict[str, List[np.ndarray]] = {m: [] for m in methods}
        for df_m in dfs.values():
            if df_m.empty or metric not in df_m.columns:
                continue
            for method in methods:
                arr = _finite_values(df_m.loc[df_m["Method"] == method, metric])
                if arr.size:
                    by_method[method].append(arr)
        merged = {
            method: np.concatenate(parts) if parts else np.array([])
            for method, parts in by_method.items()
        }
        fig._cutout_note = _attach_error_display_limits(
            np.array(axes).ravel(), merged, metric,
        )

    legend_handles = [
        mpatches.Patch(
            facecolor=get_method_color(m, i), alpha=0.7,
            label=method_plot_label(m),
            edgecolor="gray", linewidth=0.5,
        )
        for i, m in enumerate(methods)
    ]
    fig.legend(
        legend_handles,
        [method_plot_label(m) for m in methods],
        ncol=min(5, len(methods)),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        fontsize=FSIZE_LEGEND,
        frameon=False,
        handlelength=1.5,
        handletextpad=0.5,
        columnspacing=1.0,
    )
    return fig


def _distribution_period(period: str) -> str:
    return period if period in {"yearly", "across_years"} else "across_years"


def _order_dist_metrics(available: Sequence[str]) -> List[str]:
    return order_metrics(available)


def _order_igbps(igbps: Iterable[str]) -> List[str]:
    present = [g for g in igbps if pd.notna(g)]
    ordered = [g for g in DIST_IGBP_ORDER if g in present]
    ordered.extend(sorted(g for g in present if g not in DIST_IGBP_ORDER))
    return ordered


def _extract_method_metric(
    df: pd.DataFrame, method: str, flux: str, metric: str,
) -> np.ndarray:
    if df.empty or metric not in df.columns:
        return np.array([])
    mask = (df["Method"] == method) & (df["Flux"] == flux)
    return df.loc[mask, metric].dropna().to_numpy()


def _collect_grouped_method_samples(
    raw: pd.DataFrame,
    model: str,
    period: str,
    group_by: str,
    flux: str,
    metric: str,
    methods: Sequence[str],
    sites: Optional[Sequence[str]] = None,
    igbps: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[str], Dict[str, Dict[str, np.ndarray]]]:
    """Collect per-method metric arrays for each group-by category.

    Returns ``(group_keys, tick_labels, samples[group][method])``.
    """
    methods = list(methods)
    work = raw
    if methods:
        work = work[work["Method"].isin(methods)]
    if sites:
        work = work[work["Site"].isin(sites)]
    apply_igbp = bool(igbps) and group_by != "IGBP"
    if apply_igbp:
        work = work[work["IGBP"].isin(igbps)]

    samples: Dict[str, Dict[str, np.ndarray]] = {}
    group_keys: List[str] = []
    tick_labels: List[str] = []

    if group_by == "Temporal scale":
        event = "full_day" if "full_day" in work.get("Event", pd.Series(dtype=str)).dropna().unique() else None
        if event is not None:
            work = work[work["Event"] == event]
        group_keys = available_cross_scale_timescales(period, work)
        tick_labels = [scale_short_label(s) for s in group_keys]
        for scale in group_keys:
            df_g = work[work["Scale"] == scale]
            samples[scale] = {
                method: _extract_method_metric(df_g, method, flux, metric)
                for method in methods
            }

    elif group_by == "Stress regime":
        extras = _build_extra_subset_dfs(
            work, model, period, list(DIST_STRESS_KEYS), list(methods),
            list(sites) if sites else None,
            list(igbps) if apply_igbp else None,
        )
        for key in DIST_STRESS_KEYS:
            df_g = extras.get(key, pd.DataFrame())
            per_method = {
                method: _extract_method_metric(df_g, method, flux, metric)
                for method in methods
            }
            if all(vals.size == 0 for vals in per_method.values()):
                continue
            group_keys.append(key)
            tick_labels.append(DIST_STRESS_LABELS.get(key, key))
            samples[key] = per_method

    else:  # IGBP
        event = "full_day" if "full_day" in work.get("Event", pd.Series(dtype=str)).dropna().unique() else None
        if event is not None:
            work = work[work["Event"] == event]
        if "Half-Hourly" in work.get("Scale", pd.Series(dtype=str)).dropna().unique():
            work = work[work["Scale"] == "Half-Hourly"]
        if igbps:
            work = work[work["IGBP"].isin(igbps)]
        group_keys = _order_igbps(work["IGBP"].dropna().unique()) if "IGBP" in work.columns else []
        tick_labels = list(group_keys)
        for igbp in group_keys:
            df_g = work[work["IGBP"] == igbp]
            samples[igbp] = {
                method: _extract_method_metric(df_g, method, flux, metric)
                for method in methods
            }

    return group_keys, tick_labels, samples


def _plot_grouped_method_boxes(
    samples: Dict[str, Dict[str, np.ndarray]],
    group_keys: Sequence[str],
    tick_labels: Sequence[str],
    methods: Sequence[str],
    metric: str,
    show_quantiles: bool,
    fontsize: int = 11,
) -> Tuple[Optional[plt.Figure], Optional[str]]:
    """Single clustered boxplot: x = group categories, boxes = methods.

    Returns ``(figure, cutout_note)``. The note reports display clipping
    for RMSE / cRMSE / MAE / Bias, or ``None``.
    """
    methods = sort_methods(methods)
    n_groups = len(group_keys)
    n_methods = len(methods)
    if n_groups == 0 or n_methods == 0:
        return None, None

    is_nse = is_nse_metric(metric)
    plotted = False
    FSIZE_TICK = fontsize
    FSIZE_LABEL = fontsize + 2
    FSIZE_LEGEND = fontsize - 1

    cluster_x = np.arange(n_groups, dtype=float)
    cluster_w = 0.82
    box_w = cluster_w / max(n_methods, 1)
    offsets = (np.arange(n_methods) - (n_methods - 1) / 2.0) * box_w

    fig_w = max(8.5, n_groups * max(1.2, 0.22 * n_methods))
    fig, ax = plt.subplots(figsize=(fig_w, 5.2))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.86, bottom=0.16)

    for g_idx, key in enumerate(group_keys):
        per_method = samples.get(key, {})
        for m_idx, method in enumerate(methods):
            vals = np.asarray(per_method.get(method, []), dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            pos = cluster_x[g_idx] + offsets[m_idx]
            color = get_method_color(method, m_idx)
            if show_quantiles and vals.size >= 2:
                ax.boxplot(
                    vals,
                    positions=[pos],
                    widths=box_w * 0.82,
                    patch_artist=True,
                    boxprops={"facecolor": color, "alpha": 0.60, "linewidth": 0.8},
                    medianprops={"color": "black", "linewidth": 1.6},
                    whiskerprops={"color": color, "linewidth": 0.9},
                    capprops={"color": color, "linewidth": 0.9},
                    flierprops={
                        "marker": "o", "markersize": 2.4, "alpha": 0.40,
                        "markerfacecolor": color, "markeredgecolor": color,
                        "linestyle": "none",
                    },
                    showfliers=True,
                )
            else:
                med = float(np.median(vals))
                half = box_w * 0.32
                ax.plot(
                    [pos - half, pos + half], [med, med],
                    color=color, linewidth=2.4, solid_capstyle="round", zorder=3,
                )
                ax.scatter(
                    [pos], [med], s=18, color=color, zorder=4,
                    edgecolors="none",
                )
            plotted = True

    if not plotted:
        plt.close(fig)
        return None, None

    for g_idx in range(n_groups - 1):
        ax.axvline(g_idx + 0.5, color="0.88", linewidth=0.7, zorder=0)

    ax.set_xticks(cluster_x)
    ax.set_xticklabels(list(tick_labels), fontsize=FSIZE_TICK)
    ax.set_ylabel(metric_with_unit(metric), fontsize=FSIZE_LABEL)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25, linestyle=":", linewidth=0.8, zorder=0)
    ax.tick_params(axis="y", labelsize=FSIZE_TICK)
    ax.set_xlim(cluster_x[0] - 0.6, cluster_x[-1] + 0.6)

    cutout_note: Optional[str] = None
    if is_nse:
        apply_nse_display_axis(ax)
    elif metric in ERROR_DISPLAY_METRICS:
        _, by_method = _values_by_method_from_grouped(samples, methods)
        groups = [
            _finite_values(per_group.get(method, []))
            for per_group in samples.values()
            for method in methods
        ]
        q_low, q_high = DEFAULT_DISPLAY_QUANTILES
        limits = grouped_display_limits(groups, metric, q_low, q_high)
        if limits is not None:
            lo, hi = limits
            apply_error_display_axis(ax, lo, hi, metric)
            cutouts = count_display_cutouts(by_method, lo, hi)
            cutout_note = format_cutout_caption(
                metric, lo, hi, cutouts, q_low=q_low, q_high=q_high,
            )
    elif metric in TARGET_ZERO_METRICS:
        ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="-", alpha=0.6, zorder=1)

    legend_handles = [
        mpatches.Patch(
            facecolor=get_method_color(m, i), alpha=0.7,
            label=method_plot_label(m),
            edgecolor="gray", linewidth=0.5,
        )
        for i, m in enumerate(methods)
    ]
    fig.legend(
        legend_handles,
        [method_plot_label(m) for m in methods],
        ncol=min(5, n_methods),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        fontsize=FSIZE_LEGEND,
        frameon=False,
        handlelength=1.5,
        handletextpad=0.5,
        columnspacing=1.0,
    )
    return fig, cutout_note


@_st_fragment
def render_grouped_method_distributions(
    period: str,
    methods: Sequence[str],
    sites: Optional[Sequence[str]] = None,
    igbps: Optional[Sequence[str]] = None,
    default_metric: Optional[str] = None,
) -> None:
    """Interactive clustered boxplot: group along scale, stress regime, or IGBP."""
    show_plot_info("method_distributions", period)
    st.caption(
        "One clustered boxplot. X-axis categories come from Group by "
        "(temporal scale, stress regime, or IGBP). "
        "Each category has one box per method. "
        "Uses the same period and optional site filter as the table above."
    )

    method_options = list(methods) if methods else paper_methods()
    if not method_options:
        st.warning("Select at least one method at the top of the page.")
        return

    plot_period = _distribution_period(period)
    preview = load_combined_results(plot_period)
    if preview.empty and plot_period != "yearly":
        plot_period = "yearly"
        preview = load_combined_results(plot_period)
    if preview.empty:
        st.info("No evaluation data available for method distributions.")
        return

    avail_metrics = _order_dist_metrics(metric_columns(preview))
    if not avail_metrics:
        st.info("No metrics available for method distributions.")
        return
    metric_default = default_metric if default_metric in avail_metrics else (
        DEFAULT_METRIC if DEFAULT_METRIC in avail_metrics else avail_metrics[0]
    )

    flux_opts = [f for f in LEADERBOARD_FLUXES if f in preview.get("Flux", pd.Series(dtype=str)).unique()]
    if not flux_opts:
        flux_opts = sorted(preview["Flux"].dropna().unique()) if "Flux" in preview.columns else ["GPP"]

    c1, c2, c3, c4 = st.columns([2.2, 1.8, 1.4, 1.8])
    with c1:
        group_by = st.selectbox(
            "Group by",
            options=list(DIST_GROUP_OPTIONS),
            index=0,
            key="dist_group_by",
            help="What the boxes are clustered along on the x-axis.",
        )
    with c2:
        selected_metric = st.selectbox(
            "Metric",
            options=avail_metrics,
            index=avail_metrics.index(metric_default),
            format_func=metric_with_unit,
            key="dist_metric",
        )
    with c3:
        selected_flux = st.selectbox(
            "Flux",
            options=flux_opts,
            index=flux_opts.index("GPP") if "GPP" in flux_opts else 0,
            key="dist_flux",
        )
    with c4:
        selected_model = st.selectbox(
            "Model",
            options=MODELS,
            index=0,
            format_func=lambda m: MODEL_DISPLAY_NAMES[m],
            key="dist_model",
        )

    mcol, qcol = st.columns([5, 1.4])
    with mcol:
        plot_methods = st.multiselect(
            "Methods",
            options=method_options,
            default=method_options,
            format_func=method_display_name,
            key="dist_methods",
            help="Show or hide methods in this plot only.",
        )
    with qcol:
        show_quantiles = st.checkbox(
            "Quantiles",
            value=True,
            key="dist_quantiles",
            help="On: IQR box, whiskers, and outliers. Off: median only.",
        )

    if not plot_methods:
        st.warning("Select at least one method to draw the distributions.")
        return

    if not st.button("Generate distribution plot", key="dist_button"):
        return

    if period != plot_period:
        st.caption(
            f"Distributions use the {plot_period} period "
            f"(event periods have no cross-scale breakdown)."
        )

    raw = load_results(selected_model, plot_period)
    if raw.empty:
        st.info(f"No evaluation data found for {MODEL_DISPLAY_NAMES[selected_model]}.")
        return

    group_keys, tick_labels, samples = _collect_grouped_method_samples(
        raw=raw,
        model=selected_model,
        period=plot_period,
        group_by=group_by,
        flux=selected_flux,
        metric=selected_metric,
        methods=plot_methods,
        sites=sites,
        igbps=igbps,
    )
    fig, cutout_note = _plot_grouped_method_boxes(
        samples=samples,
        group_keys=group_keys,
        tick_labels=tick_labels,
        methods=plot_methods,
        metric=selected_metric,
        show_quantiles=show_quantiles,
    )
    if fig is None:
        st.info("Not enough samples to draw method distributions for this selection.")
        return
    _show_fig(fig)
    if cutout_note:
        st.caption(cutout_note)


DEFAULT_COVERAGE_MAX_PCT = 60
DEFAULT_COVERAGE_N_BINS = 4
MAX_COVERAGE_N_BINS = 10


def _equal_coverage_bins(max_pct: float, n_bins: int) -> List[Tuple[float, float]]:
    """Equally spaced coverage bins from 0 to *max_pct* (last bin is inclusive)."""
    n_bins = max(1, int(n_bins))
    edges = np.linspace(0.0, float(max_pct), n_bins + 1)
    return [(float(edges[i]), float(edges[i + 1])) for i in range(n_bins)]


def _format_coverage_edge(value: float) -> str:
    rounded = round(value)
    if abs(value - rounded) < 1e-6:
        return str(int(rounded))
    return f"{value:.1f}"


def _in_coverage_bin(pct: pd.Series, lo: float, hi: float, last: bool) -> pd.Series:
    if last:
        return (pct >= lo) & (pct <= hi)
    return (pct >= lo) & (pct < hi)


def _plot_coverage_bins_by_igbp(
    df: pd.DataFrame,
    metric: str,
    flux: str,
    methods: Sequence[str],
    bins: List[Tuple[float, float]],
    n_cols: int = 4,
    fontsize: int = 10,
    suptitle: str = "",
) -> Optional[plt.Figure]:
    """Grid (4 cols, rows = IGBP) of binned-coverage line charts.

    Each subplot: x = coverage bins, y = median *metric*, markers + lines
    per method (parametric methods dashed), error bars = IQR. Shared y
    limits come from a robust fence over all IGBP × bin × method cells.
    """
    methods = sort_methods(methods)
    n_methods = len(methods)
    if n_methods == 0:
        return None

    # ── Prepare data ─────────────────────────────────────────────────────────
    work = df[df["Flux"] == flux].copy()
    if "N_measured" not in work.columns or metric not in work.columns:
        return None
    work["_pct"] = work["N_measured"] * 100.0
    work = work[work[metric].notna() & work["_pct"].notna()]

    igbp_types = _order_igbps(work["IGBP"].dropna().unique())
    n_igbp = len(igbp_types)
    if n_igbp == 0:
        return None

    # ── Layout ───────────────────────────────────────────────────────────────
    n_rows = math.ceil(n_igbp / n_cols)
    # sharex=False: each IGBP has different per-bin sample counts shown in tick labels.
    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols,
        figsize=(n_cols * 4.2, n_rows * 3.5),
        sharex=False, sharey=True,
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.91, bottom=0.10,
                        hspace=0.48, wspace=0.10)
    axes_flat = np.array(axes).reshape(-1) if n_igbp > 1 else np.array([axes])

    FSIZE_TICK  = fontsize
    FSIZE_LABEL = fontsize + 1

    n_bins = len(bins)
    cluster_x = np.arange(n_bins, dtype=float)     # cluster centre per bin
    bar_w = 0.7 / max(n_methods, 1)
    bar_offsets = (np.arange(n_methods) - (n_methods - 1) / 2.0) * bar_w

    # Reference method used purely for counting unique site-years per bin.
    ref_method = methods[0] if methods else None

    for ig_idx, igbp in enumerate(igbp_types):
        ax = axes_flat[ig_idx]
        df_ig = work[work["IGBP"] == igbp]

        # ── Per-bin sample counts (unique site-years via reference method) ──
        df_ref = df_ig[df_ig["Method"] == ref_method] if ref_method else df_ig
        bin_counts = []
        for b_idx, (lo, hi) in enumerate(bins):
            n = int(_in_coverage_bin(df_ref["_pct"], lo, hi, b_idx == n_bins - 1).sum())
            bin_counts.append(n)
        total_n = int(sum(bin_counts))

        for m_idx, method in enumerate(methods):
            color = get_method_color(method, m_idx)
            linestyle = "--" if method in PARAMETRIC_METHODS else "-"
            marker = get_method_marker(method)
            df_m = df_ig[df_ig["Method"] == method]

            med_vals, lo_errs, hi_errs, xs = [], [], [], []
            for b_idx, (lo, hi) in enumerate(bins):
                subset = df_m[_in_coverage_bin(df_m["_pct"], lo, hi, b_idx == n_bins - 1)]
                vals = subset[metric].dropna().values
                if vals.size == 0:
                    continue
                med = float(np.median(vals))
                q1  = float(np.percentile(vals, 25))
                q3  = float(np.percentile(vals, 75))
                xpos = cluster_x[b_idx] + bar_offsets[m_idx]
                med_vals.append(med)
                lo_errs.append(max(med - q1, 0.0))   # lower arm of IQR
                hi_errs.append(max(q3 - med, 0.0))   # upper arm of IQR
                xs.append(xpos)

            if not xs:
                continue

            xs_arr  = np.array(xs)
            med_arr = np.array(med_vals)

            ax.errorbar(xs_arr, med_arr,
                        yerr=[np.array(lo_errs), np.array(hi_errs)],
                        fmt="none", color=color, linewidth=0.9,
                        capsize=2.0, capthick=0.8, alpha=0.90, zorder=3,
                        clip_on=True)
            if len(xs) >= 2:
                ax.plot(xs_arr, med_arr, color=color, linewidth=1.3,
                        linestyle=linestyle, marker=marker, markersize=4,
                        alpha=0.90, zorder=4, label=method_plot_label(method),
                        clip_on=True)
            else:
                ax.plot(xs_arr, med_arr, color=color, linewidth=0,
                        marker=marker, markersize=4, alpha=0.90, zorder=4,
                        label=method_plot_label(method), clip_on=True)

        if is_nse_metric(metric):
            apply_nse_display_axis(ax)
        elif metric not in ERROR_DISPLAY_METRICS:
            ax.axhline(0.0, color="gray", linewidth=0.6, linestyle=":",
                       alpha=0.4, zorder=1)

        # Title includes total IGBP sample count.
        ax.set_title(f"{igbp}  (n={total_n})", fontsize=FSIZE_LABEL, fontweight="bold")

        # X-ticks with bin range AND per-bin sample count on each subplot.
        tick_labels = [
            f"{_format_coverage_edge(lo)}–{_format_coverage_edge(hi)}%\n(n={cnt})"
            for (lo, hi), cnt in zip(bins, bin_counts)
        ]
        ax.set_xticks(cluster_x)
        ax.set_xticklabels(tick_labels, fontsize=FSIZE_TICK - 1, ha="center")
        ax.tick_params(axis="y", labelsize=FSIZE_TICK)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", alpha=0.20, linestyle=":", linewidth=0.7, zorder=0)

        if ig_idx % n_cols == 0:
            ax.set_ylabel(metric_with_unit(metric), fontsize=FSIZE_LABEL)

    # Hide empty subplots
    for extra in range(n_igbp, len(axes_flat)):
        axes_flat[extra].set_visible(False)

    # Shared method legend at the top
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels,
                   ncol=min(5, n_methods),
                   loc="upper center", bbox_to_anchor=(0.5, 1.0),
                   fontsize=fontsize - 1, frameon=False,
                   handlelength=2.0, handletextpad=0.4, columnspacing=1.0)

    if suptitle:
        fig.suptitle(suptitle, fontsize=11, fontweight="bold", y=1.02)

    if metric in ERROR_DISPLAY_METRICS:
        by_method = {
            method: work.loc[work["Method"] == method, metric].to_numpy()
            for method in methods
        }
        groups = _coverage_bin_groups(work, metric, methods, bins, last_inclusive=True)
        limits = coverage_display_limits(groups, metric, q_low=0.10, q_high=0.90)
        if limits is not None:
            lo, hi = limits
            for ax in axes_flat:
                if ax is None or (hasattr(ax, "get_visible") and not ax.get_visible()):
                    continue
                apply_error_display_axis(ax, lo, hi, metric)
            fig._cutout_note = format_cutout_caption(
                metric, lo, hi, count_display_cutouts(by_method, lo, hi),
                q_low=0.10, q_high=0.90,
                cell_label="method × IGBP × coverage bin",
                robust_cells=True,
            )
    return fig



# ── Whittaker biome background ───────────────────────────────────────────────
# Vertices are (mean_temperature_°C, precipitation_cm_yr⁻¹).
# Approximate from Whittaker (1975); drawn back-to-front.
_WHITTAKER_BIOMES: List[Tuple[str, str, List[Tuple[float, float]]]] = [
    ("Tundra",
     "#AED9D4",
     [(-15, 0), (-15, 60), (-4, 90), (2, 60), (3, 0)]),
    ("Boreal Forest",
     "#9DC88B",
     [(-15, 60), (-15, 430), (-3, 290), (5, 150), (5, 60), (-4, 90)]),
    ("Temperate Desert",
     "#E0CF70",
     [(3, 0), (3, 55), (5, 60), (5, 30), (20, 30), (20, 0)]),
    ("Woodland / Shrubland",
     "#C07828",
     [(5, 30), (5, 100), (15, 90), (20, 85), (20, 30)]),
    ("Temperate Seasonal Forest",
     "#6FAD5F",
     [(5, 100), (5, 230), (20, 230), (20, 85), (15, 90)]),
    ("Temperate Rain Forest",
     "#4E9957",
     [(5, 230), (5, 450), (20, 450), (20, 230)]),
    ("Subtropical Desert",
     "#C8A820",
     [(20, 0), (20, 55), (30, 70), (30, 0)]),
    ("Tropical Seasonal Forest",
     "#8AAF3C",
     [(20, 55), (20, 230), (30, 230), (30, 70)]),
    ("Tropical Rain Forest",
     "#1B7A3E",
     [(20, 230), (20, 450), (30, 450), (30, 230)]),
]


def _draw_whittaker_background(ax: plt.Axes, alpha: float = 0.22) -> None:
    """Fill Whittaker biome zones into *ax* as semi-transparent polygons."""
    import matplotlib.patches as _patches
    for _name, _color, _verts in _WHITTAKER_BIOMES:
        poly = _patches.Polygon(
            _verts, closed=True,
            facecolor=_color, edgecolor="white",
            linewidth=0.7, alpha=alpha, zorder=1,
        )
        ax.add_patch(poly)


@st.cache_data
def _load_climate_meta(path: str) -> pd.DataFrame:
    """Load and cache the site climate metadata CSV."""
    df = pd.read_csv(path)
    df = df.rename(columns={"site_ID": "Site"})
    df["precip_cm"] = df["precip_yrly_avg"] / 10.0   # mm → cm
    return df[["Site", "temp_avg", "precip_cm"]].dropna(subset=["temp_avg"])


def _plot_climate_space_performance(
    climate_df: pd.DataFrame,
    eval_dfs: Dict[str, pd.DataFrame],
    metric: str,
    methods: List[str],
    fluxes: List[str],
    show_biomes: bool = True,
    gridsize: int = 12,
    hex_alpha: float = 0.92,
    fontsize: int = 11,
    suptitle: str = "",
) -> Optional[plt.Figure]:
    """Whittaker-space hexbin grid.

    Rows = fluxes (GPP / RECO), Columns = methods.
    Each hexagon shows the median metric value of all sites within that cell.
    ``eval_dfs`` is keyed by method name and contains data for a single model
    (filtered upstream), covering all fluxes.
    """
    FSIZE_TICK  = fontsize
    FSIZE_LABEL = fontsize + 1
    FSIZE_TITLE = fontsize + 1

    # ── Aggregate per site (median across site-years) ────────────────────────
    cell_data: Dict[Tuple[str, str], pd.DataFrame] = {}
    all_vals: List[float] = []
    for flux in fluxes:
        for method in methods:
            df_m = eval_dfs.get(method, pd.DataFrame())
            if df_m.empty:
                continue
            sub = df_m[df_m["Flux"] == flux]
            if metric not in sub.columns or sub.empty:
                continue
            site_med = (sub.groupby("Site")[metric]
                          .median()
                          .reset_index()
                          .rename(columns={metric: "_val"}))
            merged = site_med.merge(climate_df, on="Site", how="inner")
            merged = merged.dropna(subset=["_val", "temp_avg", "precip_cm"])
            cell_data[(flux, method)] = merged
            all_vals.extend(merged["_val"].tolist())

    if not all_vals:
        return None

    # ── Colormap & normalization ──────────────────────────────────────────────
    # Colormaps chosen for colorblind safety:
    #   viridis  – perceptually uniform sequential (safe for all CVD types)
    #   viridis_r – reversed for "lower is better" metrics
    #   PuOr_r   – purple↔orange diverging; avoids red/green confusion
    if metric in MAXIMIZE_METRICS:
        cmap = plt.cm.viridis
        extend = "min"
        vmin_def, vmax_def = min(all_vals), 1.0
        if is_nse_metric(metric):
            vmin_def = max(vmin_def, -0.5)
    elif metric in MINIMIZE_METRICS:
        cmap = plt.cm.viridis_r
        extend = "max"
        vmin_def, vmax_def = 0.0, max(all_vals)
    elif metric in TARGET_ZERO_METRICS:
        cmap = plt.cm.PuOr_r
        extend = "both"
        abs_mx = max(abs(v) for v in all_vals)
        vmin_def, vmax_def = -abs_mx, abs_mx
    else:
        cmap = plt.cm.viridis
        extend = "both"
        vmin_def, vmax_def = min(all_vals), max(all_vals)

    norm = plt.Normalize(vmin=vmin_def, vmax=vmax_def)

    # Standalone ScalarMappable for the colorbar (not tied to any hexbin
    # collection, so graying out faces below won't affect it).
    from matplotlib.cm import ScalarMappable as _SM
    colorbar_sm = _SM(cmap=cmap, norm=norm)
    colorbar_sm.set_array([])

    # ── Figure ────────────────────────────────────────────────────────────────
    n_rows, n_cols = len(fluxes), len(methods)
    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols,
        figsize=(n_cols * 5.4, n_rows * 4.4),
        sharex=True, sharey=True,
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.10, right=0.86, top=0.88, bottom=0.10,
        hspace=0.08, wspace=0.07,
    )

    any_data = False
    for row_idx, flux in enumerate(fluxes):
        for col_idx, method in enumerate(methods):
            ax = axes[row_idx][col_idx]

            if show_biomes:
                _draw_whittaker_background(ax, alpha=0.22)
                ax.set_xlim(-15, 30)
                ax.set_ylim(0, 450)

            data = cell_data.get((flux, method))
            if data is not None and not data.empty:
                x_vals = data["temp_avg"].values
                y_vals = data["precip_cm"].values
                c_vals = data["_val"].values

                # Layer 1 – faded colour for ALL occupied cells (mincnt=1).
                # Low-count bins (1–2 samples) only ever get this layer, so
                # they remain visible but desaturated by the low alpha.
                ax.hexbin(
                    x_vals, y_vals,
                    C=c_vals,
                    reduce_C_function=np.median,
                    gridsize=gridsize,
                    cmap=cmap, norm=norm,
                    mincnt=1,
                    alpha=hex_alpha * 0.35,
                    linewidths=0.3,
                    edgecolors="white",
                    zorder=4,
                )

                # Layer 2 – full-opacity colour only for cells with ≥3 samples.
                # Drawn on top; covers layer 1 for high-count bins.
                ax.hexbin(
                    x_vals, y_vals,
                    C=c_vals,
                    reduce_C_function=np.median,
                    gridsize=gridsize,
                    cmap=cmap, norm=norm,
                    mincnt=3,
                    alpha=hex_alpha,
                    linewidths=0.3,
                    edgecolors="white",
                    zorder=5,
                )

                any_data = True

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, alpha=0.15, linestyle=":", linewidth=0.7, zorder=2)
            ax.tick_params(labelsize=FSIZE_TICK)

            if row_idx == 0:
                ax.set_title(method_plot_label(method), fontsize=FSIZE_TITLE,
                             fontweight="bold", pad=5)
            if col_idx == 0:
                ax.set_ylabel(f"{flux}\nPrecipitation (cm yr⁻¹)", fontsize=FSIZE_LABEL)
            if row_idx == n_rows - 1:
                ax.set_xlabel("Mean Temperature (°C)", fontsize=FSIZE_LABEL)

    # ── Shared colorbar ───────────────────────────────────────────────────────
    if any_data:
        cbar_ax = fig.add_axes([0.88, 0.12, 0.025, 0.74])
        cb = fig.colorbar(colorbar_sm, cax=cbar_ax, extend=extend)
        cb.set_label(metric_with_unit(metric), fontsize=FSIZE_LABEL, labelpad=8)
        cb.ax.tick_params(labelsize=FSIZE_TICK)

    # ── Biome legend (bottom, outside) ───────────────────────────────────────
    if show_biomes:
        import matplotlib.patches as _patches
        handles = [
            _patches.Patch(facecolor=c, edgecolor="white", linewidth=0.5,
                           label=n, alpha=0.7)
            for n, c, _ in _WHITTAKER_BIOMES
        ]
        fig.legend(handles, [n for n, _, _ in _WHITTAKER_BIOMES],
                   ncol=min(5, len(_WHITTAKER_BIOMES)),
                   loc="lower center", bbox_to_anchor=(0.43, -0.05),
                   fontsize=fontsize - 2, frameon=False,
                   handlelength=1.2, handletextpad=0.4, columnspacing=0.8)
        fig.subplots_adjust(bottom=0.14)

    if suptitle:
        fig.suptitle(suptitle, fontsize=fontsize + 1, fontweight="bold", y=1.01)

    return fig


def render_climate_space_performance(
    period: str,
    methods: Optional[Sequence[str]] = None,
) -> None:
    """Streamlit section: Whittaker-space hexbin coloured by metric performance.

    Layout: rows = GPP / RECO, columns = selected methods, model chosen via dropdown.
    """
    st.subheader("Climate Space Performance")
    st.caption(
        "Hexagons show the median metric value across all sites that fall within each "
        "climate-space cell. Rows = flux, columns = method. "
        "Background zones are approximate Whittaker biomes."
    )

    meta_path = (
        Path(__file__).resolve().parent.parent
        / "data" / "meta_information" / "temp_wetness.csv"
    )
    if not meta_path.exists():
        st.warning(f"Climate metadata not found at `{meta_path}`.")
        return

    climate_df = _load_climate_meta(str(meta_path))

    # ── Controls – row 1 ──────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        cs_model = st.selectbox(
            "Model", options=list(MODELS),
            format_func=lambda m: MODEL_DISPLAY_NAMES.get(m, m),
            key="cs_model",
        )
    with c2:
        _all_eval_metrics = list(MAXIMIZE_METRICS | MINIMIZE_METRICS | TARGET_ZERO_METRICS)
        cs_metrics = order_metrics(_all_eval_metrics)
        cs_metric = st.selectbox(
            "Metric", options=cs_metrics,
            index=default_metric_index(cs_metrics),
            key="cs_metric",
            format_func=metric_with_unit,
        )
    with c3:
        cs_scale = st.selectbox(
            "Scale", options=list(DEFAULT_TIMESCALES_ORDER),
            index=default_scale_index(DEFAULT_TIMESCALES_ORDER),
            format_func=scale_short_label,
            key="cs_scale",
        )

    # ── Controls – row 2 ──────────────────────────────────────────────────────
    ce1, ce2, ce3 = st.columns(3)
    with ce1:
        cs_event = st.selectbox(
            "Event", options=list(DEFAULT_EVENTS_ORDER),
            index=default_event_index(DEFAULT_EVENTS_ORDER),
            format_func=event_display_name,
            key="cs_event",
        )
    with ce2:
        cs_biomes = st.checkbox("Show biome background", value=True, key="cs_biomes")
    with ce3:
        cs_gridsize = st.slider("Hex grid size", min_value=5, max_value=30,
                                value=12, step=1, key="cs_gridsize")

    cs_methods = list(methods) if methods else paper_methods()
    if not cs_methods:
        st.info("Select at least one method at the top of the page.")
        return

    if not st.button("Generate Climate Space Plot", key="cs_button"):
        return

    # ── Load data for selected model; split by method ─────────────────────────
    raw = load_results(cs_model, "across_years")
    if raw.empty:
        st.warning(f"No across-years evaluation data for {MODEL_DISPLAY_NAMES.get(cs_model, cs_model)}.")
        return

    raw_filtered = raw[
        (raw["Scale"] == cs_scale) &
        (raw["Event"] == cs_event)
    ]

    eval_dfs: Dict[str, pd.DataFrame] = {}
    for method in cs_methods:
        subset = raw_filtered[raw_filtered["Method"] == method]
        eval_dfs[method] = subset

    if all(df.empty for df in eval_dfs.values()):
        st.warning("No data matches the selected Scale / Event for any chosen method.")
        return

    model_label = MODEL_DISPLAY_NAMES.get(cs_model, cs_model)
    with st.spinner("Rendering climate space plot…"):
        fig = _plot_climate_space_performance(
            climate_df=climate_df,
            eval_dfs=eval_dfs,
            metric=cs_metric,
            methods=cs_methods,
            fluxes=["GPP", "RECO"],
            show_biomes=cs_biomes,
            gridsize=int(cs_gridsize),
            fontsize=11,
            suptitle=(
                f"{model_label} — "
                f"{metric_with_unit(cs_metric)} — "
                f"{cs_scale} / {cs_event}"
            ),
        )

    if fig is None:
        st.warning("Unable to generate the plot. Check that metric columns exist.")
    else:
        _show_fig(fig)



@_st_fragment
def render_coverage_bins_by_igbp(
    combined_df: pd.DataFrame,
    period: str,
    methods: Optional[Sequence[str]] = None,
) -> None:
    """Streamlit section: binned-coverage performance grid, one subplot per IGBP."""
    show_plot_info("coverage_bins", period)
    st.caption(
        "Each subplot = one IGBP. Lines and markers show the median metric per "
        "equally spaced coverage bin; error bars = IQR. Parametric methods are "
        "dashed. Shared y-limits use a robust fence over all IGBP × bin × method "
        "cells so a single outlier bin cannot stretch the axis."
    )

    c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
    with c1:
        cb_model = st.selectbox("Model", options=MODELS,
                                format_func=lambda m: MODEL_DISPLAY_NAMES[m],
                                key="cb_model")
    with c2:
        cb_flux = st.selectbox("Flux",
                               options=sorted(combined_df["Flux"].dropna().unique()),
                               key="cb_flux")
    with c3:
        cb_metrics = metric_columns(combined_df)
        cb_metric = st.selectbox(
            "Metric", options=cb_metrics,
            index=default_metric_index(cb_metrics),
            format_func=metric_with_unit, key="cb_metric",
        )
    with c4:
        cb_scales = order_scales(combined_df["Scale"].dropna().unique())
        cb_scale = st.selectbox(
            "Scale", options=cb_scales,
            index=default_scale_index(cb_scales),
            format_func=scale_short_label, key="cb_scale",
        )

    cs1, cs2, cs3 = st.columns(3)
    with cs1:
        evt_opts = order_events(combined_df["Event"].dropna().unique())
        cb_event = st.selectbox(
            "Event", options=evt_opts,
            index=default_event_index(evt_opts),
            format_func=event_display_name,
            key="cb_event",
        )
    with cs2:
        cb_max_pct = st.slider(
            "Max measured (%)",
            min_value=10, max_value=100, value=DEFAULT_COVERAGE_MAX_PCT, step=5,
            key="cb_max_pct",
            help="Upper edge of the last coverage bin. Site-years above this are excluded.",
        )
    with cs3:
        cb_n_bins = st.slider(
            "Number of bins",
            min_value=1, max_value=MAX_COVERAGE_N_BINS, value=DEFAULT_COVERAGE_N_BINS, step=1,
            key="cb_n_bins",
            help="Equally spaced bins from 0% to the maximum measured percentage.",
        )
        coverage_bins = _equal_coverage_bins(cb_max_pct, cb_n_bins)
    st.caption(
        "Bins: "
        + ", ".join(
            f"{_format_coverage_edge(lo)}–{_format_coverage_edge(hi)}%"
            for lo, hi in coverage_bins
        )
    )

    cb_methods = list(methods) if methods else available_paper_methods(combined_df)
    if not cb_methods:
        st.warning("Select at least one method at the top of the page.")
        return

    igbp_opts = _order_igbps(combined_df["IGBP"].dropna().unique())
    cb_igbps = st.multiselect("IGBP (leave empty = all)", options=igbp_opts,
                              default=[], key="cb_igbps",
                              help="Keep only the selected IGBP types.")

    if not st.button("Generate Coverage Bin Plot", key="cb_button"):
        return

    raw = load_results(cb_model, period)
    if raw.empty:
        st.warning(f"No evaluation data found for {cb_model}.")
        return

    plot_df = raw[
        (raw["Scale"] == cb_scale) &
        (raw["Event"] == cb_event) &
        (raw["Method"].isin(cb_methods))
    ]
    if cb_igbps:
        plot_df = plot_df[plot_df["IGBP"].isin(cb_igbps)]
    if "N_measured" in plot_df.columns:
        plot_df = plot_df[plot_df["N_measured"] * 100 <= cb_max_pct]
    if plot_df.empty:
        st.warning("No rows match the selected filters.")
        return

    with st.spinner("Rendering coverage bin plot…"):
        fig = _plot_coverage_bins_by_igbp(
            df=plot_df,
            metric=cb_metric,
            flux=cb_flux,
            methods=cb_methods,
            bins=coverage_bins,
            fontsize=10,
            suptitle=f"{MODEL_DISPLAY_NAMES[cb_model]} — {cb_flux}",
        )
    if fig is None:
        st.warning("Unable to generate the plot with the current selection.")
    else:
        _show_fig(fig)
        note = getattr(fig, "_cutout_note", None)
        if note:
            st.caption(note)


# ═══════════════════════════════════════════════════════════════════════════
# Pulse-composite error timeseries
# For each method: median error and median absolute error across all events,
# aligned to the 5-day evaluation window, plotted vs. half-hourly step.
# ═══════════════════════════════════════════════════════════════════════════

_PULSE_WINDOW_STEPS = 240   # 5 days × 48 HH steps
_PULSE_SOP_STEP     = 120   # SOP marker at noon of day 2: 2×48 + 24 = 120 steps
_PULSE_CACHE_TAG    = "pulse_error_composite_v1"  # bump to invalidate on-disk cache


@st.cache_data(show_spinner=False)
def _load_pulse_event_catalog() -> pd.DataFrame:
    """Load pulse_events.csv; try the configured pulses path first."""
    paths = _configuration_paths()
    candidates = [
        Path(paths.get("pulses", "")) / "pulse_events.csv",
        Path(__file__).resolve().parent.parent / "data" / "pulses" / "pulse_events.csv",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p, parse_dates=["start_eval", "end_eval", "SOP", "EOP"])
            return df
    return pd.DataFrame()


def _pulse_cache_path(model: str, paths_config: Dict[str, str]) -> Path:
    results_dir = Path(paths_config.get("results", "."))
    return results_dir / f"{_PULSE_CACHE_TAG}_{model}_{model_data_id(model)}_{RUN_ID}.parquet"


def _compute_pulse_raw_errors(
    model: str,
    methods: List[str],
    paths_config: Dict[str, str],
    pulse_events: pd.DataFrame,
    fluxes: List[str],
    progress_cb=None,
) -> pd.DataFrame:
    """Compute per-step errors for every (method, site-event, flux).

    Returns a long-form DataFrame with columns:
        method, flux, step, error, abs_error
    where ``step`` is the half-hourly index within the 5-day window (0..239).
    """
    base_path = paths_config.get("partitioned", "")
    data_id   = model_data_id(model)
    records: List[Dict] = []

    sites = sorted(pulse_events["SITE_ID"].dropna().unique())
    n_sites = len(sites)

    for s_idx, site in enumerate(sites):
        if progress_cb:
            progress_cb(s_idx / n_sites, f"Processing site {site} ({s_idx+1}/{n_sites})…")

        site_events = pulse_events[pulse_events["SITE_ID"] == site]

        # Load ground truth once per site
        truth_path = _find_truth_file(paths_config, site, model, data_id)
        if truth_path is None or not truth_path.exists():
            continue
        try:
            truth_raw = pd.read_csv(truth_path)
            truth_raw["DateTime"] = pd.to_datetime(truth_raw["DateTime"], errors="coerce")
            truth_raw = truth_raw.dropna(subset=["DateTime"]).set_index("DateTime")
        except Exception:
            continue

        for method in methods:
            csv_path, _ = _resolve_partitioned_file(
                base_path, site, model, data_id, RUN_ID, method
            )
            if csv_path is None:
                continue
            try:
                pred_raw = pd.read_csv(csv_path)
                pred_raw["DateTime"] = pd.to_datetime(pred_raw["DateTime"], errors="coerce")
                pred_raw = pred_raw.dropna(subset=["DateTime"]).set_index("DateTime")
            except Exception:
                continue

            for _, evt in site_events.iterrows():
                start   = evt["start_eval"]
                win_end = start + pd.Timedelta(days=5) - pd.Timedelta(minutes=30)

                try:
                    pred_win  = pred_raw.loc[start:win_end]
                    truth_win = truth_raw.loc[start:win_end]
                except Exception:
                    continue
                if pred_win.empty or truth_win.empty:
                    continue

                # Align on DateTime
                joined = pred_win.join(truth_win, how="inner", lsuffix="", rsuffix="_truth")
                if joined.empty:
                    continue

                # Half-hourly step relative to start
                steps = ((joined.index - start).total_seconds() / 1800).round().astype(int)
                valid = (steps >= 0) & (steps < _PULSE_WINDOW_STEPS)
                joined = joined.loc[valid]
                steps  = steps[valid]

                for flux in fluxes:
                    truth_col = f"{flux}_truth"
                    if flux not in joined.columns or truth_col not in joined.columns:
                        continue
                    err = (joined[flux] - joined[truth_col]).values
                    for step, e in zip(steps.values, err):
                        if not np.isnan(e):
                            records.append({
                                "method":    method,
                                "flux":      flux,
                                "step":      int(step),
                                "error":     float(e),
                                "abs_error": float(abs(e)),
                            })

    if not records:
        return pd.DataFrame(columns=["method", "flux", "step", "error", "abs_error"])
    return pd.DataFrame(records)


def _aggregate_pulse_errors(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw per-step errors → median + IQR per (method, flux, step)."""
    if raw_df.empty:
        return pd.DataFrame()
    agg = (
        raw_df
        .groupby(["method", "flux", "step"])
        .agg(
            median_error   =("error",     "median"),
            q25_error      =("error",     lambda x: x.quantile(0.25)),
            q75_error      =("error",     lambda x: x.quantile(0.75)),
            median_abs_error=("abs_error", "median"),
            q25_abs_error  =("abs_error", lambda x: x.quantile(0.25)),
            q75_abs_error  =("abs_error", lambda x: x.quantile(0.75)),
            n_samples      =("error",     "count"),
        )
        .reset_index()
    )
    return agg


def _load_or_compute_pulse_errors(
    model: str,
    methods: List[str],
    paths_config: Dict[str, str],
    fluxes: List[str],
    force_recompute: bool = False,
    progress_cb=None,
) -> pd.DataFrame:
    """Return aggregated pulse-error stats, loading from disk cache when available."""
    cache_path = _pulse_cache_path(model, paths_config)

    if not force_recompute and cache_path.exists():
        try:
            cached = pd.read_parquet(cache_path)
            # Check that all requested methods and fluxes are covered
            if (
                set(methods).issubset(set(cached["method"].unique()))
                and set(fluxes).issubset(set(cached["flux"].unique()))
            ):
                return cached
        except Exception:
            pass

    pulse_events = _load_pulse_event_catalog()
    if pulse_events.empty:
        return pd.DataFrame()

    raw = _compute_pulse_raw_errors(
        model, methods, paths_config, pulse_events, fluxes, progress_cb
    )
    if raw.empty:
        return pd.DataFrame()

    agg = _aggregate_pulse_errors(raw)
    if not agg.empty:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            agg.to_parquet(cache_path, index=False)
        except Exception:
            pass
    return agg


def _plot_pulse_error_composite(
    agg_df: pd.DataFrame,
    methods: List[str],
    fluxes: List[str],
    fontsize: int = 11,
    suptitle: str = "",
) -> Optional[plt.Figure]:
    """3×2 figure: rows = fluxes, cols = [median error, median |error|]."""
    if agg_df.empty:
        return None

    n_rows = len(fluxes)
    fig, axes = plt.subplots(
        nrows=n_rows, ncols=2,
        figsize=(13, n_rows * 3.2),
        sharex=True,
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.08, right=0.98, top=0.88 if suptitle else 0.93,
        bottom=0.09, hspace=0.12, wspace=0.25,
    )

    hours = np.arange(_PULSE_WINDOW_STEPS) / 2.0   # steps → hours
    sop_h = _PULSE_SOP_STEP / 2.0                  # 48 h

    col_titles = ["Median Error", "Median |Error|"]

    for col_idx, (val_col, q25_col, q75_col, col_title) in enumerate([
        ("median_error",     "q25_error",     "q75_error",     col_titles[0]),
        ("median_abs_error", "q25_abs_error", "q75_abs_error", col_titles[1]),
    ]):
        for row_idx, flux in enumerate(fluxes):
            ax = axes[row_idx][col_idx]

            # gray pre-SOP shading
            ax.axvspan(0, sop_h, color="0.93", zorder=0)
            ax.axvline(sop_h, color="0.4", lw=1.2, ls="--", zorder=2,
                       label="SOP" if (row_idx == 0 and col_idx == 0) else None)
            ax.axhline(0, color="0.5", lw=0.8, ls=":", zorder=1)

            flux_agg = agg_df[agg_df["flux"] == flux]

            for m_idx, method in enumerate(sort_methods(methods)):
                mdata = flux_agg[flux_agg["method"] == method].sort_values("step")
                if mdata.empty:
                    continue
                style = _get_method_style(method, m_idx)
                h = mdata["step"].values / 2.0
                y     = mdata[val_col].values
                y_lo  = mdata[q25_col].values
                y_hi  = mdata[q75_col].values

                ax.plot(h, y, label=method_plot_label(method),
                        lw=1.5, zorder=4, **style)
                ax.fill_between(h, y_lo, y_hi,
                                color=style["color"], alpha=0.15, zorder=3)

            ax.set_xlim(0, _PULSE_WINDOW_STEPS / 2.0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, alpha=0.15, linestyle=":", linewidth=0.7)
            ax.tick_params(labelsize=fontsize - 1)

            if row_idx == 0:
                ax.set_title(col_title, fontsize=fontsize + 1, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{flux}\n(µmol m⁻² s⁻¹)", fontsize=fontsize)
            if row_idx == n_rows - 1:
                ax.set_xlabel("Hours from start of evaluation window", fontsize=fontsize)
                # x-ticks every 24 h, labelled Day 0..4
                day_ticks = np.arange(0, _PULSE_WINDOW_STEPS / 2.0 + 1, 24)
                ax.set_xticks(day_ticks)
                ax.set_xticklabels(
                    [f"Day {int(t/24)}" for t in day_ticks],
                    fontsize=fontsize - 1,
                )

    # Shared legend below plot
    handles, labels = axes[0][0].get_legend_handles_labels()
    # also grab SOP line if needed
    for ax_row in axes:
        for ax in ax_row:
            h2, l2 = ax.get_legend_handles_labels()
            for hh, ll in zip(h2, l2):
                if ll not in labels:
                    handles.append(hh)
                    labels.append(ll)

    fig.legend(
        handles, labels,
        ncol=min(6, len(handles)),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        fontsize=fontsize - 1,
        frameon=False,
        handlelength=1.8,
        handletextpad=0.4,
        columnspacing=0.8,
    )
    fig.subplots_adjust(bottom=0.13)

    if suptitle:
        fig.suptitle(suptitle, fontsize=fontsize + 2, fontweight="bold", y=0.995)

    return fig


def _processed_subdirs(subdir: str) -> List[Path]:
    """Candidate directories under the configured ``processed`` path + local fallback."""
    dirs: List[Path] = []
    try:
        paths = _configuration_paths()
        processed = paths.get("processed", "")
        if processed:
            dirs.append(Path(processed) / subdir)
    except Exception:
        pass
    dirs.append(
        Path(__file__).resolve().parent.parent / "data" / "processed" / subdir
    )
    return dirs


def _find_processed_all_file(site: str, model: str) -> Optional[Path]:
    """Return ``{processed}/all/{site}_{model}_{DATA_ID}.csv``.

    This is the fully-featured file produced by ``add_pulses`` and contains
    RECO (with pulse), RECO_pulse, RECO_daily, SWC, P, etc.
    """
    filename = f"{site}_{model}_{DATA_ID}.csv"
    for d in _processed_subdirs("all"):
        p = d / filename
        if p.exists():
            return p
    return None


def _find_model_input_file(site: str, model: str) -> Optional[Path]:
    """Return ``{processed}/method_input/{site}_{model}_{DATA_ID}.csv``."""
    filename = f"{site}_{model}_{DATA_ID}.csv"
    for d in _processed_subdirs("method_input"):
        p = d / filename
        if p.exists():
            return p
    return None


def _find_measured_reco_file(site: str) -> Optional[Path]:
    """Return the *measured* file containing RECO_NT / RECO_DT columns."""
    filename = f"{site}_measured_{DATA_ID}.csv"
    for d in _processed_subdirs("method_input"):
        p = d / filename
        if p.exists():
            return p
    return None


def _plot_pulse_env(
    env_df: pd.DataFrame,
    start: pd.Timestamp,
    sop: pd.Timestamp,
    win_end: pd.Timestamp,
    site: str,
    event_label: str,
    model_label: str = "",
    fontsize: int = 11,
) -> plt.Figure:
    """3-panel: RECO (top), Precipitation (middle), SWC (bottom).

    ``env_df`` is the model-specific input file merged with RECO_NT/DT from the
    measured reference file.  Both ``SWC`` (model files) and ``SWC_1`` (measured
    files) are handled transparently.
    """
    FILL = -9999.0

    def _clean(s: pd.Series) -> pd.Series:
        return pd.to_numeric(s, errors="coerce").replace(FILL, np.nan)

    # RECO with pulse (from processed/all file)
    has_reco       = "RECO" in env_df.columns
    # RECO without pulse: reverse add_pulses → RECO - RECO_pulse + RECO_daily
    has_reco_daily = "RECO_daily" in env_df.columns and "RECO_pulse" in env_df.columns
    # Fallback: NT/DT partitioning from measured file
    has_reco_nt    = "RECO_NT" in env_df.columns
    has_reco_dt    = "RECO_DT" in env_df.columns
    has_p       = "P" in env_df.columns
    # Accept both 'SWC' (model files) and 'SWC_1', 'SWC_2', … (measured files)
    swc_cols = [
        c for c in env_df.columns
        if (c == "SWC" or (c.startswith("SWC_") and not c.endswith("QC")))
    ]
    has_swc = bool(swc_cols)

    fig, axes = plt.subplots(
        3, 1,
        figsize=(13, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.5, 1.5]},
    )
    fig.subplots_adjust(
        left=0.08, right=0.98, top=0.88,
        bottom=0.09, hspace=0.08,
    )

    ax_reco, ax_p, ax_swc = axes
    dt = env_df["DateTime"]

    # ── vertical SOP line + pre-SOP shading (shared) ─────────────────────
    for ax in axes:
        ax.axvspan(start, sop, color="0.93", zorder=0, label="_nolegend_")
        ax.axvline(sop, color="0.35", lw=1.2, ls="--", zorder=3,
                   label="SOP" if ax is ax_reco else "_nolegend_")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.18, linestyle=":", linewidth=0.7, zorder=1)
        ax.tick_params(labelsize=fontsize - 1)
        ax.set_xlim(start, win_end)

    # ── RECO panel ────────────────────────────────────────────────────────
    plotted_reco = False
    if has_reco:
        ax_reco.plot(dt, _clean(env_df["RECO"]),
                     color="#2166ac", lw=1.5, label="RECO (with pulse)", zorder=5)
        plotted_reco = True
    if has_reco_daily:
        reco_no_pulse = _clean(env_df["RECO"]) - _clean(env_df["RECO_pulse"]) + _clean(env_df["RECO_daily"])
        ax_reco.plot(dt, reco_no_pulse,
                     color="#2166ac", lw=1.2, ls="--", alpha=0.6,
                     label="RECO (no pulse)", zorder=4)
        plotted_reco = True
    # Fallback: NT/DT from measured file if processed/all columns absent
    if not plotted_reco:
        if has_reco_nt:
            ax_reco.plot(dt, _clean(env_df["RECO_NT"]),
                         color="#2166ac", lw=1.2, label="RECO$_{NT}$", zorder=4)
        if has_reco_dt:
            ax_reco.plot(dt, _clean(env_df["RECO_DT"]),
                         color="#d6604d", lw=1.2, ls="--", label="RECO$_{DT}$", zorder=4)
        if not has_reco_nt and not has_reco_dt:
            ax_reco.text(0.5, 0.5, "RECO not available", transform=ax_reco.transAxes,
                         ha="center", va="center", fontsize=fontsize - 1, color="gray")
    ax_reco.set_ylabel("RECO (µmol m⁻² s⁻¹)", fontsize=fontsize)
    ax_reco.legend(loc="upper right", fontsize=fontsize - 1, frameon=False)

    # ── Precipitation panel ───────────────────────────────────────────────
    if has_p:
        p_vals = _clean(env_df["P"])
        bar_w = pd.Timedelta(minutes=29)
        ax_p.bar(dt, p_vals, width=bar_w, color="#4393c3", alpha=0.85,
                 align="edge", zorder=4)
        ax_p.set_ylabel("P (mm)", fontsize=fontsize)
        ax_p.set_ylim(bottom=0)
    else:
        ax_p.text(0.5, 0.5, "P not available", transform=ax_p.transAxes,
                  ha="center", va="center", fontsize=fontsize - 1, color="gray")
        ax_p.set_ylabel("P (mm)", fontsize=fontsize)

    # ── SWC panel ─────────────────────────────────────────────────────────
    swc_colors = ["#1a9850", "#66bd63", "#a6d96a", "#d9ef8b"]
    if has_swc:
        for i, col in enumerate(swc_cols):
            ax_swc.plot(dt, _clean(env_df[col]),
                        color=swc_colors[i % len(swc_colors)],
                        lw=1.3, label=col.replace("_", " "), zorder=4)
        ax_swc.set_ylabel("SWC (%)", fontsize=fontsize)
        if len(swc_cols) > 1:
            ax_swc.legend(loc="upper right", fontsize=fontsize - 2,
                          frameon=False, ncol=len(swc_cols))
    else:
        ax_swc.text(0.5, 0.5, "SWC not available", transform=ax_swc.transAxes,
                    ha="center", va="center", fontsize=fontsize - 1, color="gray")
        ax_swc.set_ylabel("SWC (%)", fontsize=fontsize)

    ax_swc.set_xlabel("Date / Time", fontsize=fontsize)

    suffix = f" [{model_label}]" if model_label else ""
    fig.suptitle(
        f"{site}{suffix} — {event_label}",
        fontsize=fontsize + 2, fontweight="bold",
    )
    return fig


def _render_pulse_env_explorer(period: str, final_df: pd.DataFrame) -> None:
    """Streamlit section: rain-pulse environment explorer (RECO + P + SWC)."""
    if period != "pulse_event":
        return
    if not {"event_site_ID", "start_eval", "end_eval"}.issubset(final_df.columns):
        return

    st.subheader("Pulse Environment Explorer")
    st.caption(
        "Data from `processed/all/` (fully-featured file including pulse adjustments).  "
        "**RECO with pulse** = solid line; **RECO without pulse** = dashed line "
        "(reversed `add_pulses`: RECO − RECO_pulse + RECO_daily).  "
        "Gray shading = pre-SOP window; dashed vertical line = SOP noon "
        "(pulse applied from 12:00 on the SOP date, matching `add_pulses` logic)."
    )

    site_opts = sorted(final_df["Site"].dropna().unique())
    col1, col2, col3 = st.columns(3)
    with col1:
        pe_site = st.selectbox("Site", options=site_opts, key="penv_site")
    with col2:
        pe_model = st.selectbox(
            "Model",
            options=list(MODELS),
            format_func=lambda m: MODEL_DISPLAY_NAMES.get(m, m),
            key="penv_model",
        )
    with col3:
        pe_fontsize = st.slider("Font size", min_value=9, max_value=16,
                                value=11, key="penv_fontsize")

    # Build event selector for the chosen site (include SOP for the vertical marker)
    _evt_cols = ["event_site_ID", "start_eval", "end_eval", "event_ID"]
    if "SOP" in final_df.columns:
        _evt_cols.append("SOP")
    site_rows = (
        final_df[final_df["Site"] == pe_site][_evt_cols]
        .drop_duplicates(subset=["event_site_ID"])
        .sort_values("event_site_ID")
    )
    if site_rows.empty:
        st.info("No events available for the selected site.")
        return

    event_opts = {
        f"E{int(r.event_site_ID)} ({r.start_eval} to {r.end_eval})": r
        for _, r in site_rows.iterrows()
    }
    selected_label = st.selectbox(
        "Event", options=list(event_opts.keys()), key="penv_event"
    )
    selected_evt = event_opts[selected_label]

    if not st.button("Generate Environment Plot", key="penv_button"):
        return

    model_label = MODEL_DISPLAY_NAMES.get(pe_model, pe_model)

    # ── Locate processed/all file (RECO + pulse columns + P + SWC) ───────
    all_path = _find_processed_all_file(pe_site, pe_model)
    if all_path is None:
        st.warning(
            f"`processed/all` file for **{pe_site} / {model_label}** not found "
            f"(`{pe_site}_{pe_model}_{DATA_ID}.csv`)."
        )
        return

    start   = pd.to_datetime(selected_evt["start_eval"])
    win_end = start + pd.Timedelta(days=5) - pd.Timedelta(minutes=30)
    # SOP marker: noon of the actual SOP date (pulse increase starts at 12:00
    # because add_pulses shifts DateTime back by 12 h before matching dates).
    if "SOP" in selected_evt.index and pd.notna(selected_evt["SOP"]):
        sop = pd.to_datetime(selected_evt["SOP"]).normalize() + pd.Timedelta(hours=12)
    else:
        sop = start + pd.Timedelta(days=2, hours=12)  # fallback

    def _load_window(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
        df = df.dropna(subset=["DateTime"])
        return df[
            (df["DateTime"] >= start) & (df["DateTime"] <= win_end)
        ].reset_index(drop=True)

    try:
        env_df = _load_window(all_path)
    except Exception as exc:
        st.error(f"Failed to read processed/all file: {exc}")
        return

    if env_df.empty:
        st.warning("No data found in the selected event window.")
        return

    with st.spinner("Rendering…"):
        fig = _plot_pulse_env(
            env_df=env_df,
            start=start,
            sop=sop,
            win_end=win_end,
            site=pe_site,
            event_label=selected_label,
            model_label=model_label,
            fontsize=pe_fontsize,
        )

    _show_fig(fig)


@_st_fragment
def render_pulse_error_composite(period: str, methods: Optional[Sequence[str]] = None) -> None:
    """Streamlit section: composite pulse-error timeseries (pulse_event period only)."""
    if period != "pulse_event":
        return

    st.subheader("Pulse-Event Composite Error Timeseries")
    st.caption(
        "For each method, half-hourly prediction errors are computed across all "
        "pulse events and aligned to the 5-day evaluation window (2 days pre-SOP + 3 days "
        "post-SOP).  Lines = median; shaded band = IQR (25th–75th percentile).  "
        "Results are cached to disk so subsequent runs are instant."
    )

    paths_config = _configuration_paths()

    pe_methods = list(methods) if methods else paper_methods()
    c1, c2 = st.columns(2)
    with c1:
        pe_model = st.selectbox(
            "Model",
            options=list(MODELS),
            format_func=lambda m: MODEL_DISPLAY_NAMES.get(m, m),
            key="pe_model",
        )
    with c2:
        pe_fluxes = st.multiselect(
            "Fluxes",
            options=["GPP", "RECO", "NEE"],
            default=["GPP", "RECO", "NEE"],
            key="pe_fluxes",
        )

    pe_force = st.checkbox(
        "Force recompute (ignore disk cache)", value=False, key="pe_force"
    )

    if not pe_methods or not pe_fluxes:
        st.info("Select at least one method at the top of the page, and at least one flux.")
        return

    if not st.button("Generate Pulse Error Plot", key="pe_button"):
        cache_path = _pulse_cache_path(pe_model, paths_config)
        if cache_path.exists():
            st.caption(f"Cache available: `{cache_path.name}` — click to use it.")
        return

    progress_bar = st.progress(0.0)
    status_text  = st.empty()

    def _progress(frac: float, msg: str = "") -> None:
        progress_bar.progress(min(frac, 1.0))
        if msg:
            status_text.text(msg)

    with st.spinner("Computing pulse errors…"):
        agg = _load_or_compute_pulse_errors(
            model=pe_model,
            methods=pe_methods,
            paths_config=paths_config,
            fluxes=pe_fluxes,
            force_recompute=pe_force,
            progress_cb=_progress,
        )

    progress_bar.empty()
    status_text.empty()

    if agg.empty:
        st.warning(
            "No errors could be computed. Check that the partitioned time series "
            "files exist for the selected model."
        )
        return

    model_label = MODEL_DISPLAY_NAMES.get(pe_model, pe_model)
    with st.spinner("Rendering plot…"):
        fig = _plot_pulse_error_composite(
            agg_df=agg[agg["method"].isin(pe_methods) & agg["flux"].isin(pe_fluxes)],
            methods=pe_methods,
            fluxes=pe_fluxes,
            fontsize=11,
            suptitle=f"Pulse-Event Composite Error — {model_label}",
        )

    if fig is None:
        st.warning("Unable to generate the plot.")
    else:
        cache_path = _pulse_cache_path(pe_model, paths_config)
        _show_fig(fig)
        if cache_path.exists():
            st.caption(f"Results cached at `{cache_path}`.")


def render_nse_appendix_plots(
    combined_df: pd.DataFrame,
    period: str,
    event: str,
    methods: Optional[Sequence[str]] = None,
) -> None:
    """Streamlit section: 6×2 timescales and 5×2 events NSE appendix grids."""
    st.subheader("NSE Appendix Plots")
    st.caption(
        "Publication-ready NSE box-plot grids. "
        "6×2 shows all standard timescales; 5×2 shows the five special half-hourly events. "
        "Rows = scale/event, Columns = Quincy / Cable-Pop. Boxes = per-method distributions."
    )

    if is_event_period(period):
        st.info("Appendix plots are available for yearly and across_years periods only.")
        return

    app_methods = list(methods) if methods else available_paper_methods(combined_df)
    if not app_methods:
        st.warning("Select at least one method at the top of the page.")
        return

    app_flux = st.selectbox(
        "Flux", options=["GPP", "RECO", "NEE"], index=0, key="app_flux",
    )

    # ── Two generate buttons side-by-side ──────────────────────────────────
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        st.markdown("**6×2 — Standard Timescales**")
        btn_scales = st.button("Generate Timescales Grid", key="app_btn_scales")
    with btn_col2:
        st.markdown("**5×2 — Special Half-Hourly Events**")
        btn_events = st.button("Generate Events Grid", key="app_btn_events")

    def _load_model_data() -> Tuple[Dict, Dict]:
        dfs_out: Dict[str, pd.DataFrame] = {}
        for model in MODELS:
            raw = load_results(model, period)
            if raw.empty:
                st.warning(f"No evaluation data found for {model}.")
                return {}, {}
            dfs_out[model] = apply_filters_without_timescale(
                raw, event, app_methods, None, None
            )
        return dfs_out, {}

    # ── 6×2 Timescales ─────────────────────────────────────────────────────
    if btn_scales:
        scales = available_cross_scale_timescales(period, combined_df)
        if not scales:
            st.warning("No standard timescales found for the selected period.")
        else:
            dfs, _ = _load_model_data()
            if dfs:
                with st.spinner("Rendering 6×2 timescales grid…"):
                    fig = _plot_nse_appendix_grid(
                        dfs=dfs,
                        extra_scale_dfs={m: {} for m in MODELS},
                        scales=scales,
                        methods=app_methods,
                        flux=app_flux,
                        fontsize=11,
                        suptitle=f"NSE — {app_flux}",
                    )
                if fig:
                    _show_fig(fig)
                else:
                    st.warning("No data available for the selected combination.")

    # ── 5×2 Events ─────────────────────────────────────────────────────────
    if btn_events:
        scales = CROSS_SCALE_EXTRA_HH_SUBSETS
        dfs_raw: Dict[str, pd.DataFrame] = {}
        extra_dfs: Dict[str, Dict[str, pd.DataFrame]] = {}
        ok = True
        for model in MODELS:
            raw = load_results(model, period)
            if raw.empty:
                st.warning(f"No evaluation data found for {model}.")
                ok = False
                break
            dfs_raw[model] = apply_filters_without_timescale(
                raw, event, app_methods, None, None
            )
            extra_dfs[model] = _build_extra_subset_dfs(
                raw, model, period, list(scales), app_methods, None, None
            )
        if ok:
            with st.spinner("Rendering 5×2 events grid…"):
                fig = _plot_nse_appendix_grid(
                    dfs=dfs_raw,
                    extra_scale_dfs=extra_dfs,
                    scales=list(scales),
                    methods=app_methods,
                    flux=app_flux,
                    fontsize=11,
                    suptitle=f"NSE — {app_flux}",
                )
            if fig:
                _show_fig(fig)
            else:
                st.warning("No data available for the selected combination.")


def _draw_cross_scale_panel(
    ax: plt.Axes,
    samples: Dict[str, Dict[str, np.ndarray]],
    methods: Sequence[str],
    timescales: Sequence[str],
    display_mode: str,
    show_uncertainty: bool,
    n_base: int,
    fontsize: int = 10,
) -> None:
    """Render one cross-scale panel into *ax* (shared by the 2×2 overview grid)."""
    x_positions = np.arange(len(timescales))

    if display_mode in {"Boxplot", "Violin"}:
        n_meth = len(methods)
        slot_w = 0.8 / max(n_meth, 1)
        for s_idx, scale in enumerate(timescales):
            for m_idx, method in enumerate(methods):
                vals = samples[method].get(scale, np.array([]))
                if vals.size == 0:
                    continue
                pos = s_idx + (m_idx - (n_meth - 1) / 2) * slot_w
                color = get_method_color(method, m_idx)
                if display_mode == "Boxplot":
                    if show_uncertainty:
                        ax.boxplot(
                            vals, positions=[pos], widths=slot_w * 0.85,
                            patch_artist=True,
                            boxprops={"facecolor": color, "alpha": 0.55, "linewidth": 1.0},
                            medianprops={"color": "black", "linewidth": 1.2},
                            whiskerprops={"color": color, "linewidth": 1.0},
                            capprops={"color": color, "linewidth": 1.0},
                            flierprops={"marker": "o", "markersize": 2, "alpha": 0.4, "color": color},
                            showfliers=False,
                        )
                    else:
                        ax.plot([pos], [float(np.median(vals))], marker="o", markersize=5,
                                color=color, linestyle="none")
                else:  # Violin
                    parts = ax.violinplot([vals], positions=[pos], widths=slot_w * 0.9,
                                         showmeans=False, showmedians=True, showextrema=False)
                    if show_uncertainty:
                        for body in parts["bodies"]:
                            body.set_facecolor(color); body.set_edgecolor(color); body.set_alpha(0.55)
                    else:
                        for body in parts["bodies"]:
                            body.set_visible(False)
                    if "cmedians" in parts:
                        parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(1.2)
        for m_idx, method in enumerate(methods):
            ax.plot([], [], color=get_method_color(method, m_idx), linewidth=2,
                    label=method_plot_label(method))
    else:
        if display_mode in OVERVIEW_QUANTILE_OPTIONS:
            band = OVERVIEW_QUANTILE_OPTIONS[display_mode]
        else:
            band = CROSS_SCALE_DISPLAY_OPTIONS.get(display_mode)
        draw_band = (
            show_uncertainty
            and isinstance(band, tuple)
            and len(band) == 2
        )
        q_low, q_high = band if draw_band else (None, None)
        for m_idx, method in enumerate(methods):
            color = get_method_color(method, m_idx)
            linestyle = "--" if method in PARAMETRIC_METHODS else "-"
            marker = get_method_marker(method)
            meds, lows, highs, xs = [], [], [], []
            for s_idx, scale in enumerate(timescales):
                vals = samples[method].get(scale, np.array([]))
                if vals.size == 0:
                    continue
                xs.append(float(s_idx))
                meds.append(float(np.median(vals)))
                if draw_band:
                    lows.append(float(np.quantile(vals, q_low)))
                    highs.append(float(np.quantile(vals, q_high)))
            if not xs:
                continue
            x_arr = np.array(xs)
            if draw_band:
                ax.fill_between(x_arr, np.array(lows), np.array(highs),
                                color=color, alpha=0.13, linewidth=0)
            ax.plot(x_arr, np.array(meds), color=color, linestyle=linestyle,
                    linewidth=1.6, marker=marker, markersize=5,
                    label=method_plot_label(method))
            if draw_band:
                for arr in (np.array(lows), np.array(highs)):
                    ax.plot(x_arr, arr, color=color, linewidth=0.8, alpha=0.7, linestyle=":")

    ax.set_xticks(x_positions)
    ax.grid(True, axis="y", alpha=0.25, linestyle=":", linewidth=0.8, zorder=0)
    ax.tick_params(axis="both", labelsize=fontsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if n_base > 0 and n_base < len(timescales):
        ax.axvline(n_base - 0.5, color="gray", linewidth=0.9, linestyle="--", alpha=0.5, zorder=1)


def _plot_2x2_correlation_grid(
    dfs: Dict[str, pd.DataFrame],
    extra_scale_dfs: Dict[str, Dict[str, pd.DataFrame]],
    timescales: List[str],
    methods: Sequence[str],
    display_mode: str,
    show_uncertainty: bool,
    n_base_timescales: int,
    metric: str = "Correlation",
) -> Tuple[Optional[plt.Figure], Optional[str]]:
    """Create a publication-ready 2×2 cross-scale grid (rows=GPP/RECO, cols=Quincy/Cable-Pop)."""
    FSIZE_TICK = 10
    FSIZE_LABEL = 12
    FSIZE_LEGEND = 9

    methods = sort_methods(methods)
    fluxes = ["GPP", "RECO"]
    short_labels = [scale_short_label(s) for s in timescales]

    fig, axes = plt.subplots(
        nrows=2, ncols=2,
        figsize=(12, 7),
        sharex=True, sharey="row",
    )
    fig.subplots_adjust(left=0.08, right=0.995, top=0.80, bottom=0.13,
                        hspace=0.07, wspace=0.06)

    legend_handles: List[Any] = []
    legend_labels: List[str] = []
    row_samples: Dict[str, Dict[str, List[np.ndarray]]] = {
        flux: {m: [] for m in methods} for flux in fluxes
    }

    for row_idx, flux in enumerate(fluxes):
        for col_idx, model in enumerate(MODELS):
            ax = axes[row_idx][col_idx]
            df_m = dfs.get(model, pd.DataFrame())
            extra_m = extra_scale_dfs.get(model, {})

            samples = _collect_cross_scale_samples(
                df_m, metric, flux, methods, timescales,
                extra_scale_dfs=extra_m or None,
                skip_lower_clip=(metric in MAXIMIZE_METRICS),
            )
            for method in methods:
                for arr in samples.get(method, {}).values():
                    finite = _finite_values(arr)
                    if finite.size:
                        row_samples[flux][method].append(finite)

            _draw_cross_scale_panel(
                ax=ax, samples=samples, methods=methods,
                timescales=timescales, display_mode=display_mode,
                show_uncertainty=show_uncertainty,
                n_base=n_base_timescales, fontsize=FSIZE_TICK,
            )

            # Column titles on top row
            if row_idx == 0:
                ax.set_title(MODEL_DISPLAY_NAMES[model], fontsize=FSIZE_LABEL,
                             fontweight="bold", pad=5)

            # Row labels on left column
            if col_idx == 0:
                ax.set_ylabel(f"{flux}\n{metric_with_unit(metric)}",
                              fontsize=FSIZE_LABEL, labelpad=5)

            # X-tick labels only on bottom row
            if row_idx == 1:
                ax.set_xticklabels(short_labels, rotation=30, ha="right",
                                   fontsize=FSIZE_TICK)
            else:
                ax.set_xticklabels([])

            # Collect legend from top-left panel
            if row_idx == 0 and col_idx == 0 and not legend_handles:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

    cutout_notes: List[str] = []
    if is_nse_metric(metric):
        for ax_row in axes:
            for ax in ax_row:
                apply_nse_display_axis(ax)
    elif metric in ERROR_DISPLAY_METRICS:
        q_low, q_high = _display_quantile_band(display_mode)
        for row_idx, flux in enumerate(fluxes):
            by_method = {
                method: np.concatenate(parts) if parts else np.array([])
                for method, parts in row_samples[flux].items()
            }
            groups = [
                arr
                for parts in row_samples[flux].values()
                for arr in parts
            ]
            limits = grouped_display_limits(groups, metric, q_low, q_high)
            if limits is None:
                continue
            lo, hi = limits
            for ax in axes[row_idx]:
                apply_error_display_axis(ax, lo, hi, metric)
            cutouts = count_display_cutouts(by_method, lo, hi)
            cutout_notes.append(
                f"{flux}: {format_cutout_caption(metric, lo, hi, cutouts, q_low=q_low, q_high=q_high)}"
            )

    # Shared legend above the grid
    if legend_handles:
        fig.legend(
            legend_handles, legend_labels,
            ncol=min(5, len(legend_handles)),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.01),
            fontsize=FSIZE_LEGEND,
            frameon=False,
            handlelength=2.2,
            handletextpad=0.5,
            columnspacing=1.2,
        )

    note = " ".join(cutout_notes) if cutout_notes else None
    return fig, note


@_st_fragment
def render_2x2_correlation_overview(
    methods: Sequence[str],
) -> None:
    """Site-level 2×2 summary (paper correlation-overview layout).

    Always uses the across-years period so Inter-Annual Variability is available.
    Harvest is excluded from the extra half-hourly subsets.
    """
    show_plot_info("correlation_overview", "across_years")
    st.caption(
        "Site-level 2×2 summary: rows = GPP / RECO, columns = Quincy / Cable-Pop. "
        "Harvest events are excluded. Change the metric and quantile band below."
    )

    if not methods:
        st.warning("Select at least one method above.")
        return

    period = "across_years"
    event = "full_day"
    metrics = list(METRIC_ORDER)
    quantile_keys = list(OVERVIEW_QUANTILE_OPTIONS.keys())
    default_quantile = "10th-90th" if "10th-90th" in quantile_keys else quantile_keys[0]

    c1, c2 = st.columns(2)
    with c1:
        ov_metric = st.selectbox(
            "Metric",
            options=metrics,
            index=default_metric_index(metrics),
            format_func=metric_with_unit,
            key="ov2x2_metric",
        )
    with c2:
        ov_quantile = st.selectbox(
            "Quantiles",
            options=quantile_keys,
            index=quantile_keys.index(default_quantile),
            key="ov2x2_quantile",
            help="Shaded band around the median. None shows the median only.",
        )

    if not st.button("Generate overview plot", key="ov2x2_button"):
        return

    combined_df = load_combined_results(period)
    if combined_df.empty:
        st.warning("No across-years evaluation data found for the summary plot.")
        return

    combined_df = combined_df[combined_df["Method"].isin(methods)]
    if ov_metric not in combined_df.columns:
        st.warning(f"{metric_with_unit(ov_metric)} is not available in the loaded results.")
        return

    ov_extra = list(OVERVIEW_EXTRA_SUBSETS)
    dfs: Dict[str, pd.DataFrame] = {}
    extra_dfs: Dict[str, Dict[str, pd.DataFrame]] = {}
    for model in MODELS:
        raw = load_results(model, period)
        if raw.empty:
            st.warning(f"No evaluation data found for {model}.")
            return
        raw = raw[raw["Method"].isin(methods)]
        dfs[model] = apply_filters_without_timescale(raw, event, list(methods), None, None)
        extra_dfs[model] = _build_extra_subset_dfs(
            raw, model, period, ov_extra, list(methods), None, None
        )

    base_scales = cross_scale_timescales(period, combined_df, None)
    all_labels = list(base_scales) + ov_extra
    if not all_labels:
        st.warning("No timescale data available for the summary plot.")
        return

    fig, cutout_note = _plot_2x2_correlation_grid(
        dfs=dfs,
        extra_scale_dfs=extra_dfs,
        timescales=all_labels,
        methods=methods,
        display_mode=ov_quantile,
        show_uncertainty=OVERVIEW_QUANTILE_OPTIONS[ov_quantile] is not None,
        n_base_timescales=len(base_scales),
        metric=ov_metric,
    )
    if fig is None:
        st.warning("Unable to generate the overview plot with the current selection.")
    else:
        _show_fig(fig)
        if cutout_note:
            st.caption(cutout_note)


def _escape_latex(text: str) -> str:
    """Escape plain text for safe inclusion in a LaTeX document."""
    # Backslash must be first.
    replacements = [
        ("\\",       "\\textbackslash{}"),
        ("&",        "\\&"),
        ("%",        "\\%"),
        ("$",        "\\$"),
        ("#",        "\\#"),
        ("_",        "\\_"),
        ("{",        "\\{"),
        ("}",        "\\}"),
        ("~",        "\\textasciitilde{}"),
        ("^",        "\\textasciicircum{}"),
        # Unicode → LaTeX math
        ("↑",        "$\\uparrow$"),
        ("↓",        "$\\downarrow$"),
        ("→0",       "$\\to 0$"),
        ("—",        "---"),
        ("µ",        "$\\mu$"),
        ("⁻¹",       "$^{-1}$"),
        ("⁻²",       "$^{-2}$"),
        ("⁻³",       "$^{-3}$"),
        ("·",        "$\\cdot$"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def _multicolumn_header_lines(
    columns: pd.MultiIndex,
) -> List[str]:
    """Return LaTeX header lines (including \\cmidrule rules) for a MultiIndex."""
    nlevels = columns.nlevels
    n_data_cols = len(columns)
    lines: List[str] = []

    for level in range(nlevels):
        level_vals = [col[level] for col in columns]
        cells: List[str] = [""]          # empty cell for the row-index column
        cmidrules: List[str] = []
        i = 0
        col_offset = 2                   # 1-based: col 1 = row index, data starts at 2
        while i < len(level_vals):
            val = level_vals[i]
            span = 1
            while i + span < len(level_vals) and level_vals[i + span] == val:
                span += 1
            escaped = _escape_latex(str(val))
            if span > 1:
                cells.append(f"\\multicolumn{{{span}}}{{c}}{{{escaped}}}")
                if level < nlevels - 1:
                    cmidrules.append(
                        f"\\cmidrule(lr){{{col_offset}-{col_offset + span - 1}}}"
                    )
            else:
                cells.append(escaped)
            col_offset += span
            i += span

        lines.append("    " + " & ".join(cells) + " \\\\")
        if cmidrules:
            lines.append("    " + " ".join(cmidrules))

    return lines


def _col_better_direction(col_name: str) -> str:
    """Return 'max', 'min', or 'zero' for a column based on its arrow annotation."""
    if "↑" in col_name:
        return "max"
    if "↓" in col_name:
        return "min"
    if "→0" in col_name:
        return "zero"
    return "max"


def _best_cells_per_column(
    ref: pd.DataFrame,
    is_pct: bool,
) -> set:
    """Return a set of (row_label, col_key) pairs that hold the best value per column.

    Ties are included (rounded to 3 decimal places for stability).
    """
    best: set = set()
    for col in ref.columns:
        col_vals = ref[col]
        if not pd.api.types.is_numeric_dtype(col_vals):
            continue
        valid = col_vals.dropna().round(3)
        if valid.empty:
            continue

        if is_pct:
            direction = "max"
        elif isinstance(col, tuple):
            # MultiIndex: metric is at level 1 (Flux × Metric × LSM layout)
            metric_part = col[1] if len(col) > 1 else col[-1]
            direction = _col_better_direction(str(metric_part))
        else:
            direction = _col_better_direction(str(col))

        if direction == "max":
            target = valid.max()
            winners = valid[valid == target].index
        elif direction == "min":
            target = valid.min()
            winners = valid[valid == target].index
        else:  # zero — minimise |value|
            target = valid.abs().min()
            winners = valid[valid.abs() == target].index

        for row_idx in winners:
            best.add((row_idx, col))
    return best


def _df_to_latex_code(
    df: pd.DataFrame,
    caption: str = "",
    label: str = "",
    is_pct: bool = False,
    na_rep: str = "---",
    bold_best: bool = False,
    numeric_ref: Optional[pd.DataFrame] = None,
    escape_index: bool = True,
) -> str:
    """Generate a copy-ready LaTeX booktabs table from *df*.

    Parameters
    ----------
    df:          Source DataFrame (index = rows, columns may be a MultiIndex).
    caption:     Table caption (optional).
    label:       LaTeX \\label key (optional).
    is_pct:      When True float values are formatted as ``XX.X\\%``.
    na_rep:      Replacement for NaN / missing cells.
    bold_best:   Wrap the best cell in each column with ``\\textbf{}``.
    numeric_ref: Numeric DataFrame with the same shape as *df* used to
                 determine the best cell.  When None *df* itself is used
                 (works only when *df* already contains numeric values).
    """
    work = df.copy()

    # --- Determine which cells to bold ----------------------------------------
    best_cells: set = set()
    if bold_best:
        ref = numeric_ref if numeric_ref is not None else df
        best_cells = _best_cells_per_column(ref, is_pct)

    # --- Cell formatter -------------------------------------------------------
    def _fmt_cell(val: Any, row_label: Any, col_key: Any) -> str:
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return na_rep
        if is_pct and isinstance(val, (int, float)):
            raw = f"{val:.1f}\\%"
        else:
            raw = _escape_latex(str(val))
        if bold_best and (row_label, col_key) in best_cells:
            return f"\\textbf{{{raw}}}"
        return raw

    # --- Column spec ----------------------------------------------------------
    multi = isinstance(work.columns, pd.MultiIndex)
    col_spec = "l" + "r" * len(work.columns)

    # --- Build lines ----------------------------------------------------------
    lines: List[str] = ["% Requires \\usepackage{booktabs}"]
    lines.append("\\begin{table}[htbp]")
    lines.append("  \\centering")
    if caption:
        lines.append(f"  \\caption{{{_escape_latex(caption)}}}")
    if label:
        lines.append(f"  \\label{{{label}}}")
    lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    lines.append("    \\toprule")

    # Header rows
    if multi:
        for hdr_line in _multicolumn_header_lines(work.columns):
            lines.append(hdr_line)
    else:
        idx_name = _escape_latex(str(work.index.name or "Method"))
        col_headers = [idx_name] + [_escape_latex(str(c)) for c in work.columns]
        lines.append("    " + " & ".join(col_headers) + " \\\\")

    lines.append("    \\midrule")

    # Data rows
    for idx_val, row in work.iterrows():
        idx_cell = str(idx_val) if not escape_index else _escape_latex(str(idx_val))
        cells = [idx_cell] + [
            _fmt_cell(v, idx_val, col) for v, col in zip(row, work.columns)
        ]
        lines.append("    " + " & ".join(cells) + " \\\\")

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def _render_latex_expander(
    df: pd.DataFrame,
    caption: str = "",
    label: str = "",
    is_pct: bool = False,
    bold_best: bool = False,
    numeric_ref: Optional[pd.DataFrame] = None,
    escape_index: bool = True,
) -> None:
    """Render a collapsible LaTeX expander beneath a table."""
    with st.expander("LaTeX", expanded=False):
        code = _df_to_latex_code(
            df,
            caption=caption,
            label=label,
            is_pct=is_pct,
            bold_best=bold_best,
            numeric_ref=numeric_ref,
            escape_index=escape_index,
        )
        st.code(code, language="latex")


def render_flux_tables(df: pd.DataFrame, metrics: List[str]) -> None:
    fluxes = ["NEE", "GPP", "RECO"]
    for flux in fluxes:
        flux_df = df[df["Flux"] == flux]
        st.subheader(f"{flux} Metrics")
        if flux_df.empty:
            st.info(f"No data available for {flux} with the current selection.")
            continue
        summary_display, summary_median = summarise_metrics(flux_df, metrics)
        summary_display = summary_display.loc[sort_methods(summary_display.index)]
        summary_median = summary_median.loc[summary_display.index]

        raw_index = list(summary_display.index)
        display_index = [method_display_name(m) for m in raw_index]
        summary_display.index = display_index
        summary_median.index = display_index

        # Add arrows to column headers
        summary_display = summary_display.rename(columns={col: _add_metric_arrows(col) for col in summary_display.columns})
        summary_median = summary_median.rename(columns={col: _add_metric_arrows(col) for col in summary_median.columns})

        styler = _highlight_best(summary_display, summary_median)
        st.dataframe(styler, width="stretch")


@_st_fragment
def render_info_expanders(metrics: Sequence[str], methods: Sequence[str]) -> None:
    show_scale_info()
    show_event_info()
    show_metric_info(list(metrics))
    show_method_info(list(methods))


@_st_fragment
def render_site_performance_map(
    selected_methods: Sequence[str],
    period: str,
    timescale: str,
    event: str,
) -> None:
    show_plot_info("site_map", period)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        model = st.selectbox(
            "Model",
            MODELS,
            index=0,
            format_func=lambda m: MODEL_DISPLAY_NAMES[m],
            key="map_model",
        )

    df = load_results(model, period)
    if df.empty:
        st.warning(f"No results found for model '{model}'. Please ensure the evaluation has been run.")
        return
    df = df[df["Method"].isin(selected_methods)]

    available_fluxes = sorted(df["Flux"].dropna().unique().tolist())
    metrics = metric_columns(df)
    scales = order_scales(df["Scale"].dropna().unique())
    if not available_fluxes or not metrics or not scales:
        st.info("No flux, metric, or scale data available for the site map.")
        return

    flux_index = available_fluxes.index("GPP") if "GPP" in available_fluxes else 0
    scale_index = list(scales).index(timescale) if timescale in scales else default_scale_index(scales)

    with c2:
        map_flux = st.selectbox("Flux", options=available_fluxes, index=flux_index, key="map_flux")
    with c3:
        map_metric = st.selectbox(
            "Metric",
            options=metrics,
            index=default_metric_index(metrics),
            key="map_metric",
            format_func=metric_with_unit,
        )
    with c4:
        map_scale = st.selectbox(
            "Scale",
            options=scales,
            index=scale_index,
            format_func=scale_short_label,
            key="map_scale",
        )

    filtered_base = df[(df["Scale"] == map_scale) & (df["Event"] == event)]
    site_options = sorted(filtered_base["Site"].dropna().unique())
    igbp_options = sorted(filtered_base["IGBP"].dropna().unique())

    cs1, cs2, cs3 = st.columns(3)
    with cs1:
        map_method = st.selectbox(
            "Method",
            options=selected_methods,
            format_func=method_display_name,
            key="map_method",
        )
    with cs2:
        selected_sites = st.multiselect(
            "Sites (optional)",
            options=site_options,
            default=[],
            key="map_sites",
            help="Leave empty to include all sites.",
        )
    with cs3:
        selected_igbps = st.multiselect(
            "IGBP (optional)",
            options=igbp_options,
            default=[],
            key="map_igbps",
            help="Leave empty to include all IGBP types.",
        )

    map_continent = st.selectbox(
        "Continent/Region",
        options=list(CONTINENT_BOUNDS.keys()),
        index=0,
        key="map_continent",
        help="Select a continent to zoom in on, or 'World' to show all sites globally.",
    )

    if not st.button("Generate Site Performance Map"):
        return

    final_df = apply_filters(
        df, map_scale, event, list(selected_methods),
        selected_sites or None, selected_igbps or None,
    )
    if final_df.empty:
        st.warning("No data matches the selected filters.")
        return

    fig = _plot_site_performance_map(
        final_df, map_flux, map_metric, map_method, map_scale, event, map_continent,
        model_label=MODEL_DISPLAY_NAMES[model],
        sites=selected_sites or None,
        igbps=selected_igbps or None,
    )
    if fig is None:
        st.warning("Unable to generate site map with the current selection.")
    else:
        _show_fig(fig)


# ICLR double-blind: the host must not track visitors, and the UI must not
# expose author or viewer identity. Never read st.user, st.context.ip_address,
# cookies, or any other visitor/reviewer statistic.
ANONYMOUS_REVIEW_MODE = True

_ANONYMOUS_ABOUT = (
    "FluxPart-E Dashboard (anonymous conference submission). "
    "This app does not collect, store, or use visitor statistics."
)

_ANONYMOUS_CHROME_CSS = """
<style>
#MainMenu {visibility: hidden;}
header[data-testid="stHeader"] {height: 0; visibility: hidden;}
footer {visibility: hidden;}
.stAppDeployButton, .stDeployButton {display: none;}
[data-testid="stToolbar"] {display: none;}
[data-testid="stDecoration"] {display: none;}
[data-testid="stStatusWidget"] {display: none;}
[data-testid="stHeaderActionElements"] {display: none;}
[data-testid="appCreatorAvatar"] {display: none;}
[data-testid="stAppCreatorAvatar"] {display: none;}
[data-testid="manage-cloud-button"] {display: none;}
</style>
"""


def _disable_visitor_statistics() -> None:
    """Turn off Streamlit telemetry so reviewer visits are not recorded."""
    for key, value in (
        ("browser.gatherUsageStats", False),
        ("client.showErrorDetails", "none"),
        ("client.toolbarMode", "minimal"),
    ):
        try:
            st.set_option(key, value)
        except Exception:
            pass


def _hide_host_identity_chrome() -> None:
    """Hide Streamlit/GitHub chrome that can deanonymize authors or viewers."""
    st.markdown(_ANONYMOUS_CHROME_CSS, unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(
        page_title="FluxPart-E Dashboard",
        layout="wide",
        menu_items={
            "Get Help": None,
            "Report a bug": None,
            "About": _ANONYMOUS_ABOUT,
        },
    )
    if ANONYMOUS_REVIEW_MODE:
        _disable_visitor_statistics()
        _hide_host_identity_chrome()
    st.title("FluxPart-E Dashboard")
    if ANONYMOUS_REVIEW_MODE:
        st.caption(
            "Anonymous conference submission. "
            "This app does not collect or use visitor statistics."
        )

    pointer_files = lfs_pointer_result_files()
    if pointer_files:
        st.error(
            "Evaluation tables are Git LFS pointer files, not the actual CSVs. "
            "Install Git LFS and run `git lfs pull` from the repository root, "
            "or download the ZIP from anonymous.4open.science rather than a "
            "GitHub source archive."
        )
        st.code("\n".join(pointer_files[:30]))
        return

    methods_src = load_combined_results("across_years")
    if methods_src.empty:
        methods_src = load_combined_results("yearly")
    method_options = available_paper_methods(methods_src)
    if not method_options:
        method_options = paper_methods()
    selected_methods = st.multiselect(
        "Methods",
        options=method_options,
        default=method_options,
        format_func=method_display_name,
        help="Chosen once here and used by every plot on this page.",
    )
    if not selected_methods:
        st.warning("Select at least one method to display results.")
        return

    info_src = methods_src
    info_metrics = _order_dist_metrics(metric_columns(info_src)) if not info_src.empty else list(DIST_METRIC_PRIORITY)
    if not info_metrics:
        info_metrics = [m for m in DIST_METRIC_PRIORITY]
    render_info_expanders(info_metrics, selected_methods)

    st.markdown("---")

    render_2x2_correlation_overview(selected_methods)
    st.markdown("---")

    table_state = render_best_method_leaderboard(selected_methods)
    if table_state is None:
        return
    period, timescale, event, combined_df = table_state
    st.markdown("---")

    render_pulse_error_composite(period, methods=selected_methods)
    st.markdown("---")

    render_coverage_bins_by_igbp(combined_df, period, methods=selected_methods)
    st.markdown("---")

    render_site_performance_map(selected_methods, period, timescale, event)


if __name__ == "__main__":
    main()

