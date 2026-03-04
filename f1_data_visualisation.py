# -*- coding: utf-8 -*-
"""Formula 1 dataset visualisations (static PNG export).

This script loads a multi-CSV F1 dataset from `datasetPath`, performs light
cleaning, generates multiple charts, and saves them as PNG images into ./outputs.

No Plotly is used, so you won't need:
- statsmodels (Plotly OLS trendline)
- kaleido (Plotly image export)

OpenCV (cv2) is used to write the final PNG when available. If cv2 is not
installed, the script falls back to Matplotlib's savefig.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Configuration
# ----------------------------

# Dataset folder (keep beside this script)
datasetPath = Path(__file__).resolve().parent / "f1-dataset"

# Output folder
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

# Required CSVs
REQUIRED_KEYS = [
    "drivers",
    "driver_standings",
    "results",
    "constructors",
    "constructor_standings",
    "pit_stops",
    "status",
]


# ----------------------------
# Helpers
# ----------------------------

def load_csv_folder(folder: str | Path) -> dict[str, pd.DataFrame]:
    """Load all .csv files in `folder` into a dict keyed by filename stem."""
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {folder_path}. "
            "Tip: ensure 'f1-dataset' sits beside this script."
        )

    csv_files = sorted([p for p in folder_path.iterdir() if p.is_file() and p.suffix.lower() == ".csv"])
    data: dict[str, pd.DataFrame] = {}
    for p in csv_files:
        data[p.stem] = pd.read_csv(p)
    return data


def assert_required_tables(data: dict[str, pd.DataFrame]) -> None:
    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        raise KeyError(
            f"Missing expected CSVs in datasetPath: {missing}. "
            f"Found: {sorted(list(data.keys()))}"
        )


def prepare_tables(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Keep only the tables and columns needed for the plots."""
    tables: dict[str, pd.DataFrame] = {}

    # Core lookup tables
    tables["drivers"] = data["drivers"][["driverId", "forename", "surname"]].copy()
    tables["constructors"] = data["constructors"][["constructorId", "name"]].copy()
    tables["status"] = data["status"][["statusId", "status"]].copy()

    # Standings / results
    tables["driver_standings"] = data["driver_standings"].copy()
    if "positionText" in tables["driver_standings"].columns:
        tables["driver_standings"].drop(columns=["positionText"], inplace=True)

    tables["constructor_standings"] = data["constructor_standings"].copy()
    if "positionText" in tables["constructor_standings"].columns:
        tables["constructor_standings"].drop(columns=["positionText"], inplace=True)

    tables["results"] = data["results"].copy()
    if "positionText" in tables["results"].columns:
        tables["results"].drop(columns=["positionText"], inplace=True)

    # Pit stops
    tables["pit_stops"] = data["pit_stops"].copy()

    return tables


def save_figure_png(fig: plt.Figure, out_path: Path) -> None:
    """Save a Matplotlib figure to PNG using OpenCV when available."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(out_path)

    # Render figure to an RGBA array (backend-safe on macOS)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())  # shape: (H, W, 4)
    rgb = rgba[..., :3].copy()  # drop alpha

    # Try OpenCV
    try:
        import cv2  # type: ignore

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok = cv2.imwrite(str(out_path), bgr)
        if not ok:
            raise RuntimeError("cv2.imwrite returned False")
        print(f"Saved: {out_path}")
        return
    except Exception as e:
        # Fallback
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved (matplotlib fallback): {out_path} | OpenCV not used: {e}")


# ----------------------------
# Plots
# ----------------------------

def plot_top_10_drivers_by_points(driver_standings: pd.DataFrame, drivers: pd.DataFrame) -> plt.Figure:
    """Bar chart of top 10 drivers by total accumulated points."""
    top10 = driver_standings.groupby("driverId")["points"].sum().reset_index()
    top10 = top10.merge(drivers, on="driverId", how="left")
    top10["full_name"] = top10["forename"].fillna("") + " " + top10["surname"].fillna("")
    top10 = top10.sort_values("points", ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(top10["full_name"], top10["points"])
    ax.set_title("Top 10 Drivers by Total Points")
    ax.set_xlabel("Driver")
    ax.set_ylabel("Total Points")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    return fig


def plot_poles_vs_wins(results: pd.DataFrame, drivers: pd.DataFrame) -> plt.Figure:
    """Scatter chart of pole positions vs race wins + simple trendline (NumPy)."""
    winners = results[results["positionOrder"] == 1]
    driver_wins = winners.groupby("driverId").size().reset_index(name="race_wins")

    pole_starts = results[results["grid"] == 1].groupby("driverId").size().reset_index(name="pole_positions")

    stats = pd.merge(driver_wins, pole_starts, on="driverId", how="left").fillna(0)
    stats = pd.merge(stats, drivers, on="driverId", how="left")
    stats["driver_name"] = stats["forename"].fillna("") + " " + stats["surname"].fillna("")

    x = stats["pole_positions"].to_numpy(dtype=float)
    y = stats["race_wins"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y)

    # Trendline without statsmodels
    if len(x) >= 2 and np.any(x != x[0]):
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ys = m * xs + b
        ax.plot(xs, ys, linestyle="--")

    # Label only top 10 to keep it readable
    top_label = stats.sort_values(["race_wins", "pole_positions"], ascending=False).head(10)
    for _, row in top_label.iterrows():
        ax.annotate(
            row["driver_name"],
            (row["pole_positions"], row["race_wins"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_title("Pole Positions vs Race Wins")
    ax.set_xlabel("Pole Positions")
    ax.set_ylabel("Race Wins")
    fig.tight_layout()
    return fig


# ----------------------------
# Additional plots
# ----------------------------

def plot_top_10_constructors_by_points(constructor_standings: pd.DataFrame, constructors: pd.DataFrame) -> plt.Figure:
    """Bar chart of top 10 constructors by total accumulated points."""
    top10 = constructor_standings.groupby("constructorId")["points"].sum().reset_index()
    top10 = top10.merge(constructors, on="constructorId", how="left")
    top10 = top10.sort_values("points", ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(top10["name"], top10["points"])
    ax.set_title("Top 10 Constructors by Total Points")
    ax.set_xlabel("Constructor")
    ax.set_ylabel("Total Points")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    return fig


def plot_top_10_fastest_laps(results: pd.DataFrame, drivers: pd.DataFrame) -> plt.Figure:
    """Bar chart of top 10 drivers by number of fastest laps (rank == 1)."""
    # 'rank' == 1 indicates fastest lap in many Ergast-style datasets
    fastest = results[results.get("rank").notna()].copy()
    fastest["rank"] = pd.to_numeric(fastest["rank"], errors="coerce")
    fastest = fastest[fastest["rank"] == 1]

    counts = fastest.groupby("driverId").size().reset_index(name="fastest_laps")
    counts = counts.merge(drivers, on="driverId", how="left")
    counts["full_name"] = counts["forename"].fillna("") + " " + counts["surname"].fillna("")
    counts = counts.sort_values("fastest_laps", ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(counts["full_name"], counts["fastest_laps"])
    ax.set_title("Top 10 Drivers by Fastest Laps")
    ax.set_xlabel("Driver")
    ax.set_ylabel("Fastest Laps")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    return fig


def plot_avg_pitstop_duration_by_constructor(pit_stops: pd.DataFrame, results: pd.DataFrame, constructors: pd.DataFrame) -> plt.Figure:
    """Bar chart of average pit stop duration (ms) by constructor (top 10 fastest)."""
    ps = pit_stops.copy()
    ps["milliseconds"] = pd.to_numeric(ps["milliseconds"], errors="coerce")
    ps = ps.dropna(subset=["milliseconds"])

    # Map pit stops to constructor via (raceId, driverId) join with results
    rr = results[["raceId", "driverId", "constructorId"]].drop_duplicates()
    merged = ps.merge(rr, on=["raceId", "driverId"], how="left")
    merged = merged.dropna(subset=["constructorId"])

    avg = merged.groupby("constructorId")["milliseconds"].mean().reset_index()
    avg = avg.merge(constructors, on="constructorId", how="left")

    # Show top 10 fastest (lowest average duration)
    avg = avg.sort_values("milliseconds", ascending=True).head(10)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(avg["name"], avg["milliseconds"])
    ax.set_title("Top 10 Fastest Constructors by Avg Pit Stop Duration")
    ax.set_xlabel("Constructor")
    ax.set_ylabel("Avg Pit Stop Duration (ms)")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    return fig


def plot_top_10_dnfs_by_driver(results: pd.DataFrame, drivers: pd.DataFrame, status: pd.DataFrame) -> plt.Figure:
    """Bar chart of top 10 drivers by DNF count.

    We treat a result as "Finished" if status is exactly 'Finished' or starts with '+' (e.g., '+1 Lap').
    Everything else is counted as a DNF.
    """
    rs = results[["driverId", "statusId"]].copy()
    rs = rs.merge(status, on="statusId", how="left")
    rs["status"] = rs["status"].fillna("")

    finished_mask = (rs["status"].str.lower() == "finished") | (rs["status"].str.startswith("+"))
    dnfs = rs[~finished_mask]

    counts = dnfs.groupby("driverId").size().reset_index(name="dnfs")
    counts = counts.merge(drivers, on="driverId", how="left")
    counts["full_name"] = counts["forename"].fillna("") + " " + counts["surname"].fillna("")
    counts = counts.sort_values("dnfs", ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(counts["full_name"], counts["dnfs"])
    ax.set_title("Top 10 Drivers by DNF Count")
    ax.set_xlabel("Driver")
    ax.set_ylabel("DNFs")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    return fig


# ----------------------------
# Entry point
# ----------------------------

def main() -> None:
    data = load_csv_folder(datasetPath)
    print(f"Loaded {len(data)} CSV files from: {datasetPath}")
    assert_required_tables(data)

    tables = prepare_tables(data)

    fig1 = plot_top_10_drivers_by_points(tables["driver_standings"], tables["drivers"])
    fig2 = plot_poles_vs_wins(tables["results"], tables["drivers"])
    fig3 = plot_top_10_constructors_by_points(tables["constructor_standings"], tables["constructors"])
    fig4 = plot_top_10_fastest_laps(tables["results"], tables["drivers"])
    fig5 = plot_avg_pitstop_duration_by_constructor(tables["pit_stops"], tables["results"], tables["constructors"])
    fig6 = plot_top_10_dnfs_by_driver(tables["results"], tables["drivers"], tables["status"])

    save_figure_png(fig1, OUTPUT_DIR / "f1_top_10_drivers_by_total_points.png")
    save_figure_png(fig2, OUTPUT_DIR / "f1_pole_positions_vs_race_wins.png")
    save_figure_png(fig3, OUTPUT_DIR / "f1_top_10_constructors_by_total_points.png")
    save_figure_png(fig4, OUTPUT_DIR / "f1_top_10_drivers_by_fastest_laps.png")
    save_figure_png(fig5, OUTPUT_DIR / "f1_top_10_fastest_constructors_by_avg_pitstop.png")
    save_figure_png(fig6, OUTPUT_DIR / "f1_top_10_drivers_by_dnfs.png")

    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)
    plt.close(fig5)
    plt.close(fig6)


if __name__ == "__main__":
    main()
