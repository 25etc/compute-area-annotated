# Author: N. Afroze and A.H
# Script description:
# - Read AnyLabeling JSON files
# - Each polygon = 1 grain (labels ignored)
# - Compute grain area in nm^2
# - Save ONE CSV with per-grain rows + overall stats columns
# - Save ONE PNG figure: stats table above histogram 
#
# Output supports:
# - CSV or TXT (choose in annotation_area(..., output_format="csv"/"txt"))
# - One-line toggle to include/exclude pixel^2 in output (SHOW_PIXEL_AREA)

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import csv

# ==============================
# GLOBAL PLOT STYLE (MINIMAL)
# ==============================
mpl.rcParams.update({
    "axes.linewidth": 0.8,
    "axes.edgecolor": "black",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
})

# ==============================
# CALIBRATION (nm per pixel)
# ==============================
PIXEL_SIZE_NM = 0.079179
PIXEL_AREA_NM2 = PIXEL_SIZE_NM * PIXEL_SIZE_NM


class annotation_area:
    def __init__(
        self,
        set_inputdir=True,
        output_format="csv",
        bins=15,
        decimals=2
    ):
        self.set_inputdir = set_inputdir
        self.output_format = output_format.lower()
        self.bins = bins
        self.decimals = decimals
        self.input_path = None

    # --------------------------
    # Folder picker
    # --------------------------
    def _set_input_dir(self):
        root = tk.Tk()
        root.withdraw()
        folder = filedialog.askdirectory(
            title="Select folder with AnyLabeling JSON files"
        )
        root.destroy()
        if not folder:
            raise RuntimeError("No folder selected.")
        return Path(folder).resolve()

    # --------------------------
    # Polygon area (shoelace)
    # --------------------------
    def _polygon_area_px2(self, points):
        x = np.array([p[0] for p in points], dtype=float)
        y = np.array([p[1] for p in points], dtype=float)
        return 0.5 * abs(
            np.dot(x, np.roll(y, -1)) -
            np.dot(y, np.roll(x, -1))
        )

    # --------------------------
    # Analyze ONE json
    # --------------------------
    def analyze_one_json(self, json_path: Path):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        shapes = data.get("shapes", [])

        areas_nm2 = []

        for s in shapes:
            if s.get("shape_type") != "polygon":
                continue

            points = s.get("points", [])
            if len(points) < 3:
                continue

            area_px2 = self._polygon_area_px2(points)
            area_nm2 = area_px2 * PIXEL_AREA_NM2
            areas_nm2.append(area_nm2)

        return areas_nm2

    # --------------------------
    # Analyze folder
    # --------------------------
    def analyze_folder(self):
        # pick folder
        if self.set_inputdir:
            self.input_path = self._set_input_dir()
        else:
            self.input_path = Path(".").resolve()

        json_files = sorted(self.input_path.glob("*.json"))
        if not json_files:
            print("[INFO] No JSON files found.")
            return

        rows = []
        all_areas = []

        for jp in json_files:
            areas = self.analyze_one_json(jp)
            for idx, a in enumerate(areas, start=1):
                all_areas.append(a)
                rows.append({
                    "image": jp.name,
                    "grain_id": idx,
                    "area_nm2": a
                })

        if len(all_areas) == 0:
            print("[INFO] No polygon annotations found.")
            return

        areas_nm2 = np.array(all_areas)

        # --------------------------
        # Statistics
        # --------------------------
        stats = {
            "count": len(areas_nm2),
            "mean": np.mean(areas_nm2),
            "median": np.median(areas_nm2),
            "std": np.std(areas_nm2),
            "min": np.min(areas_nm2),
            "max": np.max(areas_nm2),
        }

        stats_fmt = {
            k: (f"{v:.{self.decimals}f}" if k != "count" else str(v))
            for k, v in stats.items()
        }

        # --------------------------
        # Output folder
        # --------------------------
        out_dir = self.input_path / "area_computed"
        out_dir.mkdir(exist_ok=True)

        # --------------------------
        # CSV output
        # --------------------------
        csv_path = out_dir / "grain_area_results.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "image", "grain_id", "area_nm2",
                "count", "mean", "median", "std", "min", "max"
            ])
            for r in rows:
                writer.writerow([
                    r["image"],
                    r["grain_id"],
                    round(r["area_nm2"], self.decimals),
                    stats_fmt["count"],
                    stats_fmt["mean"],
                    stats_fmt["median"],
                    stats_fmt["std"],
                    stats_fmt["min"],
                    stats_fmt["max"],
                ])

        # ==========================
        # FIGURE: TABLE + HISTOGRAM
        # ==========================
        fig = plt.figure(figsize=(8, 6))
        gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 3.0])

        # ---- TABLE (TOP) ----
        ax_table = fig.add_subplot(gs[0])
        ax_table.axis("off")
        ax_table.set_title("Stats (units = nm²)", pad=6)

        col_labels = ["group", "count", "mean", "median", "std", "min", "max"]
        table = ax_table.table(
            cellText=[[
                "all",
                stats_fmt["count"],
                stats_fmt["mean"],
                stats_fmt["median"],
                stats_fmt["std"],
                stats_fmt["min"],
                stats_fmt["max"],
            ]],
            colLabels=col_labels,
            cellLoc="center",
            loc="center"
        )
        table.scale(1.0, 1.4)

        # --- STYLE: keep outside boundary + horizontal lines, remove internal vertical lines ---
        ncols = len(col_labels)

        # Apply to every cell: show only Top/Bottom lines (no verticals)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("black")
            cell.set_linewidth(0.8)
            cell.visible_edges = "TB"  # keep horizontal rules only

        # Restore left boundary on first column and right boundary on last column
        # Table rows: 0 = header, 1 = data (here we have exactly one data row)
        for row in [0, 1]:
            table[(row, 0)].visible_edges = "LTB"           # left outer border + horizontals
            table[(row, ncols - 1)].visible_edges = "RTB"   # right outer border + horizontals

        # ---- HISTOGRAM (BOTTOM) ----
        ax = fig.add_subplot(gs[1])

        counts, bins = np.histogram(areas_nm2, bins=self.bins)
        widths = np.diff(bins)
        colors = plt.cm.tab20(np.linspace(0, 1, len(counts)))

        ax.bar(
            bins[:-1],
            counts,
            width=widths,
            align="edge",
            color=colors,
            edgecolor="black",
            linewidth=0.5
        )

        ax.set_xlabel("Grain area (nm²)")
        ax.set_ylabel("Count")
        ax.set_title("Grain area distribution")

        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)

        fig.tight_layout()
        fig_path = out_dir / "grain_area_table_plus_histogram_nm2.png"
        fig.savefig(fig_path, dpi=300)
        plt.show()

        print("Saved:")
        print(f" - {csv_path}")
        print(f" - {fig_path}")
