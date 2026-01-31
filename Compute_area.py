# Author: Tian. M and H. A.
# Script description:
# - Read AnyLabeling JSON files
# - Each polygon = 1 grain (labels ignored)
# - Compute grain area in nm^2
# - Compute equivalent diameter (nm) from area
# - Save TWO CSVs:
#     1) per-grain area + overall area stats columns
#     2) per-grain eq_diameter + overall diameter stats columns
# - Save TWO PNG figures:
#     1) stats table above AREA histogram
#     2) stats table above DIAMETER histogram
#
# Output supports:
# - CSV or TXT (choose in annotation_area(..., output_format="csv"/"txt"))

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
PIXEL_SIZE_NM = 0.079179   ### FOR Pristine= 0.15882   brokenatHT=0.15882
PIXEL_AREA_NM2 = PIXEL_SIZE_NM * PIXEL_SIZE_NM


class annotation_area:
    def __init__(
        self,
        set_inputdir=True,
        output_format="csv",
        bins=15,
        decimals=1
    ):
        self.set_inputdir = set_inputdir
        self.output_format = output_format.lower()
        self.bins = bins
        self.decimals = decimals
        self.input_path = None
        self.input_size = 4096
        # self.input_size = np.shape(s.data)[0] = 4096
        self.output_size = 512
        self.factors_nm2 = int(self.input_size // self.output_size) ** 2  # + 10-6

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
    # Equivalent diameter from area
    # D = 2*sqrt(A/pi)
    # --------------------------
    def _eq_diameter_nm(self, area_nm2: float) -> float:
        if area_nm2 <= 0:
            return 0.0
        return 2.0 * np.sqrt(area_nm2 / np.pi)

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
            area_nm2 = area_px2 * PIXEL_AREA_NM2 * self.factors_nm2
            areas_nm2.append(area_nm2)

        return areas_nm2

    # --------------------------
    # FIGURE helper: TABLE + HISTOGRAM
    # --------------------------
    def _save_table_plus_histogram(self, values, stats_fmt, out_path: Path,
                                   x_label: str, title: str):
        values = np.array(values, dtype=float)

        fig = plt.figure(figsize=(8, 6))
        gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 3.0])

        # ---- TABLE (TOP) ----
        ax_table = fig.add_subplot(gs[0])
        ax_table.axis("off")
        ax_table.set_title("Stats (units = nm²)" if "area" in x_label.lower() else "Stats (units = nm)", pad=6)

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
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("black")
            cell.set_linewidth(0.8)
            cell.visible_edges = "TB"  # horizontals only

        for row in [0, 1]:
            table[(row, 0)].visible_edges = "LTB"
            table[(row, ncols - 1)].visible_edges = "RTB"

        # ---- HISTOGRAM (BOTTOM) ----
        ax = fig.add_subplot(gs[1])

        counts, bins = np.histogram(values, bins=self.bins)
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

        ax.set_xlabel(x_label)
        ax.set_ylabel("Count")
        ax.set_title(title)

        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)

        fig.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.show()

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

        rows_area = []
        rows_diam = []
        all_areas = []
        all_diams = []

        for jp in json_files:
            areas = self.analyze_one_json(jp)

            for idx, a in enumerate(areas, start=1):
                all_areas.append(a)
                rows_area.append({
                    "image": jp.name,
                    "grain_id": idx,
                    "area_nm2": a
                })

                d = self._eq_diameter_nm(a)
                all_diams.append(d)
                rows_diam.append({
                    "image": jp.name,
                    "grain_id": idx,
                    "eq_diameter_nm": d
                })

        if len(all_areas) == 0:
            print("[INFO] No polygon annotations found.")
            return

        areas_nm2 = np.array(all_areas, dtype=float)
        diams_nm = np.array(all_diams, dtype=float)

        # --------------------------
        # Statistics (AREA)
        # --------------------------
        stats_area = {
            "count": len(areas_nm2),
            "mean": np.mean(areas_nm2),
            "median": np.median(areas_nm2),
            "std": np.std(areas_nm2),
            "min": np.min(areas_nm2),
            "max": np.max(areas_nm2),
        }
        stats_area_fmt = {
            k: (f"{v:.{self.decimals}f}" if k != "count" else str(v))
            for k, v in stats_area.items()
        }

        # --------------------------
        # Statistics (DIAMETER)
        # --------------------------
        stats_d = {
            "count": len(diams_nm),
            "mean": np.mean(diams_nm),
            "median": np.median(diams_nm),
            "std": np.std(diams_nm),
            "min": np.min(diams_nm),
            "max": np.max(diams_nm),
        }
        stats_d_fmt = {
            k: (f"{v:.{self.decimals}f}" if k != "count" else str(v))
            for k, v in stats_d.items()
        }

        # --------------------------
        # Output folder
        # --------------------------
        out_dir = self.input_path / "area_computed"
        out_dir.mkdir(exist_ok=True)

        # --------------------------
        # CSV output (AREA)
        # --------------------------
        csv_area_path = out_dir / "grain_area_results.csv"
        with open(csv_area_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "image", "grain_id", "area_nm2",
                "count", "mean", "median", "std", "min", "max"
            ])
            for r in rows_area:
                writer.writerow([
                    r["image"],
                    r["grain_id"],
                    round(r["area_nm2"], self.decimals),
                    stats_area_fmt["count"],
                    stats_area_fmt["mean"],
                    stats_area_fmt["median"],
                    stats_area_fmt["std"],
                    stats_area_fmt["min"],
                    stats_area_fmt["max"],
                ])

        # --------------------------
        # CSV output (EQUIVALENT DIAMETER)
        # --------------------------
        csv_d_path = out_dir / "grain_eqdiam_results.csv"
        with open(csv_d_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "image", "grain_id", "eq_diameter_nm",
                "count", "mean", "median", "std", "min", "max"
            ])
            for r in rows_diam:
                writer.writerow([
                    r["image"],
                    r["grain_id"],
                    round(r["eq_diameter_nm"], self.decimals),
                    stats_d_fmt["count"],
                    stats_d_fmt["mean"],
                    stats_d_fmt["median"],
                    stats_d_fmt["std"],
                    stats_d_fmt["min"],
                    stats_d_fmt["max"],
                ])

        # ==========================
        # FIGURES
        # ==========================
        fig_area_path = out_dir / "grain_area_table_plus_histogram_nm2.png"
        self._save_table_plus_histogram(
            values=areas_nm2,
            stats_fmt=stats_area_fmt,
            out_path=fig_area_path,
            x_label="Grain area (nm²)",
            title="Grain area distribution"
        )

        fig_d_path = out_dir / "grain_eqdiam_table_plus_histogram_nm.png"
        self._save_table_plus_histogram(
            values=diams_nm,
            stats_fmt=stats_d_fmt,
            out_path=fig_d_path,
            x_label="Equivalent diameter (nm)",
            title="Equivalent diameter distribution"
        )

        print("Saved:")
        print(f" - {csv_area_path}")
        print(f" - {fig_area_path}")
        print(f" - {csv_d_path}")
        print(f" - {fig_d_path}")
