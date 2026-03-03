# -*- coding: utf-8 -*-
"""
Crown Distress, summary plus HTML report with up to twenty plots
Units, spacing_m is meters between vertical lines, dy is meters along a line,
linear feet totals use spacing_m times 3.280839895
Neighbor mode, if enabled, runs multiple sampling frequencies around f,
aggregates linear feet by severity across those frequencies,
and replaces elevation plots with a single mean plus std plot per severity

Update, rut detection enforces a maximum ridge to ridge span of 1.0 foot along the line
spacing_m does not affect rut detection, it is only used to accumulate linear feet by severity
"""

import arcpy
import os
import io
import base64
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from math import ceil

try:
    import rasterio
except Exception:
    rasterio = None


# ---------------- helpers ----------------

def _feet_per_meter():
    return 3.280839895


# maximum allowed horizontal span between rut ridges along a profile, feet
RUT_MAX_FEET = 3.0
RUT_MAX_METERS = RUT_MAX_FEET * 0.3048  # 1 ft in meters


def _fit_line_percent_slope(y_vals, step_m):
    y = np.asarray(y_vals, dtype=float)
    mask = np.isfinite(y)
    if mask.sum() < 2:
        return np.nan, np.nan, "0"
    x = np.arange(mask.sum(), dtype=float) * float(step_m)
    yy = y[mask]
    m, b = np.polyfit(x, yy, 1)       # slope in meters per meter
    m_pct = float(m) * 100.0          # percent grade
    sign = "+" if m_pct > 0 else "-" if m_pct < 0 else "0"
    return m_pct, float(b), sign


def _avg_slope_from_profile(depths, dy_m):
    y = np.asarray(depths, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 4:
        return np.nan, np.nan, np.nan, "0", "0", 0
    mid_idx = int(y.size // 2)
    peak_idx = int(np.nanargmax(y))
    ref_idx = int((mid_idx + peak_idx) // 2)
    left = y[:ref_idx] if ref_idx > 1 else y[:1]
    right = y[ref_idx + 1:] if ref_idx + 1 < y.size else y[-1:]
    m_left, _, s_left = _fit_line_percent_slope(left, dy_m)
    m_right, _, s_right = _fit_line_percent_slope(right, dy_m)
    avg_abs = np.nanmean([abs(m_left), abs(m_right)])
    return m_left, m_right, avg_abs, s_left, s_right, ref_idx


def _severity_from_depth_inches(depth_in):
    if depth_in > 3.0:
        return "High"
    if 1.0 <= depth_in <= 3.0:
        return "Medium"
    if 0.5 < depth_in < 1.0:
        return "Low"
    return None


def _detect_rut_severity(depths, dy_m, spacing_m, inches_per_meter=39.3701):
    """
    depths in meters, dy_m in meters,
    spacing_m is meters between lines and is not used for detection

    returns, best_severity, best_depth_in, (idx_h1, idx_low, idx_h2)

    constraint, ridge to ridge span along the line must be within RUT_MAX_FEET
    implemented as, idx_h2 - idx_h1 <= ceil(RUT_MAX_METERS / dy_m)
    """
    # explicitly ignore spacing_m in detection
    _ = spacing_m

    y = np.asarray(depths, dtype=float)
    mask = np.isfinite(y)
    if mask.sum() < 3:
        return None, 0.0, None

    vals = y[mask]
    res = []
    low = float(vals[0])
    idx_low = 0
    prev_dir = None
    left_height = None
    peaks_h = [float(vals[0])]
    peaks_i = [0]
    idx_h1 = 0
    h1 = low
    min_depth_diff_m = 0.0127  # half inch

    # samples permitted between rut ridges, at most one foot along the profile
    max_span_samples = max(1, int(math.ceil(RUT_MAX_METERS / float(dy_m)))) 

    for i in range(1, vals.size):
        cur = float(vals[i])
        prev = float(vals[i - 1])
        direction = "UP" if cur > prev else "DOWN"

        if direction == "UP":
            if cur > peaks_h[-1]:
                peaks_h[-1] = cur
                peaks_i[-1] = i
            else:
                peaks_h.append(cur)
                peaks_i.append(i)

        if prev_dir == "DOWN" and direction == "UP":
            low = float(vals[i - 1])
            idx_low = i - 1
            if peaks_i:
                valids = [ii for ii in peaks_i if ii <= idx_low]
                if valids:
                    ph_valid = [peaks_h[peaks_i.index(ii)] for ii in valids]
                    best = int(np.argmax(ph_valid))
                    idx_h1 = valids[best]
                    h1 = float(vals[idx_h1])
                    left_height = h1 - low
                else:
                    left_height = None

        elif prev_dir == "UP" and direction == "DOWN" and left_height is not None:
            if peaks_i:
                valids = [ii for ii in peaks_i if ii > idx_low and ii <= i]
                if valids:
                    ph_valid = [peaks_h[peaks_i.index(ii)] for ii in valids]
                    best = int(np.argmax(ph_valid))
                    idx_h2 = valids[best]
                    h2 = float(vals[idx_h2])
                    right_height = h2 - low
                    rut_depth_m = min(h1, h2) - low
                    rut_depth_in = float(rut_depth_m) * inches_per_meter

                    if (
                        idx_h2 > idx_low > idx_h1
                        and (idx_h2 - idx_h1) <= max_span_samples
                        and (min(left_height, right_height) >= 0.1 * max(left_height, right_height))
                        and rut_depth_m >= min_depth_diff_m
                    ):
                        res.append((rut_depth_in, idx_h1, idx_low, idx_h2))

        prev_dir = direction

    # tail case, check the last rising peak after the last low
    if left_height is not None and peaks_i:
        valids = [ii for ii in peaks_i if ii > idx_low]
        if valids:
            ph_valid = [peaks_h[peaks_i.index(ii)] for ii in valids]
            best = int(np.argmax(ph_valid))
            idx_h2 = valids[best]
            h2 = float(vals[idx_h2])
            right_height = h2 - low
            rut_depth_m = min(h1, h2) - low
            rut_depth_in = float(rut_depth_m) * inches_per_meter

            if (
                idx_h2 > idx_low > idx_h1
                and (idx_h2 - idx_h1) <= max_span_samples
                and (min(left_height, right_height) >= 0.1 * max(left_height, right_height))
                and rut_depth_m >= min_depth_diff_m
            ):
                res.append((rut_depth_in, idx_h1, idx_low, idx_h2))

    if not res:
        return None, 0.0, None

    # pick worst by severity, then by depth inside class
    def _sev_rank(depth_in):
        return {"Low": 1, "Medium": 2, "High": 3}.get(_severity_from_depth_inches(depth_in), 0)

    best = max(res, key=lambda t: (_sev_rank(t[0]), t[0]))
    best_depth = float(best[0])
    best_sev = _severity_from_depth_inches(best_depth)
    best_idxs = (int(best[1]), int(best[2]), int(best[3]))
    return best_sev, best_depth, best_idxs


def _crown_severity(avg_slope_pct, left_sign, right_sign, rut_sev):
    if left_sign == "-" and right_sign == "+":
        return "High"
    x = float(avg_slope_pct)
    if x >= 4.0:
        if rut_sev == "Medium":
            return "Low"
        if rut_sev == "High":
            return "Medium"
        return None
    if 1.0 < x < 4.0:
        if rut_sev == "Low":
            return "Low"
        if rut_sev == "Medium":
            return "Medium"
        if rut_sev == "High":
            return "High"
        return None
    if 0.0 <= x <= 1.0:
        if rut_sev is None:
            return "Low"
        if rut_sev == "Low":
            return "Medium"
        if rut_sev == "Medium":
            return "High"
        if rut_sev == "High":
            return "High"
    return None


def _sample_profile_from_line(src, band1, clipped_line, dy_m, dem_bounds):
    if clipped_line is None:
        return None
    try:
        if clipped_line.partCount == 0 or float(clipped_line.length) <= 0:
            return None
    except Exception:
        return None

    length = float(clipped_line.length)
    nsteps = max(2, int(math.floor(length / dy_m)))
    if nsteps < 2:
        return None

    distances = [i * dy_m for i in range(nsteps)]
    depths = []
    for d in distances:
        try:
            p = clipped_line.positionAlongLine(d).firstPoint
            x, y = p.X, p.Y
            if not (dem_bounds.left <= x <= dem_bounds.right and dem_bounds.bottom <= y <= dem_bounds.top):
                depths.append(np.nan)
                continue
            r, c = src.index(x, y)
            val = band1[r, c]
            if val == -32767:
                val = np.nan
            depths.append(float(val))
        except Exception:
            depths.append(np.nan)
    return depths


# severity shading colors for elevation plots
_SEV_COLOR = {"Low": "tab:blue", "Medium": "tab:orange", "High": "red"}


def _annotate_footnote(ax, info_dict):
    text = "{ " + ", ".join(f"{k}: {v}" for k, v in info_dict.items()) + " }"
    ax.text(
        0.99, -0.18, text,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#444", linewidth=1.0)
    )


def _outline_axes(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_edgecolor("#222")


def _png_data_uri_from_profile(
    depths, dy_m, ref_idx, idx_triplet, m_left_pct, m_right_pct, title_text, rut_sev, crown
):
    y = np.asarray(depths, dtype=float)
    mask = np.isfinite(y)
    y = y[mask]
    if y.size < 3:
        return None

    x = np.arange(y.size, dtype=float) * float(dy_m)
    y0 = np.nanmin(y)

    left = y[:ref_idx] if ref_idx > 1 else y[:1]
    right = y[ref_idx + 1:] if ref_idx + 1 < y.size else y[-1:]

    def linear_fit(vals):
        if vals.size < 2:
            return None, None
        xx = np.arange(vals.size, dtype=float) * float(dy_m)
        m, b = np.polyfit(xx, vals, 1)
        return m, b

    mL, bL = linear_fit(left)
    mR, bR = linear_fit(right)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.fill_between(x, y0, y, alpha=0.25)
    ax.plot(x, y, linewidth=1.8, label="elevation")
    if mL is not None:
        xxL = np.arange(left.size, dtype=float) * float(dy_m)
        yyL = mL * xxL + bL
        ax.plot(xxL, yyL, linewidth=2.2, label=f"left fit {m_left_pct:.2f}%")
    if mR is not None:
        xxR_local = np.arange(right.size, dtype=float) * float(dy_m)
        yyR = mR * xxR_local + bR
        xxR = x[ref_idx + 1: ref_idx + 1 + right.size]
        ax.plot(xxR, yyR, linewidth=2.2, label=f"right fit {m_right_pct:.2f}%")

    ax.axvline(x[ref_idx] if ref_idx < x.size else x[-1], linestyle="--", alpha=0.5, label="reference")

    if idx_triplet is not None:
        idx_h1, idx_low, idx_h2 = idx_triplet
        idx_h1 = max(0, min(idx_h1, x.size - 1))
        idx_low = max(0, min(idx_low, x.size - 1))
        idx_h2 = max(0, min(idx_h2, x.size - 1))
        color = _SEV_COLOR.get(rut_sev, "gray")
        ax.axvspan(x[idx_h1], x[idx_h2], alpha=0.25, color=color)
        ax.scatter([x[idx_h1], x[idx_low], x[idx_h2]], [y[idx_h1], y[idx_low], y[idx_h2]], s=30, color=color)

    _outline_axes(ax)
    info = {"left_slope_pct": f"{m_left_pct:.2f}", "right_slope_pct": f"{m_right_pct:.2f}",
            "rut": rut_sev if rut_sev else "None", "crown": crown if crown else "None"}
    # _annotate_footnote(ax, info)

    ax.set_title(title_text)
    ax.set_xlabel("distance along line, meters")
    ax.set_ylabel("elevation, units of DEM")
    ax.legend(loc="best", fontsize=8)
    ax.margins(x=0)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{data}"


def _png_data_uri_freq_summary(means_ft, stds_ft, title_text, foot_text):
    """
    means_ft, stds_ft are dicts keyed by severity, values in linear feet.
    X: Low, Medium, High. Y: mean linear feet.
    Shows ±1 std as a narrow band and annotates each point with "mean ± std".
    """
    sev_order = ["Low", "Medium", "High"]
    x_idx = np.arange(len(sev_order), dtype=float)

    y = np.array([float(means_ft.get(s, np.nan)) for s in sev_order], dtype=float)
    s = np.array([float(stds_ft.get(s, 0.0)) for s in sev_order], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 3))

    # std band per category
    half_width = 0.35
    for xi, mu, sd in zip(x_idx, y, s):
        if np.isnan(mu):
            continue
        y0 = max(0.0, mu - (0.0 if np.isnan(sd) else sd))
        y1 = mu + (0.0 if np.isnan(sd) else sd)
        ax.fill_between([xi - half_width, xi + half_width], [y0, y0], [y1, y1], alpha=0.25)

    # connect means with a line + markers
    ax.plot(x_idx, y, marker="o", linewidth=2.2, label="mean linear feet")

    # annotate each point with mean ± std
    for xi, mu, sd in zip(x_idx, y, s):
        if np.isnan(mu):
            continue
        sd_val = 0.0 if np.isnan(sd) else sd
        label = f"{mu:.2f} ft ± {sd_val:.2f}"
        ax.annotate(
            label,
            xy=(xi, mu),
            xytext=(0, 10), textcoords="offset points",
            ha="center", va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#666", linewidth=0.8)
        )

    # x/y cosmetics
    ax.set_xticks(x_idx)
    ax.set_xticklabels(sev_order)
    ax.set_ylabel("linear feet")
    ax.set_title(title_text)
    ax.legend(loc="best", fontsize=8)
    _outline_axes(ax)
    _annotate_footnote(ax, {"mode": "frequency_neighbor", "note": foot_text})

    # prevent annotation clipping: expand y-limit a bit if we have data
    finite_upper = y + np.where(np.isnan(s), 0.0, s)
    finite_upper = finite_upper[np.isfinite(finite_upper)]
    if finite_upper.size:
        ax.set_ylim(0.0, max(0.1, float(np.nanmax(finite_upper)) * 1.15))

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=400)
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{data}"



def _html_report(totals_ft, counts, plots_info, out_path, dem_info, spacing_m, neighbor_mode):
    title = "Crown Distress Report"
    dem_res_x, dem_res_y = dem_info
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def row(sev):
        return f"<tr><td>{sev}</td><td>{totals_ft.get(sev, 0.0):.2f}</td><td>{counts.get(sev, 0)}</td></tr>"

    imgs_html = ""
    for p in plots_info[:20]:
        imgs_html += f"""
        <div class="card">
          <div class="h">{p['header']}</div>
          <img src="{p['data_uri']}" alt="profile plot"/>
          <div class="meta">{p['foot']}</div>
        </div>
        """

    mode_desc = "frequency neighbor mean and std" if neighbor_mode else "single line elevation"

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 20px; }}
h1 {{ margin-bottom: 4px; }}
.small {{ color: #555; font-size: 12px; margin-bottom: 16px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(460px, 1fr)); gap: 12px; }}
table {{ border-collapse: collapse; margin-top: 8px; }}
th, td {{ border: 1px solid #ccc; padding: 6px 10px; }}
.card {{ border: 1px solid #ddd; padding: 10px; border-radius: 6px; }}
.card img {{ width: 100%; height: auto; display: block; }}
.h {{ font-weight: bold; margin-bottom: 6px; }}
.meta {{ color: #333; font-size: 12px; margin-top: 6px; font-family: monospace; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="small">generated at {now} , DEM resolution x {dem_res_x:.4f} m , y {dem_res_y:.4f} m , line spacing {spacing_m:.3f} m , plot mode {mode_desc}</div>

<h2>Summary</h2>
<table>
  <thead><tr><th>severity</th><th>linear feet</th><th>count lines</th></tr></thead>
  <tbody>
    {row("Low")}
    {row("Medium")}
    {row("High")}
  </tbody>
</table>

<h2>Plots</h2>
<div class="grid">
{imgs_html if imgs_html else "<div class='small'>no plots</div>"}
</div>

</body>
</html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


# --------------- toolbox ---------------

class Toolbox(object):
    def __init__(self):
        self.label = "Crown Distress"
        self.alias = "crown_distress"
        self.tools = [CrownDistressSummaryHTML]


class CrownDistressSummaryHTML(object):
    def __init__(self):
        self.label = "Compute Crown Distress , HTML report"
        self.description = (
            "Divide polygon by lines , sample DEM , compute slopes , pick highest rut severity on each line , "
            "combine with slope signs for crown severity , summarize linear feet by severity , "
            "write an HTML report with up to twenty plots , "
            "optional neighbor frequency sweep that shows mean and std of linear feet by severity"
        )
        self.category = "Raster analysis"

    def getParameterInfo(self):
        in_polygon = arcpy.Parameter(
            displayName="Input polygon feature",
            name="in_polygon",
            datatype="Feature Layer",
            parameterType="Required",
            direction="Input",
        )
        in_dem = arcpy.Parameter(
            displayName="Input DEM raster",
            name="in_dem",
            datatype="DERasterDataset",
            parameterType="Required",
            direction="Input",
        )
        spacing_m = arcpy.Parameter(
            displayName="Line spacing in meters",
            name="line_spacing_m",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input",
        )
        spacing_m.value = 0.2

        freq = arcpy.Parameter(
            displayName="Sampling frequency f per 100 pixels",
            name="frequency",
            datatype="GPLong",
            parameterType="Required",
            direction="Input",
        )
        freq.value = 10

        use_neighbor = arcpy.Parameter(
            displayName="Use neighbor frequency sweep",
            name="use_neighbor_sampling",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input",
        )
        use_neighbor.value = False

        freq_range = arcpy.Parameter(
            displayName="Neighbor frequency range, plus or minus",
            name="neighbor_freq_range",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input",
        )
        freq_range.value = 2

        out_html = arcpy.Parameter(
            displayName="Output HTML report",
            name="out_html",
            datatype="DEFile",
            parameterType="Required",
            direction="Output",
        )
        return [in_polygon, in_dem, spacing_m, freq, use_neighbor, freq_range, out_html]

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        if parameters[4].value:
            parameters[5].enabled = True
        else:
            parameters[5].enabled = False

        if parameters[0].value and parameters[6].value is None:
            scratch = arcpy.env.scratchFolder or arcpy.env.scratchGDB
            base = "crown_distress_report_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".html"
            try:
                parameters[6].value = os.path.join(scratch, base)
            except Exception:
                pass
        return

    def updateMessages(self, parameters):
        if rasterio is None:
            parameters[0].setWarningMessage("rasterio not available , please install rasterio")
        return

    def execute(self, parameters, messages):
        arcpy.env.overwriteOutput = True

        in_polygon = parameters[0].valueAsText
        in_dem = parameters[1].valueAsText
        spacing_m = float(parameters[2].value)              # meters between vertical lines
        f_center = int(parameters[3].value)
        neighbor_mode = bool(parameters[4].value)
        neighbor_freq_range = int(parameters[5].value) if parameters[5].value else 0
        out_html = parameters[6].valueAsText

        if rasterio is None:
            raise RuntimeError("rasterio is required")

        # open DEM
        with rasterio.open(in_dem) as src:
            dem_bounds = src.bounds
            dem_crs = src.crs
            res_x, res_y = float(src.res[0]), float(src.res[1])   # meters per pixel
            pix_m = float(np.mean(np.abs(src.res)))               # meters
            band1 = src.read(1)
            arcpy.AddMessage(f"DEM resolution, x, {res_x:.4f} m, y, {res_y:.4f} m")

        dem_sr = arcpy.SpatialReference()
        dem_sr.loadFromString(str(dem_crs))

        # polygon geometry, project if needed
        with arcpy.da.SearchCursor(in_polygon, ["SHAPE@"]) as cur:
            poly_geom = next(cur)[0]
        poly_sr = poly_geom.spatialReference
        if poly_sr.name != dem_sr.name:
            arcpy.AddWarning("Spatial reference mismatch , reprojecting polygon to DEM spatial reference")
            temp_poly = "memory\\temp_poly_proj"
            arcpy.management.Project(in_polygon, temp_poly, dem_sr)
            with arcpy.da.SearchCursor(temp_poly, ["SHAPE@"]) as cur2:
                poly_geom = next(cur2)[0]
            arcpy.management.Delete(temp_poly)

        # step across x in map units
        meters_per_unit = float(dem_sr.metersPerUnit)
        dx_map = spacing_m / meters_per_unit

        # per line contribution to linear feet
        dx_feet = spacing_m * _feet_per_meter()

        # iterate vertical lines across polygon bounds once to get x positions
        ext = poly_geom.extent
        min_x, max_x = float(ext.XMin), float(ext.XMax)
        min_y, max_y = float(ext.YMin), float(ext.YMax)
        xs = np.arange(min_x, max_x + dx_map, dx_map)

        # one full pass for a given frequency
        def _run_once(f_val):
            dy_m = (100.0 / float(f_val)) * float(pix_m)
            totals_ft_run = {"Low": 0.0, "Medium": 0.0, "High": 0.0}
            counts_run = {"Low": 0, "Medium": 0, "High": 0}
            with rasterio.open(in_dem) as src_again:
                band_again = src_again.read(1)
                for x in xs:
                    base_line = arcpy.Polyline(arcpy.Array([arcpy.Point(x, min_y), arcpy.Point(x, max_y)]), dem_sr)
                    clipped = base_line.intersect(poly_geom, 2)
                    if clipped is None:
                        continue
                    try:
                        if clipped.partCount == 0 or float(clipped.length) <= 0:
                            continue
                    except Exception:
                        continue
                    depths = _sample_profile_from_line(src_again, band_again, clipped, dy_m, src_again.bounds)
                    if not depths or np.isfinite(np.array(depths, dtype=float)).sum() < 3:
                        continue
                    m_left, m_right, m_avg, s_left, s_right, ref_idx = _avg_slope_from_profile(depths, dy_m)
                    rut_sev, max_rut_in, idxs = _detect_rut_severity(depths, dy_m, spacing_m)
                    crown = _crown_severity(m_avg, s_left, s_right, rut_sev)
                    if crown in totals_ft_run:
                        totals_ft_run[crown] += dx_feet
                        counts_run[crown] += 1
            return totals_ft_run, counts_run

        plots_info = []

        if neighbor_mode:
            start_f = max(1, f_center - neighbor_freq_range)
            end_f = f_center + neighbor_freq_range
            freq_list = list(range(start_f, end_f + 1))
            arcpy.AddMessage(f"Neighbor frequency sweep, center f, {f_center}, range, plus or minus {neighbor_freq_range}, frequencies, {freq_list}")

            per_f_results = []
            base_totals = None
            base_counts = None
            for f_val in freq_list:
                t_run, c_run = _run_once(f_val)
                per_f_results.append((f_val, t_run, c_run))
                if f_val == f_center:
                    base_totals = t_run
                    base_counts = c_run

            sev_order = ["Low", "Medium", "High"]
            means_ft = {}
            stds_ft = {}
            for sev in sev_order:
                arr = np.array([t[sev] for _, t, _ in per_f_results], dtype=float)
                means_ft[sev] = float(np.nanmean(arr)) if arr.size else np.nan
                stds_ft[sev]  = float(np.nanstd(arr, ddof=1)) if arr.size > 1 else 0.0

            uri = _png_data_uri_freq_summary(
                means_ft, stds_ft,
                title_text="linear feet by severity, mean and std across frequencies",
                foot_text=f"center f {f_center}, range plus or minus {neighbor_freq_range}"
            )
            if uri:
                plots_info.append({
                    "header": "frequency neighbor summary",
                    "data_uri": uri,
                    "foot": "{ plot: mean_std_by_severity, y: linear feet }",
                })

            totals_ft = base_totals if base_totals else {"Low": 0.0, "Medium": 0.0, "High": 0.0}
            counts = base_counts if base_counts else {"Low": 0, "Medium": 0, "High": 0}

        else:
            f_val = f_center
            dy_m = (100.0 / float(f_val)) * float(pix_m)
            totals_ft = {"Low": 0.0, "Medium": 0.0, "High": 0.0}
            counts = {"Low": 0, "Medium": 0, "High": 0}
            all_profiles = []
            sev_rank = {"Low": 1, "Medium": 2, "High": 3, None: 0}

            with rasterio.open(in_dem) as src_again:
                band_again = src_again.read(1)
                for i, x in enumerate(xs):
                    base_line = arcpy.Polyline(arcpy.Array([arcpy.Point(x, min_y), arcpy.Point(x, max_y)]), dem_sr)
                    clipped = base_line.intersect(poly_geom, 2)
                    if clipped is None:
                        continue
                    try:
                        if clipped.partCount == 0 or float(clipped.length) <= 0:
                            continue
                    except Exception:
                        continue
                    depths = _sample_profile_from_line(src_again, band_again, clipped, dy_m, src_again.bounds)
                    if not depths or np.isfinite(np.array(depths, dtype=float)).sum() < 3:
                        continue
                    m_left, m_right, m_avg, s_left, s_right, ref_idx = _avg_slope_from_profile(depths, dy_m)
                    rut_sev, max_rut_in, idxs = _detect_rut_severity(depths, dy_m, spacing_m)
                    crown = _crown_severity(m_avg, s_left, s_right, rut_sev)
                    if crown in totals_ft:
                        totals_ft[crown] += dx_feet
                        counts[crown] += 1
                    all_profiles.append({
                        "i": i, "x": x, "depths": depths, "dy_m": dy_m,
                        "ref_idx": ref_idx, "idxs": idxs,
                        "m_left": m_left, "m_right": m_right, "m_avg": m_avg,
                        "s_left": s_left, "s_right": s_right,
                        "rut_sev": rut_sev, "crown": crown,
                        "rank": (sev_rank.get(crown, 0) * 1_000_000
                                 + sev_rank.get(rut_sev, 0) * 10_000
                                 + int(round(abs(max_rut_in) * 100)))
                    })

            all_profiles.sort(key=lambda d: d["rank"], reverse=True)
            for cand in all_profiles[:20]:
                uri = _png_data_uri_from_profile(
                    cand["depths"], cand["dy_m"], cand["ref_idx"], cand["idxs"],
                    cand["m_left"], cand["m_right"],
                    f"x position {cand['x']:.3f}",
                    cand["rut_sev"], cand["crown"]
                )
                if uri:
                    foot = "{ " + (
                        f"left_slope_pct: {cand['m_left']:.2f}, "
                        f"right_slope_pct: {cand['m_right']:.2f}, "
                        f"avg_slope_pct: {cand['m_avg']:.2f}, "
                        f"signs: {cand['s_left']}{cand['s_right']}, "
                        f"rut: {cand['rut_sev'] if cand['rut_sev'] else 'None'}, "
                        f"crown: {cand['crown'] if cand['crown'] else 'None'}"
                    ) + " }"
                    plots_info.append({
                        "header": f"line index {cand['i']} , x {cand['x']:.3f}",
                        "data_uri": uri,
                        "foot": foot,
                    })

        out_dir = os.path.dirname(out_html)
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        _html_report(totals_ft, counts, plots_info, out_html, (res_x, res_y), spacing_m, neighbor_mode)

        arcpy.AddMessage(f"Linear feet, Low, {totals_ft.get('Low', 0.0):.2f}")
        arcpy.AddMessage(f"Linear feet, Medium, {totals_ft.get('Medium', 0.0):.2f}")
        arcpy.AddMessage(f"Linear feet, High, {totals_ft.get('High', 0.0):.2f}")
        arcpy.AddMessage(f"Report, {out_html}")

        arcpy.SetParameter(6, out_html)
        return
