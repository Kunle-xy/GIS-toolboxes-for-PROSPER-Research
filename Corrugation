import arcpy
import os
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Polygon
import rasterio
from rasterio.transform import rowcol

# Plotly for interactive, hoverable charts embedded in HTML
import plotly.graph_objects as go
import plotly.io as pio

# Conversion factors
M_TO_IN = 39.3701
M2_TO_FT2 = 10.7639

# --- Gaussian smoothing, length safe, no zero padding at edges ---
def smooth_gaussian(x, sigma_samples: float):
    x = np.asarray(x, dtype=float)
    if sigma_samples is None or sigma_samples <= 0 or x.size == 0:
        return x.copy()
    try:
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(x, sigma=float(sigma_samples), mode="nearest", truncate=3.0)
    except Exception:
        # lightweight fallback, mirror ends and convolve with gaussian kernel
        win = int(max(3, round(6.0 * float(sigma_samples))))
        if win % 2 == 0:
            win += 1
        half = win // 2
        if half > 0 and x.size > 1:
            left = x[1:half+1][::-1]
            right = x[-half-1:-1][::-1]
        else:
            left = x[:0]
            right = x[:0]
        ypad = np.concatenate([left, x, right])
        t = np.arange(-half, half + 1, dtype=float)
        sigma = float(sigma_samples)
        k = np.exp(-(t ** 2) / (2.0 * sigma * sigma))
        k /= k.sum()
        y = np.convolve(ypad, k, mode="valid")
        if y.shape[0] != x.shape[0]:
            y = y[:x.shape[0]]
        return y
    
def detect_corrugations_trend(section, depth_m, dx_m, sigma_samples=2.0):
    if len(section) != len(depth_m):
        raise ValueError(f"Input arrays mismatch, section length={len(section)}, depth_m length={len(depth_m)}")

    y = smooth_gaussian(depth_m.astype(float), sigma_samples=sigma_samples)
    assert len(y) == len(depth_m) == len(section), "smoothing must preserve length"
    n = len(y)

    # up/down trend on smoothed signal
    trend = ['UP' if y[i] > y[i - 1] else 'DOWN' for i in range(1, n)]

    sev_str = ["NO CORRUGATION"] * n
    amp_in = np.zeros(n, dtype=float)
    peaks = []    # (idx, value)
    valleys = []  # (idx, value)

    index = 0
    while index < len(trend) - 1:
        if trend[index] == 'UP' and trend[index + 1] == 'DOWN':
            left_peak_idx = index + 1
            peaks.append((left_peak_idx, y[left_peak_idx]))

            # walk DOWN
            j = index + 1
            while j < len(trend) and trend[j] == 'DOWN':
                j += 1
            if j >= len(trend) or trend[j] != 'UP':
                index += 1
                continue

            # walk UP
            k = j
            while k < len(trend) - 1 and trend[k] == 'UP':
                k += 1
            if k >= len(trend) or trend[k] != 'DOWN':
                index += 1
                continue

            right_peak_idx = k
            peaks.append((right_peak_idx, y[right_peak_idx]))

            if right_peak_idx - left_peak_idx > 1:
                # valley between peaks
                valley_region = y[left_peak_idx + 1:right_peak_idx]
                if valley_region.size > 0:
                    valley_rel = int(np.argmin(valley_region))
                    valley_idx = left_peak_idx + 1 + valley_rel
                    valleys.append((valley_idx, y[valley_idx]))

                    if 0 <= left_peak_idx < n and 0 <= right_peak_idx < n and 0 <= valley_idx < n:
                        left_peak = y[left_peak_idx]
                        right_peak = y[right_peak_idx]
                        valley = y[valley_idx]

                        if not (np.isnan(left_peak) or np.isnan(right_peak) or np.isnan(valley)):
                            # Interpolated crest at the valley x-position (works even if right < left)
                            drop_inches = 0.0
                            dx_total = right_peak_idx - left_peak_idx
                            if dx_total > 0:
                                t = (valley_idx - left_peak_idx) / dx_total  # in [0,1]
                                crest_at_valley = left_peak + t * (right_peak - left_peak)
                                depth_diff_m = max(0.0, crest_at_valley - valley)  # clamp negative
                                drop_inches = depth_diff_m * M_TO_IN

                            # symmetry/shape check (min within % of max)
                            max_p = max(left_peak, right_peak)
                            min_p = min(left_peak, right_peak)

                            if max_p == 0:
                                severity = "NO CORRUGATION"
                            else:
                                symmetry_ok = (min_p / max_p) >= 0.10  # 10% balance guard
                                if (not symmetry_ok) or (drop_inches <= 0.0):
                                    severity = "NO CORRUGATION"
                                elif drop_inches < 1.0:
                                    severity = "LOW"
                                elif drop_inches < 3.0:
                                    severity = "MEDIUM"
                                else:
                                    severity = "HIGH"

                            # Write labels & amplitudes consistently across the span
                            for ii in range(left_peak_idx, right_peak_idx + 1):
                                if severity == "NO CORRUGATION":
                                    sev_str[ii] = "NO CORRUGATION"
                                    amp_in[ii] = max(0.0, amp_in[ii])
                                else:
                                    sev_str[ii] = severity
                                    amp_in[ii] = max(amp_in[ii], drop_inches)

            # allow overlap with the next structure a bit
            index = max(right_peak_idx - 1, 0)
        else:
            index += 1

    # --- Post-pass: promote subtle lows without being overly aggressive ---
    sev_arr = np.array(sev_str, dtype=object)
    low_cands = amp_in[(amp_in > 0) & (sev_arr == "NO CORRUGATION")]
    if low_cands.size > 0:
        mu = float(np.mean(low_cands))
        sd = float(np.std(low_cands))
        # gentler threshold and safe when sd ~ 0
        if np.isfinite(sd) and sd > 1e-6:
            thr = max(0.0, mu - 0.5 * sd)
        else:
            thr = max(0.0, 0.5 * mu)

        for i in range(n):
            if sev_str[i] == "NO CORRUGATION" and amp_in[i] > 0:
                if amp_in[i] <= thr:
                    amp_in[i] = 0.0
                else:
                    sev_str[i] = "LOW"

    df = pd.DataFrame({
        "section": section.astype(int),
        "depth_m": depth_m.astype(float),
        "drop_in": amp_in,
        "severity": sev_str
    })
    return df, y, peaks, valleys

# --- HTML helpers ---
def html_escape(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

# interactive profile with hover, peak, and valley arrows
def render_profile_plotly_html(sections, depth_vals, smoothed_vals, severity_labels, peaks, valleys, title_suffix=""):
    sev_order = ["NO CORRUGATION", "LOW", "MEDIUM", "HIGH"]
    sev_to_num = {s: i for i, s in enumerate(sev_order)}
    sev_num = [sev_to_num.get(s, 0) for s in severity_labels]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=sections,
        y=depth_vals,
        mode="lines",
        name="Original",
        opacity=0.35,
        hovertemplate="Section, %{x}<br>Elevation raw, %{y:.6f} m<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=sections,
        y=smoothed_vals,
        mode="lines",
        name="Smoothed",
        line=dict(width=2),
        hovertemplate="Section, %{x}<br>Elevation smoothed, %{y:.6f} m<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=sections,
        y=sev_num,
        mode="lines",
        name="Severity",
        line=dict(width=2, dash="dash"),
        yaxis="y2",
        hovertemplate="Section, %{x}<br>Severity, %{text}<extra></extra>",
        text=severity_labels
    ))

    # Add arrows at peaks
    for peak_idx, peak_val in peaks:
        fig.add_annotation(
            x=sections[peak_idx],
            y=peak_val,
            xref="x",
            yref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="red",
            ax=0,
            ay=-30,  # Position arrow above the peak
            text="Peak",
            font=dict(size=10, color="red"),
            align="center",
            standoff=10
        )

    # Add arrows at valleys
    for valley_idx, valley_val in valleys:
        fig.add_annotation(
            x=sections[valley_idx],
            y=valley_val,
            xref="x",
            yref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="blue",
            ax=0,
            ay=30,  # Position arrow below the valley
            text="Valley",
            font=dict(size=10, color="blue"),
            align="center",
            standoff=10
        )

    fig.update_layout(
        title=f"Depth Profile with Corrugation Severity {title_suffix}",
        template="plotly_white",
        margin=dict(l=60, r=60, t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        xaxis=dict(title="Section index"),
        yaxis=dict(title="Elevation, meters"),
        yaxis2=dict(
            title="Severity, none, low, medium, high",
            overlaying="y",
            side="right",
            range=[-0.5, 3.5],
            tickmode="array",
            tickvals=[0, 1, 2, 3],
            ticktext=["None", "Low", "Medium", "High"]
        )
    )

    return pio.to_html(fig, include_plotlyjs="cdn", full_html=False)

# interactive severity total area band with hover
def render_area_mean_std_band_plotly_html(neighbor_summary_rows):
    sev_order = ["LOW", "MEDIUM", "HIGH"]
    pretty = {"LOW": "Low", "MEDIUM": "Medium", "HIGH": "High"}

    vals = {s: [] for s in sev_order}
    for r in neighbor_summary_rows:
        s = r.get("Severity")
        a = r.get("Total Area, ft²")
        if s in vals and a is not None and np.isfinite(a):
            vals[s].append(float(a))

    means = [float(np.mean(vals[s])) if len(vals[s]) else np.nan for s in sev_order]
    stds  = [float(np.std(vals[s]))  if len(vals[s]) else np.nan for s in sev_order]
    ns    = [int(len(vals[s])) for s in sev_order]

    x = list(range(len(sev_order)))
    fig = go.Figure()

    # bands as error bars around the mean
    fig.add_trace(go.Scatter(
        x=x,
        y=means,
        mode="lines+markers",
        name="Total area",
        error_y=dict(
            type="data",
            array=[s if np.isfinite(s) else 0 for s in stds],
            visible=True
        ),
        hovertemplate=(
            "Severity, %{text}"
            "<br>Mean area, %{y:.2f} ft²"
            "<br>Std, %{customdata:.2f}"
            "<br>n, %{meta}"
            "<extra></extra>"
        ),
        text=[pretty[s] for s in sev_order],
        customdata=stds,
        meta=ns
    ))

    fig.update_layout(
        title="Total area with standard deviation, by severity",
        template="plotly_white",
        margin=dict(l=60, r=60, t=60, b=50),
        xaxis=dict(
            title="Severity",
            tickmode="array",
            tickvals=x,
            ticktext=[pretty[s] for s in sev_order]
        ),
        yaxis=dict(title="Total area, square feet")
    )

    return pio.to_html(fig, include_plotlyjs=False, full_html=False)

# --- Corrugation Tool ---
class CorrugationTool(object):
    def __init__(self):
        self.label = "Corrugation Severity Detector"
        self.description = (
            "Detects and visualizes corrugation severity along sampled lines within a polygon using a DEM, "
            "with optional neighbor frequency study."
        )

    def getParameterInfo(self):
        params = []
        p0 = arcpy.Parameter(
            displayName="Input Polygon Feature",
            name="in_polygon",
            datatype="Feature Layer",
            parameterType="Required",
            direction="Input"
        )
        p1 = arcpy.Parameter(
            displayName="DEM Raster File",
            name="dem_path",
            datatype="DERasterDataset",
            parameterType="Required",
            direction="Input"
        )
        p2 = arcpy.Parameter(
            displayName="Sampling Frequency per 100 pixels",
            name="sampling_frequency",
            datatype="Long",
            parameterType="Required",
            direction="Input"
        )
        p2.value = 10
        p3 = arcpy.Parameter(
            displayName="Line Interval, meters",
            name="line_interval",
            datatype="Double",
            parameterType="Required",
            direction="Input"
        )
        p3.value = 0.1
        p4 = arcpy.Parameter(
            displayName="Try Neighboring Sampling Frequencies",
            name="neighbor_eval",
            datatype="Boolean",
            parameterType="Optional",
            direction="Input"
        )
        p4.value = False
        p5 = arcpy.Parameter(
            displayName="Neighbor Range, plus or minus",
            name="neighbor_range",
            datatype="Long",
            parameterType="Optional",
            direction="Input"
        )
        p5.value = 2
        p6 = arcpy.Parameter(
            displayName="Output Plot, HTML",
            name="output_html",
            datatype="DEFile",
            parameterType="Required",
            direction="Output"
        )
        p6.filter.list = ["html"]
        p7 = arcpy.Parameter(
            displayName="Smooth Sigma, samples",
            name="sigma_samples",
            datatype="Double",
            parameterType="Optional",
            direction="Input"
        )
        p7.value = 2.0
        params.extend([p0, p1, p2, p3, p4, p5, p6, p7])
        return params

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        if parameters[6].valueAsText and not parameters[6].valueAsText.lower().endswith(".html"):
            parameters[6].value = parameters[6].valueAsText + ".html"
        return

    def updateMessages(self, parameters):
        if parameters[0].valueAsText:
            desc = arcpy.Describe(parameters[0].valueAsText)
            if desc.shapeType != "Polygon":
                parameters[0].setErrorMessage("Input feature must be a polygon.")
        if parameters[1].valueAsText:
            dem = parameters[1].valueAsText
            if not os.path.exists(dem):
                parameters[1].setErrorMessage("DEM file does not exist.")
            elif not dem.lower().endswith((".tif", ".tiff", ".img")):
                parameters[1].setWarningMessage("DEM should be .tif, .tiff, or .img.")
        if parameters[2].value is not None and int(parameters[2].value) <= 0:
            parameters[2].setErrorMessage("Sampling frequency per 100 pixels must be greater than zero.")
        if parameters[3].value is not None and float(parameters[3].value) <= 0:
            parameters[3].setErrorMessage("Line interval must be greater than zero.")
        if parameters[5].value is not None and int(parameters[5].value) < 1:
            parameters[5].setWarningMessage("Neighbor range is less than one, it will be clamped to one at run time.")
        if parameters[6].valueAsText:
            outp = parameters[6].valueAsText
            if not outp.lower().endswith(".html"):
                parameters[6].setErrorMessage("Output must end with .html")
            out_dir = os.path.dirname(outp)
            if out_dir and not out_dir == '' and not os.path.exists(out_dir):
                parameters[6].setErrorMessage("Output directory does not exist.")
        if parameters[7].value is not None and float(parameters[7].value) < 0:
            parameters[7].setErrorMessage("Smooth Sigma must be greater than or equal to zero.")
        return

    def execute(self, parameters, messages):
        try:
            arcpy.AddMessage("Starting Corrugation Severity Detection for Polygon...")
            in_polygon = parameters[0].valueAsText
            dem_path = parameters[1].valueAsText
            sampling_frequency = int(parameters[2].valueAsText)
            line_interval = float(parameters[3].valueAsText)
            neighbor_eval = bool(parameters[4].value) if parameters[4].value is not None else False
            neighbor_range = int(parameters[5].value) if parameters[5].value is not None else 2
            neighbor_range = max(1, neighbor_range)
            output_html = parameters[6].valueAsText
            sigma_samples = float(parameters[7].valueAsText) if parameters[7].valueAsText else 2.0

            if neighbor_eval:
                start_f = max(1, sampling_frequency - neighbor_range)
                end_f = sampling_frequency + neighbor_range
                frequencies = list(range(start_f, end_f + 1))
            else:
                frequencies = [sampling_frequency]

            desc = arcpy.Describe(in_polygon)
            if desc.FIDSet:
                in_polygon = arcpy.management.MakeFeatureLayer(in_polygon, "selected_polygons")
            poly_srs = desc.spatialReference

            with rasterio.open(dem_path) as src:
                band1 = src.read(1)
                transform = src.transform
                h_px, w_px = src.height, src.width
                bounds = src.bounds
                dem_crs = src.crs
                nodata = src.nodata
                pixel_size = src.res[0]

            dem_srs = arcpy.SpatialReference()
            dem_srs.loadFromString(str(dem_crs))

            if poly_srs.name != dem_srs.name:
                tmp = "memory/tmp_poly"
                arcpy.management.Project(in_polygon, tmp, dem_srs)
                in_poly_proj = tmp
            else:
                in_poly_proj = in_polygon

            with arcpy.da.SearchCursor(in_poly_proj, ["SHAPE@"]) as cur:
                try:
                    poly_geom = next(cur)[0]
                except StopIteration:
                    raise arcpy.ExecuteError("No polygon features found.")
                coords = [(pt.X, pt.Y) for pt in poly_geom.getPart(0) if pt]
                polygon = Polygon(coords)

            try:
                if poly_srs.name != dem_srs.name and arcpy.Exists(tmp):
                    arcpy.management.Delete(tmp)
            except Exception:
                pass

            minx, miny, maxx, maxy = polygon.bounds
            transect_ys = np.arange(miny, maxy + line_interval, line_interval)

            elevation_plots = []
            all_ruts = []
            neighbor_summary_rows = []
            severity_levels = ["LOW", "MEDIUM", "HIGH"]

            for f in frequencies:
                dx = (100.0 * pixel_size) / float(f)

                freq_ruts = []

                for transect_y in transect_ys:
                    h_line = LineString([(minx - 10, transect_y), (maxx + 10, transect_y)])
                    inter = polygon.intersection(h_line)
                    if inter.is_empty:
                        continue
                    if inter.geom_type == 'MultiLineString':
                        line = max(inter.geometries, key=lambda g: g.length)
                    else:
                        line = inter

                    if line.length < dx:
                        continue

                    length = line.length
                    dists = np.arange(0.0, length + dx, dx)
                    pts = [line.interpolate(d) for d in dists]
                    xs = np.array([p.x for p in pts])
                    ys = np.array([p.y for p in pts])

                    in_bounds = (xs >= bounds.left) & (xs <= bounds.right) & (ys >= bounds.bottom) & (ys <= bounds.top)
                    xs, ys, dists = xs[in_bounds], ys[in_bounds], dists[in_bounds]
                    if xs.size == 0:
                        continue

                    rows, cols = rowcol(transform, xs, ys, op=round)
                    rows = np.asarray(rows, dtype=int)
                    cols = np.asarray(cols, dtype=int)
                    valid = (rows >= 0) & (rows < h_px) & (cols >= 0) & (cols < w_px)
                    rows, cols, dists = rows[valid], cols[valid], dists[valid]

                    depths = band1[rows, cols].astype(float)
                    if nodata is not None:
                        depths = np.where(depths == nodata, np.nan, depths)

                    if depths.size == 0:
                        continue

                    sections = np.arange(len(depths))
                    df_input = pd.DataFrame({"section": sections, "depth_m": depths})
                    df_input["depth_m"] = df_input["depth_m"].interpolate("linear").bfill().ffill()

                    if len(df_input) < 5:
                        continue

                    df_result, y_smooth, peaks, valleys = detect_corrugations_trend(
                        df_input["section"].to_numpy(),
                        df_input["depth_m"].to_numpy(),
                        dx,
                        sigma_samples
                    )

                    df_line = df_result.copy()
                    df_line["position_m"] = dists[df_line["section"]]
                    df_line_nonzero = df_line[(df_line["drop_in"] > 0) | (df_line["severity"].isin(["MEDIUM", "HIGH"]))]

                    line_ruts = []
                    if not df_line_nonzero.empty:
                        seg_break = (df_line_nonzero["severity"].shift() != df_line_nonzero["severity"]).cumsum()
                        for _, group in df_line_nonzero.groupby(seg_break):
                            start_pos = float(group["position_m"].min())
                            depth_in = float(group["drop_in"].max())
                            severity = str(group["severity"].iloc[0])
                            num_sections = int(len(group))
                            length_m = num_sections * dx
                            area_ft2 = float(length_m * line_interval * M2_TO_FT2)
                            line_ruts.append({
                                "Transect Y, m": float(transect_y),
                                "Depth, in": depth_in,
                                "Area, ft²": area_ft2,
                                "Severity": severity,
                                "Frequency": int(f)
                            })

                    elevation_plots.append({
                        "transect_y": float(transect_y),
                        "sections": df_input["section"].tolist(),
                        "depths": df_input["depth_m"].tolist(),
                        "smoothed": y_smooth.tolist(),
                        "severity_labels": df_result["severity"].tolist(),
                        "peaks": peaks,
                        "valleys": valleys
                    })

                    freq_ruts.extend(line_ruts)

                for sev in severity_levels:
                    items = [r for r in freq_ruts if r["Severity"] == sev]
                    depths_arr = np.array([r["Depth, in"] for r in items], dtype=float) if items else np.array([])
                    areas_arr = np.array([r["Area, ft²"] for r in items], dtype=float) if items else np.array([])
                    row = {
                        "Frequency": int(f),
                        "Severity": sev,
                        "Count": int(len(items)),
                        "Mean Depth, in": float(np.nanmean(depths_arr)) if depths_arr.size else np.nan,
                        "Std Depth, in": float(np.nanstd(depths_arr)) if depths_arr.size else np.nan,
                        "Mean Area, ft²": float(np.nanmean(areas_arr)) if areas_arr.size else np.nan,
                        "Std Area, ft²": float(np.nanstd(areas_arr)) if areas_arr.size else np.nan,
                        "Total Area, ft²": float(np.nansum(areas_arr)) if areas_arr.size else 0.0
                    }
                    neighbor_summary_rows.append(row)

                all_ruts.extend(freq_ruts)

            # --- HTML report ---
            html_parts = []
            html_parts.append("<!DOCTYPE html>")
            html_parts.append("<html><head><meta charset='utf-8'><title>Corrugation Severity Report</title>")
            html_parts.append("<style>body{font-family:Arial, sans-serif; margin:24px;} h1,h2,h3{color:#333} table{border-collapse:collapse; width:100%; margin-top:12px} th,td{border:1px solid #ccc; padding:6px 8px; text-align:left} th{background:#f0f8ff} .box{padding:12px; border:1px solid #ddd; border-radius:6px; background:#fcfcff; margin:12px 0;} .pill{display:inline-block; margin-right:10px; padding:6px 10px; border-radius:999px; background:#eef2ff; border:1px solid #d0d7ff}</style>")
            html_parts.append("</head><body>")
            html_parts.append("<h1>Corrugation Severity Report</h1>")

            sev_classes = ["LOW", "MEDIUM", "HIGH"]
            totals = {s: {"count": 0, "area": 0.0} for s in sev_classes}
            for r in all_ruts:
                if r["Severity"] in totals:
                    totals[r["Severity"]]["count"] += 1
                    totals[r["Severity"]]["area"] += float(r["Area, ft²"])
            html_parts.append("<div class='box'><b>Totals across all transects</b><br>")
            for s in sev_classes:
                cnt = totals[s]["count"]
                area = totals[s]["area"]
                html_parts.append(f"<span class='pill'>{s}: segments {cnt}, total area {area:.2f} ft²</span>")
            html_parts.append("</div>")

            # neighbor trial section
            if neighbor_eval:
                html_parts.append("<h2>Neighbor Frequency Summary</h2>")
                if neighbor_summary_rows:
                    cols = ["Frequency", "Severity", "Count", "Mean Depth, in", "Std Depth, in", "Mean Area, ft²", "Std Area, ft²", "Total Area, ft²"]
                    html_parts.append("<table><tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>")
                    for r in neighbor_summary_rows:
                        html_parts.append("<tr>" + "".join(f"<td>{html_escape(r[c])}</td>" for c in cols) + "</tr>")
                    html_parts.append("</table>")
                    # interactive severity plot
                    area_div = render_area_mean_std_band_plotly_html(neighbor_summary_rows)
                    html_parts.append("<h2>Total area with standard deviation, by severity</h2>")
                    html_parts.append(f"<div>{area_div}</div>")
                else:
                    html_parts.append("<p>No corrugations detected in neighbor frequency study.</p>")

            # profiles only when trial is off
            if not neighbor_eval:
                if not elevation_plots:
                    html_parts.append("<h2>No transect profiles to display</h2>")
                else:
                    total_lines = len(elevation_plots)
                    plotted_plots = elevation_plots  # show all, or sample as desired
                    html_parts.append("<h2>Transect Profiles</h2>")
                    # render each as an interactive plotly div
                    for plot in plotted_plots:
                        plotly_div = render_profile_plotly_html(
                            sections=plot["sections"],
                            depth_vals=plot["depths"],
                            smoothed_vals=plot["smoothed"],
                            severity_labels=plot["severity_labels"],
                            peaks=plot["peaks"],
                            valleys=plot["valleys"],
                            title_suffix=f"at Y = {plot['transect_y']:.2f} m"
                        )
                        html_parts.append(f"<h3>Y = {plot['transect_y']:.2f} m</h3>")
                        html_parts.append(f"<div style='border:1px solid #ddd; border-radius:4px; padding:4px;'>{plotly_div}</div>")

            # all segments table only when trial is off
            if not neighbor_eval:
                html_parts.append("<h2>Detected Corrugations, all segments</h2>")
                if all_ruts:
                    html_parts.append("<table><tr><th>Transect Y, m</th><th>Depth, in</th><th>Area, ft²</th><th>Severity</th><th>Frequency</th></tr>")
                    for r in all_ruts:
                        html_parts.append(
                            "<tr>"
                            f"<td>{html_escape(r['Transect Y, m'])}</td>"
                            f"<td>{html_escape(round(r['Depth, in'], 4))}</td>"
                            f"<td>{html_escape(round(r['Area, ft²'], 4))}</td>"
                            f"<td>{html_escape(r['Severity'])}</td>"
                            f"<td>{html_escape(r['Frequency'])}</td>"
                            "</tr>"
                        )
                    html_parts.append("</table>")
                else:
                    html_parts.append("<p>No corrugations detected.</p>")

            html_parts.append("</body></html>")

            with open(output_html, "w", encoding="utf-8") as f:
                f.write("\n".join(html_parts))

            arcpy.AddMessage(f"Report saved to {output_html}")
            arcpy.AddMessage("Corrugation severity detection complete.")

        except Exception as e:
            raise arcpy.ExecuteError(f"Execution failed, {str(e)}")

# --- Toolbox ---
class Toolbox(object):
    def __init__(self):
        self.label = "Corrugation Detection Toolbox"
        self.alias = "corrugation"
        self.tools = [CorrugationTool]

