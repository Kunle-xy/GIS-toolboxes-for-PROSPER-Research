# -*- coding: utf-8 -*-
"""
Loose Aggregate Detector (Polygon)

dx: spacing between vertical lines (meters)
dy: sampling distance along each vertical line (meters)

Peak detection window: length is a function of dy, capped at 3 ft (0.9144 m):
    window_samples = max(1, floor(0.9144 / dy))

Linear feet totals: each vertical line contributes dx meters => dx * 3.28084 feet
Worst severity per line is used for summary lengths and the map overlay (drawn in HTML only; no feature class).
"""

import arcpy
import os
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, LineString
from collections import defaultdict
import rasterio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import plotly.colors

# ---------------- constants ----------------
INCHES_PER_METER = 39.3701
FEET_PER_METER = 3.28084
FT_TO_M = 0.3048
MAX_WINDOW_FT = 3.0
MAX_WINDOW_M = MAX_WINDOW_FT * FT_TO_M

# ranking helper
_SEV_RANK = {"Low": 1, "Medium": 2, "High": 3}
def _worse_severity(s1, s2):
    """Return worse (higher rank) severity among s1 and s2 (strings or None)."""
    r1 = _SEV_RANK.get(s1, 0)
    r2 = _SEV_RANK.get(s2, 0)
    return s1 if r1 >= r2 else s2

class LooseAggregatePolygonTool(object):
    def __init__(self):
        self.label = "Loose Aggregate Detector (Polygon)"
        self.description = (
            "Detects and visualizes loose aggregate severity within a polygon using a DEM. "
            "dx = spacing between vertical lines; dy = sampling distance along each line."
        )

    def getParameterInfo(self):
        params = []

        param0 = arcpy.Parameter(
            displayName="Input Polygon Feature",
            name="in_polygon",
            datatype="Feature Layer",
            parameterType="Required",
            direction="Input")

        param1 = arcpy.Parameter(
            displayName="DEM Raster File",
            name="dem_path",
            datatype="DERasterDataset",
            parameterType="Required",
            direction="Input")

        # dx = spacing between vertical lines
        param2 = arcpy.Parameter(
            displayName="Line spacing (dx) between vertical lines (meters)",
            name="dx",
            datatype="Double",
            parameterType="Required",
            direction="Input")
        param2.value = 0.1

        # dy = sampling distance along each line
        param3 = arcpy.Parameter(
            displayName="Sampling distance (dy) along each line (meters)",
            name="dy",
            datatype="Double",
            parameterType="Required",
            direction="Input")
        param3.value = 0.1

        param4 = arcpy.Parameter(
            displayName="Output Plot (HTML)",
            name="output_html",
            datatype="DEFile",
            parameterType="Required",
            direction="Output")
        param4.filter.list = ["html"]

        params.extend([param0, param1, param2, param3, param4])
        return params

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        if parameters[4].valueAsText:
            output_path = parameters[4].valueAsText
            if not output_path.lower().endswith('.html'):
                parameters[4].value = output_path + '.html'
        return

    def updateMessages(self, parameters):
        if parameters[0].valueAsText:
            desc = arcpy.Describe(parameters[0].valueAsText)
            if desc.shapeType != "Polygon":
                parameters[0].setErrorMessage("Input feature must be a polygon.")

        if parameters[1].valueAsText:
            dem_path = parameters[1].valueAsText
            if not os.path.exists(dem_path):
                parameters[1].setErrorMessage("DEM file does not exist.")
            elif not dem_path.lower().endswith(('.tif', '.tiff', '.img')):
                parameters[1].setWarningMessage("DEM file should be a raster format (e.g., .tif, .img).")

        if parameters[2].value is not None and parameters[2].value <= 0:
            parameters[2].setErrorMessage("Line spacing (dx) must be greater than 0.")

        if parameters[3].value is not None and parameters[3].value <= 0:
            parameters[3].setErrorMessage("Sampling distance (dy) must be greater than 0.")

        if parameters[4].valueAsText:
            output_path = parameters[4].valueAsText
            if not output_path.lower().endswith('.html'):
                parameters[4].setErrorMessage("Output file must have a .html extension.")
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                parameters[4].setErrorMessage("Output directory does not exist.")

        return

    def execute(self, parameters, messages):
        try:
            arcpy.AddMessage("Starting Loose Aggregate Severity Detection for Polygon...")
            arcpy.AddMessage("Semantics: dx = spacing between vertical lines; dy = sampling distance along a line")

            in_polygon = parameters[0].valueAsText
            dem_path = parameters[1].valueAsText
            dx = float(parameters[2].valueAsText)   # spacing between vertical lines
            dy = float(parameters[3].valueAsText)   # sampling along each line
            output_html = parameters[4].valueAsText

            # compute window size (in samples) capped at 3 ft
            window_samples = max(1, int(math.floor(MAX_WINDOW_M / dy)))
            window_len_m = window_samples * dy
            window_len_ft = window_len_m * FEET_PER_METER
            arcpy.AddMessage(
                f"Peak-detection window: {window_samples} samples "
                f"(~{window_len_m:.3f} m / {window_len_ft:.2f} ft; cap = {MAX_WINDOW_FT:.1f} ft)"
            )

            arcpy.AddMessage("Checking polygon selection...")
            desc = arcpy.Describe(in_polygon)
            if desc.FIDSet:
                arcpy.AddMessage("Using selected polygon features only.")
                in_polygon = arcpy.management.MakeFeatureLayer(in_polygon, "selected_polygons")
            else:
                arcpy.AddMessage("No selection detected. Using all polygon features.")

            arcpy.AddMessage("Extracting geometry from polygon feature...")
            selected_polygons = []
            polygon_srs = desc.spatialReference
            arcpy.AddMessage(f"Polygon spatial reference: {polygon_srs.name}")
            with arcpy.da.SearchCursor(in_polygon, ["SHAPE@"]) as cursor:
                for row in cursor:
                    selected_polygons.append(row[0])

            if len(selected_polygons) > 1:
                arcpy.AddWarning("Multiple polygons detected. Processing only the first selected polygon.")
            if not selected_polygons:
                raise arcpy.ExecuteError("No polygon features found.")

            poly_geom = selected_polygons[0]

            arcpy.AddMessage("Loading DEM and checking spatial reference...")
            with rasterio.open(dem_path) as src:
                band1 = src.read(1)
                bounds = src.bounds
                dem_crs = src.crs
                dem_srs = arcpy.SpatialReference()
                dem_srs.loadFromString(str(dem_crs))
                arcpy.AddMessage(f"DEM spatial reference: {dem_srs.name}")
                arcpy.AddMessage(f"DEM resolution: x={src.res[0]:.4f} m, y={src.res[1]:.4f} m")
                arcpy.AddMessage(f"DEM bounds: left={bounds.left:.4f}, bottom={bounds.bottom:.4f}, right={bounds.right:.4f}, top={bounds.top:.4f}")
                arcpy.AddMessage(f"Elevation range in DEM (meters): Min={np.nanmin(band1):.4f}, Max={np.nanmax(band1):.4f}")

                # Convert polygon to shapely for intersection checks
                coords = [(pt.X, pt.Y) for part in poly_geom for pt in part if pt]
                polygon = Polygon(coords)
                poly_bounds = polygon.bounds
                arcpy.AddMessage(
                    f"Polygon bounds: left={poly_bounds[0]:.4f}, bottom={poly_bounds[1]:.4f}, right={poly_bounds[2]:.4f}, top={poly_bounds[3]:.4f}"
                )
                if not (poly_bounds[0] <= bounds.right and poly_bounds[2] >= bounds.left and
                        poly_bounds[1] <= bounds.top and poly_bounds[3] >= bounds.bottom):
                    arcpy.AddWarning("Polygon does not intersect DEM extent. Check spatial alignment.")

                # Reproject polygon if needed
                if polygon_srs.name != dem_srs.name:
                    arcpy.AddWarning("Spatial reference mismatch between polygon and DEM. Reprojecting polygon to DEM's spatial reference.")
                    temp_poly = "memory/temp_poly"
                    arcpy.management.Project(in_polygon, temp_poly, dem_srs)
                    with arcpy.da.SearchCursor(temp_poly, ["SHAPE@"]) as cursor2:
                        poly_geom = next(cursor2)[0]
                    arcpy.management.Delete(temp_poly)
                    coords = [(pt.X, pt.Y) for part in poly_geom for pt in part if pt]
                    polygon = Polygon(coords)

                min_x, min_y, max_x, max_y = polygon.bounds

                # Build x positions spaced by dx (line spacing)
                x_steps = np.arange(min_x, max_x + dx, dx)

                all_results = defaultdict(list)
                line_data = []
                elevation_plots = []
                line_severities = {}  # worst severity per line

                arcpy.AddMessage(f"Processing {len(x_steps)} vertical lines across polygon...")
                for line_idx, x in enumerate(x_steps):
                    arcpy.AddMessage(f"Line {line_idx}: vertical line at x={x:.4f}")
                    line_coords = [(x, min_y), (x, max_y)]
                    line = LineString(line_coords)
                    clipped_line = line.intersection(polygon)
                    if clipped_line.is_empty or clipped_line.length <= 0:
                        continue

                    length = clipped_line.length

                    # sample along the line every dy meters
                    distances = np.arange(0, length, dy)
                    if len(distances) < 2:
                        arcpy.AddMessage(f"Line {line_idx} at x={x:.4f}: Skipped - too few points to sample.")
                        continue

                    # sample elevations
                    elevations = []
                    for d in distances:
                        try:
                            p = clipped_line.interpolate(d)
                            px, py = p.x, p.y
                            if not (bounds.left <= px <= bounds.right and bounds.bottom <= py <= bounds.top):
                                elevations.append(np.nan)
                                continue
                            r, c = src.index(px, py)
                            val = band1[r, c]
                            if val == -32767:
                                val = np.nan
                            elevations.append(float(val))
                        except Exception:
                            elevations.append(np.nan)

                    arcpy.AddMessage(f"Line {line_idx} x={x:.4f}: sampled {len(elevations)} pts, NaN={int(np.isnan(elevations).sum())}")

                    if len(elevations) < 2 or np.isfinite(np.array(elevations, dtype=float)).sum() < 2:
                        arcpy.AddMessage(f"Line {line_idx} x={x:.4f}: Skipped - insufficient valid elevation data.")
                        continue

                    df_input = pd.DataFrame({"section": range(len(elevations)), "elevation": elevations}).dropna()
                    arcpy.AddMessage(f"Line {line_idx} x={x:.4f}: valid points after dropna = {len(df_input)}")
                    arcpy.AddMessage(
                        f"Line {line_idx} x={x:.4f}: elevation range m: "
                        f"min={df_input['elevation'].min():.4f}, max={df_input['elevation'].max():.4f}"
                    )

                    # Detection of aggregates
                    res = []
                    min_height_diff_m = 0.005  # meters
                    direction_prev = None
                    low_left_idx = None
                    low_left_elev = None
                    low_right_idx = None
                    low_right_elev = None
                    peak_idx = None
                    peak_elev = None

                    # Initialize low_left as first DOWN->UP transition, else fallback to start
                    for idx in range(1, len(df_input) - 1):
                        cur = df_input["elevation"].iloc[idx]
                        prev = df_input["elevation"].iloc[idx - 1]
                        nxt = df_input["elevation"].iloc[idx + 1]
                        dir_prev = "DOWN" if cur < prev else "UP" if cur > prev else "FLAT"
                        dir_next = "UP" if nxt > cur else "DOWN" if nxt < cur else "FLAT"
                        if dir_prev == "DOWN" and dir_next == "UP":
                            low_left_idx = idx
                            low_left_elev = cur
                            arcpy.AddMessage(f"Line {line_idx}: initial low at idx={low_left_idx}, elev={low_left_elev:.4f}")
                            break
                    if low_left_idx is None:
                        low_left_idx = 0
                        low_left_elev = df_input["elevation"].iloc[0]
                        arcpy.AddMessage(f"Line {line_idx}: fallback low at idx=0, elev={low_left_elev:.4f}")

                    # scan forward
                    for idx in range(low_left_idx + 1, len(df_input)):
                        current = df_input["elevation"].iloc[idx]
                        previous = df_input["elevation"].iloc[idx - 1]
                        direction = "UP" if current > previous else "DOWN" if current < previous else "FLAT"

                        # detect the next low (right low)
                        if low_left_idx is not None:
                            if (direction_prev in ("DOWN", "FLAT")) and direction == "UP" and (idx - 1) > low_left_idx:
                                low_right_idx = idx - 1
                                low_right_elev = previous
                            elif idx == len(df_input) - 1:
                                low_right_idx = idx
                                low_right_elev = current

                        # detect a peak within a sliding window of window_samples
                        if idx >= window_samples:
                            window_start = max(0, idx - window_samples)
                            window_elev = df_input["elevation"].iloc[window_start: idx + 1]
                            rel_pos = int(np.argmax(window_elev.values))
                            max_elev = float(window_elev.iloc[rel_pos])
                            max_elev_idx = window_start + rel_pos

                            # choose peak if it's at current position or we just reversed
                            if max_elev_idx == idx and max_elev > previous and (max_elev >= current or direction == "DOWN"):
                                peak_idx = max_elev_idx
                                peak_elev = max_elev
                            elif direction_prev == "UP" and direction == "DOWN":
                                peak_idx = idx - 1
                                peak_elev = previous
                            elif direction_prev == "UP" and direction == "FLAT" and peak_idx is None:
                                peak_idx = idx - 1
                                peak_elev = previous

                        # when we have both lows and a peak between them, record aggregate
                        if (low_right_idx is not None) and (peak_idx is not None) and (low_left_idx <= peak_idx <= low_right_idx):
                            left_h = peak_elev - low_left_elev
                            right_h = peak_elev - low_right_elev
                            agg_height = max(left_h, right_h)
                            agg_width_m = (low_right_idx - low_left_idx) * dy  # along-line width in meters
                            agg_height_in = agg_height * INCHES_PER_METER

                            if agg_height >= min_height_diff_m:
                                res.append([agg_width_m, low_left_idx, peak_idx, low_right_idx])

                            # advance the left low for next search
                            low_left_idx = low_right_idx
                            low_left_elev = low_right_elev
                            low_right_idx = None
                            low_right_elev = None
                            peak_idx = None
                            peak_elev = None

                        direction_prev = direction

                    arcpy.AddMessage(f"Line {line_idx} x={x:.4f}: detected {len(res)} loose-aggregate features")

                    # record elevation plot input
                    if res:
                        elevation_plots.append({
                            "line_x": round(x, 4),
                            "sections": df_input["section"].tolist(),
                            "elevations": df_input["elevation"].tolist(),
                            "aggregates": res
                        })

                    # sectionize by std of section indices
                    section_indices = df_input["section"].values
                    mean_idx = float(np.mean(section_indices)) if len(section_indices) > 0 else 0.0
                    std_dev = float(np.std(section_indices)) if len(section_indices) > 1 else 0.0

                    section_aggregates = defaultdict(list)
                    for agg in res:
                        agg_width_m, low_left_idx, peak_idx, low_right_idx = agg
                        section_idx = peak_idx
                        if section_idx < (mean_idx - std_dev):
                            section_aggregates["left"].append(agg)
                        elif (mean_idx - std_dev) <= section_idx <= (mean_idx + std_dev):
                            section_aggregates["center"].append(agg)
                        else:
                            section_aggregates["right"].append(agg)

                    # compute severities per aggregate; track worst per line
                    line_worst = None  # worst severity name
                    line_data_for_line = []

                    for section_name, aggregates in section_aggregates.items():
                        for agg in aggregates:
                            agg_width_m, low_left_idx, peak_idx, low_right_idx = agg
                            low_left_elev = df_input["elevation"].iloc[low_left_idx]
                            peak_elev = df_input["elevation"].iloc[peak_idx]
                            low_right_elev = df_input["elevation"].iloc[low_right_idx]
                            left_h = peak_elev - low_left_elev
                            right_h = peak_elev - low_right_elev
                            agg_height_m = max(left_h, right_h)
                            agg_height_in = agg_height_m * INCHES_PER_METER

                            # severity thresholds in inches (example)
                            severity = "Low" if agg_height_in < 2.0 else ("Medium" if agg_height_in <= 4.0 else "High")

                            agg_info = {
                                "height_in": round(agg_height_in, 4),
                                "linear_extent_ft": dx * FEET_PER_METER,  # each line contributes dx feet to totals
                                "line_x": round(x, 4),
                                "section": section_name,
                                "severity": severity,
                                "low_left_idx": int(low_left_idx),
                                "peak_idx": int(peak_idx),
                                "low_right_idx": int(low_right_idx)
                            }
                            line_data_for_line.append(agg_info)
                            all_results[severity].append(agg_info)
                            line_worst = _worse_severity(line_worst, severity)

                    line_data.extend(line_data_for_line)
                    if line_worst:
                        line_severities[x] = line_worst

                # ---------------- visualization & report (no feature class created) ----------------

                arcpy.AddMessage("Generating visualization (table + elevation plots + severity map)...")
                arcpy.AddMessage(f"Total aggregates in table: {len(line_data)}")

                # Compute total severity lengths *by worst severity per line*
                severity_lengths = {"Low": 0.0, "Medium": 0.0, "High": 0.0}
                severity_counts = {"Low": 0, "Medium": 0, "High": 0}
                for x, sev in line_severities.items():
                    if sev in severity_lengths:
                        severity_lengths[sev] += dx * FEET_PER_METER
                        severity_counts[sev] += 1

                # Build figure
                num_plots = len(elevation_plots)
                if num_plots == 0:
                    fig = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=("Details of All Loose Aggregates", "Severity Map"),
                        specs=[[{'type': 'table'}, {'type': 'xy'}]],
                        column_widths=[0.7, 0.3]
                    )
                    total_height = 600
                    table_row = 1
                    table_col = 1
                    map_row = 1
                    map_col = 2
                    annotation_x = 0.85
                    annotation_y = 0.5
                    cols_per_row = 1  # not used when num_plots==0
                else:
                    cols_per_row = 3
                    num_rows = math.ceil(num_plots / cols_per_row)
                    total_rows = num_rows + 1
                    max_spacing = 1 / (total_rows - 1) if total_rows > 1 else 0.05
                    vertical_spacing = min(0.02, max_spacing)
                    specs = [[{'type': 'xy'} if (row * cols_per_row + col) < num_plots else None
                              for col in range(cols_per_row)] for row in range(num_rows)]
                    specs.append([{'type': 'table', 'colspan': 2}, None, {'type': 'xy'}])
                    row_heights = [0.7 / num_rows] * num_rows + [0.3]
                    subplot_titles = [f"Elevation at x={plot['line_x']} m" for plot in elevation_plots] + ["Details of All Loose Aggregates", "Severity Map"]
                    fig = make_subplots(
                        rows=total_rows,
                        cols=cols_per_row,
                        subplot_titles=subplot_titles,
                        row_heights=row_heights,
                        specs=specs,
                        vertical_spacing=vertical_spacing,
                        column_widths=[0.333, 0.333, 0.333]
                    )
                    total_height = 400 * total_rows
                    table_row = total_rows
                    table_col = 1
                    map_row = total_rows
                    map_col = 3
                    annotation_x = 0.85
                    annotation_y = (1 - sum(row_heights[:-1])) / 2

                colors = plotly.colors.qualitative.Plotly
                num_colors = len(colors)

                # elevation panels
                for i, plot in enumerate(elevation_plots):
                    row = (i // cols_per_row) + 1
                    col = (i % cols_per_row) + 1
                    line_color = colors[i % num_colors]
                    fig.add_trace(go.Scatter(
                        x=plot["sections"],
                        y=plot["elevations"],
                        mode='lines',
                        name=f'Elevation at x={plot["line_x"]} m',
                        line=dict(color=line_color)
                    ), row=row, col=col)

                    # overlay aggregates (as vertical markers at peak index)
                    for agg_info in line_data:
                        if agg_info["line_x"] != plot["line_x"]:
                            continue
                        peak_idx = agg_info["peak_idx"]
                        severity = agg_info["severity"]
                        color = 'green' if severity == "Low" else ('orange' if severity == "Medium" else 'red')
                        fig.add_shape(
                            type="line",
                            x0=peak_idx, x1=peak_idx,
                            y0=min(plot["elevations"]), y1=max(plot["elevations"]),
                            line=dict(color=color, width=2, dash="dash"),
                            row=row, col=col
                        )
                        fig.add_annotation(
                            x=peak_idx,
                            y=max(plot["elevations"]),
                            text=f"{severity}",
                            showarrow=True,
                            arrowhead=1,
                            arrowsize=1,
                            arrowwidth=1,
                            arrowcolor=color,
                            ax=0,
                            ay=-30,
                            font=dict(size=10, color=color),
                            row=row, col=col
                        )

                # severity map (filled rectangles per line worst severity) — in HTML only
                if line_severities:
                    for x, severity in line_severities.items():
                        shp_rect = Polygon([
                            (x - dx/2.0, min_y),
                            (x + dx/2.0, min_y),
                            (x + dx/2.0, max_y),
                            (x - dx/2.0, max_y)
                        ]).intersection(polygon)
                        if shp_rect.is_empty:
                            continue
                        try:
                            pts = list(shp_rect.exterior.coords)[:-1]
                        except Exception:
                            continue
                        color = 'green' if severity == "Low" else ('orange' if severity == "Medium" else 'red')
                        fig.add_trace(go.Scatter(
                            x=[pt[0] for pt in pts],
                            y=[pt[1] for pt in pts],
                            mode='lines',
                            line=dict(color='black', width=1),
                            fill='toself',
                            fillcolor=color,
                            opacity=0.5,
                            name=f'Severity {severity}',
                            hovertemplate=f'Severity: {severity}<br>X: %{{x:.4f}} m<br>Y: %{{y:.4f}} m<extra></extra>'
                        ), row=map_row, col=map_col)

                    fig.update_xaxes(title_text="X (m)", row=map_row, col=map_col, title_font=dict(size=10))
                    fig.update_yaxes(title_text="Y (m)", row=map_row, col=map_col, title_font=dict(size=10),
                                     scaleanchor=f"x{map_col}", scaleratio=1)

                # table
                table_data = {
                    "Line X (m)": [item["line_x"] for item in line_data],
                    "Height (in)": [item["height_in"] for item in line_data],
                    "Height (ft)": [item["height_in"] / 12.0 for item in line_data],
                    "Linear Extent (ft)": [item["linear_extent_ft"] for item in line_data],
                    "Section": [item["section"].capitalize() for item in line_data],
                    "Severity": [item["severity"] for item in line_data]
                }

                if not line_data:
                    fig.add_annotation(
                        x=0.35, y=0.5, xref="paper", yref="paper",
                        text="No loose aggregates detected. Insufficient elevation variation in the DEM.",
                        showarrow=False, font=dict(size=12, color="red"),
                        align="center"
                    )

                fig.add_trace(go.Table(
                    header=dict(
                        values=["Line X (m)", "Height (in)", "Height (ft)", "Linear Extent (ft)", "Section", "Severity"],
                        fill_color='paleturquoise',
                        align='left',
                        font=dict(size=12)
                    ),
                    cells=dict(
                        values=[
                            table_data["Line X (m)"],
                            table_data["Height (in)"],
                            table_data["Height (ft)"],
                            table_data["Linear Extent (ft)"],
                            table_data["Section"],
                            table_data["Severity"]
                        ],
                        fill_color='lavender',
                        align='left',
                        font=dict(size=11)
                    )
                ), row=table_row, col=table_col)

                # axis titles for elevation plots
                for i in range(num_plots):
                    row = (i // cols_per_row) + 1
                    col = (i % cols_per_row) + 1
                    fig.update_yaxes(title_text="Elevation (m)", row=row, col=col, title_font=dict(size=10), automargin=True)
                    fig.update_xaxes(title_text="Section Index", row=row, col=col, title_font=dict(size=10))

                # annotation with totals (worst per line)
                length_text_block = (
                    f"Total Severity Lengths (worst per line)\n"
                    f"Low: {severity_lengths['Low']:.2f} ft ({severity_counts['Low']} lines)\n"
                    f"Medium: {severity_lengths['Medium']:.2f} ft ({severity_counts['Medium']} lines)\n"
                    f"High: {severity_lengths['High']:.2f} ft ({severity_counts['High']} lines)"
                )
                fig.add_annotation(
                    x=0.85, y=0.5 if num_plots == 0 else (1 - (0.3 / 2)),  # place near the map area
                    xref="paper", yref="paper",
                    text=length_text_block,
                    showarrow=False, align="center", bgcolor="white", bordercolor="black", borderwidth=1,
                    font=dict(size=14, color="black"),
                    xanchor="center", yanchor="middle"
                )

                fig.update_layout(
                    title="Loose Aggregate Severity Analysis Across Polygon (Linear Feet; worst severity per line)",
                    height=600 if num_plots == 0 else 400 * (math.ceil(num_plots / 3) + 1),
                    showlegend=True,
                    template="simple_white",
                    margin=dict(l=60, r=100, t=100, b=100)
                )

                fig.write_html(output_html)
                arcpy.AddMessage(f"Enhanced plot saved to: {output_html}")
                arcpy.AddMessage("Loose aggregate severity detection for polygon complete.")

        except Exception as e:
            raise arcpy.ExecuteError(f"Execution failed: {str(e)}")


class Toolbox(object):
    def __init__(self):
        self.label = "TESTING TOOLBOX"
        self.alias = "testing_toolbox"
        self.tools = [LooseAggregatePolygonTool]


class ToolValidator(object):
    def __init__(self):
        self.params = None

    def initializeParameters(self, parameters):
        self.params = parameters
        return

    def updateParameters(self, parameters):
        if parameters[4].valueAsText:
            output_path = parameters[4].valueAsText
            if not output_path.lower().endswith('.html'):
                parameters[4].value = output_path + '.html'
        return

    def updateMessages(self, parameters):
        if parameters[0].valueAsText:
            desc = arcpy.Describe(parameters[0].valueAsText)
            if desc.shapeType != "Polygon":
                parameters[0].setErrorMessage("Input feature must be a polygon.")

        if parameters[1].valueAsText:
            dem_path = parameters[1].valueAsText
            if not os.path.exists(dem_path):
                parameters[1].setErrorMessage("DEM file does not exist.")
            elif not dem_path.lower().endswith(('.tif', '.tiff', '.img')):
                parameters[1].setWarningMessage("DEM file should be a raster format (e.g., .tif, .img).")

        if parameters[2].value is not None and parameters[2].value <= 0:
            parameters[2].setErrorMessage("Line spacing (dx) must be greater than 0.")

        if parameters[3].value is not None and parameters[3].value <= 0:
            parameters[3].setErrorMessage("Sampling distance (dy) must be greater than 0.")

        if parameters[4].valueAsText:
            output_path = parameters[4].valueAsText
            if not output_path.lower().endswith('.html'):
                parameters[4].value = output_path + '.html'
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                parameters[4].setErrorMessage("Output directory does not exist.")

        return





# import arcpy
# import os
# import numpy as np
# import pandas as pd
# from shapely.geometry import Polygon, LineString
# from collections import defaultdict
# import rasterio
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import math
# import plotly.colors

# class LooseAggregatePolygonTool(object):
#     def __init__(self):
#         self.label = "Loose Aggregate Detector (Polygon)"
#         self.description = "Detects and visualizes loose aggregate severity within a polygon using a DEM, with sectional elevation plots in linear feet and polygonal overlay."

#     def getParameterInfo(self):
#         params = []

#         param0 = arcpy.Parameter(
#             displayName="Input Polygon Feature",
#             name="in_polygon",
#             datatype="Feature Layer",
#             parameterType="Required",
#             direction="Input")

#         param1 = arcpy.Parameter(
#             displayName="DEM Raster File",
#             name="dem_path",
#             datatype="DERasterDataset",
#             parameterType="Required",
#             direction="Input")

#         param2 = arcpy.Parameter(
#             displayName="Sampling Distance (dx in meters)",
#             name="dx",
#             datatype="Double",
#             parameterType="Required",
#             direction="Input")
#         param2.value = 0.1

#         param3 = arcpy.Parameter(
#             displayName="Aggregate Width (dy in meters)",
#             name="dy",
#             datatype="Double",
#             parameterType="Required",
#             direction="Input")
#         param3.value = 0.1

#         param4 = arcpy.Parameter(
#             displayName="Output Plot (HTML)",
#             name="output_html",
#             datatype="DEFile",
#             parameterType="Required",
#             direction="Output")
#         param4.filter.list = ["html"]

#         param5 = arcpy.Parameter(
#             displayName="Output Polygon Feature Class",
#             name="output_polygons",
#             datatype="DEFeatureClass",
#             parameterType="Required",
#             direction="Output")

#         params.extend([param0, param1, param2, param3, param4, param5])
#         return params

#     def isLicensed(self):
#         return True

#     def updateParameters(self, parameters):
#         if parameters[4].valueAsText:
#             output_path = parameters[4].valueAsText
#             if not output_path.lower().endswith('.html'):
#                 parameters[4].value = output_path + '.html'
#         return

#     def updateMessages(self, parameters):
#         if parameters[0].valueAsText:
#             desc = arcpy.Describe(parameters[0].valueAsText)
#             if desc.shapeType != "Polygon":
#                 parameters[0].setErrorMessage("Input feature must be a polygon.")

#         if parameters[1].valueAsText:
#             dem_path = parameters[1].valueAsText
#             if not os.path.exists(dem_path):
#                 parameters[1].setErrorMessage("DEM file does not exist.")
#             elif not dem_path.lower().endswith(('.tif', '.tiff', '.img')):
#                 parameters[1].setWarningMessage("DEM file should be a raster format (e.g., .tif, .img).")

#         if parameters[2].value is not None and parameters[2].value <= 0:
#             parameters[2].setErrorMessage("Sampling distance must be greater than 0.")

#         if parameters[3].value is not None and parameters[3].value <= 0:
#             parameters[3].setErrorMessage("Aggregate width must be greater than 0.")

#         if parameters[4].valueAsText:
#             output_path = parameters[4].valueAsText
#             if not output_path.lower().endswith('.html'):
#                 parameters[4].setErrorMessage("Output file must have a .html extension.")
#             output_dir = os.path.dirname(output_path)
#             if output_dir and not os.path.exists(output_dir):
#                 parameters[4].setErrorMessage("Output directory does not exist.")

#         if parameters[5].valueAsText:
#             output_fc = parameters[5].valueAsText
#             output_dir = os.path.dirname(output_fc)
#             if output_dir and not os.path.exists(output_dir):
#                 parameters[5].setErrorMessage("Output feature class directory does not exist.")

#         return

#     def execute(self, parameters, messages):
#         try:
#             arcpy.AddMessage("Starting Loose Aggregate Severity Detection for Polygon...")

#             in_polygon = parameters[0].valueAsText
#             dem_path = parameters[1].valueAsText
#             dx = float(parameters[2].valueAsText)
#             dy = float(parameters[3].valueAsText)
#             output_html = parameters[4].valueAsText
#             output_polygons = parameters[5].valueAsText

#             arcpy.AddMessage("Checking polygon selection...")
#             desc = arcpy.Describe(in_polygon)
#             if desc.FIDSet:
#                 arcpy.AddMessage("Using selected polygon features only.")
#                 in_polygon = arcpy.management.MakeFeatureLayer(in_polygon, "selected_polygons")
#             else:
#                 arcpy.AddMessage("No selection detected. Using all polygon features.")

#             arcpy.AddMessage("Extracting geometry from polygon feature...")
#             selected_polygons = []
#             polygon_srs = desc.spatialReference
#             arcpy.AddMessage(f"Polygon spatial reference: {polygon_srs.name}")
#             with arcpy.da.SearchCursor(in_polygon, ["SHAPE@"]) as cursor:
#                 for row in cursor:
#                     selected_polygons.append(row[0])

#             if len(selected_polygons) > 1:
#                 arcpy.AddWarning("Multiple polygons detected. Processing only the first selected polygon.")
#             if not selected_polygons:
#                 raise arcpy.ExecuteError("No polygon features found.")

#             poly_geom = selected_polygons[0]

#             arcpy.AddMessage("Loading DEM and checking spatial reference...")
#             try:
#                 with rasterio.open(dem_path) as src:
#                     band1 = src.read(1)
#                     bounds = src.bounds
#                     dem_crs = src.crs
#                     dem_srs = arcpy.SpatialReference()
#                     dem_srs.loadFromString(str(dem_crs))
#                     arcpy.AddMessage(f"DEM spatial reference: {dem_srs.name}")
#                     arcpy.AddMessage(f"DEM resolution: x={src.res[0]:.4f} m, y={src.res[1]:.4f} m")
#                     arcpy.AddMessage(f"DEM bounds: left={bounds.left:.4f}, bottom={bounds.bottom:.4f}, right={bounds.right:.4f}, top={bounds.top:.4f}")
#                     arcpy.AddMessage(f"Elevation range in DEM (meters): Min={np.nanmin(band1):.4f}, Max={np.nanmax(band1):.4f}")

#                     # Check if polygon intersects DEM extent
#                     coords = [(pt.X, pt.Y) for part in poly_geom for pt in part if pt]
#                     polygon = Polygon(coords)
#                     poly_bounds = polygon.bounds
#                     arcpy.AddMessage(f"Polygon bounds: left={poly_bounds[0]:.4f}, bottom={poly_bounds[1]:.4f}, right={poly_bounds[2]:.4f}, top={poly_bounds[3]:.4f}")
#                     if not (poly_bounds[0] <= bounds.right and poly_bounds[2] >= bounds.left and
#                             poly_bounds[1] <= bounds.top and poly_bounds[3] >= bounds.bottom):
#                         arcpy.AddWarning("Polygon does not intersect DEM extent. Check spatial alignment.")

#                     # Check if spatial references match
#                     if polygon_srs.name != dem_srs.name:
#                         arcpy.AddWarning("Spatial reference mismatch between polygon and DEM. Reprojecting polygon to DEM's spatial reference.")
#                         temp_poly = "memory/temp_poly"
#                         arcpy.management.Project(in_polygon, temp_poly, dem_srs)
#                         with arcpy.da.SearchCursor(temp_poly, ["SHAPE@"]) as cursor:
#                             poly_geom = next(cursor)[0]
#                         arcpy.management.Delete(temp_poly)
#                         coords = [(pt.X, pt.Y) for part in poly_geom for pt in part if pt]
#                         polygon = Polygon(coords)

#                     min_x, min_y, max_x, max_y = polygon.bounds
#                     x_steps = np.arange(min_x, max_x + dy, dy)
#                     all_results = defaultdict(list)
#                     line_data = []
#                     elevation_plots = []
#                     severity_continuity = defaultdict(list)
#                     line_severities = {}  # To store highest severity per line

#                     arcpy.AddMessage(f"Processing {len(x_steps)} vertical lines across polygon...")
#                     for line_idx, x in enumerate(x_steps):
#                         arcpy.AddMessage(f"Line {line_idx}: Processing vertical line at x={x:.4f}")
#                         line_coords = [(x, min_y), (x, max_y)]
#                         line = LineString(line_coords)
#                         clipped_line = line.intersection(polygon)
#                         if not clipped_line.is_empty and clipped_line.length > 0:
#                             length = clipped_line.length
#                             distances = np.arange(0, length, dx)
#                             if len(distances) < 2:
#                                 arcpy.AddMessage(f"Line {line_idx} at x={x:.4f}: Skipped - too few points to sample.")
#                                 continue
#                             sample_points = [clipped_line.interpolate(distance) for distance in distances]
#                             sample_points = [pt for pt in sample_points if bounds.left <= pt.x <= bounds.right and bounds.bottom <= pt.y <= bounds.top]

#                             elevations = []
#                             for pt in sample_points:
#                                 try:
#                                     row, col = src.index(pt.x, pt.y)
#                                     elev = band1[row, col]
#                                     if elev == -32767:
#                                         elev = np.nan
#                                 except IndexError:
#                                     elev = np.nan
#                                 elevations.append(elev)

#                             arcpy.AddMessage(f"Line {line_idx} at x={x:.4f}: Sampled {len(elevations)} points, NaN count={sum(np.isnan(elevations))}.")

#                             if len(elevations) < 2 or all(np.isnan(elevations)):
#                                 arcpy.AddMessage(f"Line {line_idx} at x={x:.4f}: Skipped - insufficient valid elevation data.")
#                                 continue

#                             df_input = pd.DataFrame({"section": range(len(elevations)), "elevation": elevations}).dropna()
#                             arcpy.AddMessage(f"Line {line_idx} at x={x:.4f}: Valid points after NaN removal: {len(df_input)}")
#                             arcpy.AddMessage(f"Line {line_idx} at x={x:.4f}: Elevation range (meters): Min={df_input['elevation'].min():.4f}, Max={df_input['elevation'].max():.4f}")

#                             # Log elevation profile for all lines
#                             arcpy.AddMessage(f"Line {line_idx} elevation profile:")
#                             for idx, row in df_input.iterrows():
#                                 arcpy.AddMessage(f"Section {int(row['section'])}: Elevation={row['elevation']:.4f}")

#                             res = []
#                             min_height_diff = 0.005  # Minimum height difference in meters
#                             inches_per_meter = 39.3701
#                             window_size = dy * 15  # Window size for detecting wider peaks

#                             if len(df_input) >= 2:
#                                 direction_prev = None
#                                 low_left_idx = None
#                                 low_left_elev = None
#                                 low_right_idx = None
#                                 low_right_elev = None
#                                 peak_idx = None
#                                 peak_elev = None

#                                 # Initialize low_left_idx based on the first DOWN then UP transition
#                                 for idx in range(1, len(df_input) - 1):
#                                     current = df_input["elevation"].iloc[idx]
#                                     previous = df_input["elevation"].iloc[idx - 1]
#                                     next_val = df_input["elevation"].iloc[idx + 1]
#                                     direction_prev = "DOWN" if current < previous else "UP" if current > previous else "FLAT"
#                                     direction_next = "UP" if next_val > current else "DOWN" if next_val < current else "FLAT"
#                                     if direction_prev == "DOWN" and direction_next == "UP":
#                                         low_left_idx = idx
#                                         low_left_elev = current
#                                         arcpy.AddMessage(f"Line {line_idx}: Initial low point detected at idx={low_left_idx}, elev={low_left_elev:.4f} based on DOWN then UP transition.")
#                                         break

#                                 # Fallback to index 0 if no DOWN then UP transition is found
#                                 if low_left_idx is None:
#                                     low_left_idx = 0
#                                     low_left_elev = df_input["elevation"].iloc[0]
#                                     arcpy.AddMessage(f"Line {line_idx}: No DOWN then UP transition found. Using first point at idx={low_left_idx}, elev={low_left_elev:.4f} as fallback.")

#                                 # Continue with the rest of the detection logic
#                                 for idx in range(low_left_idx + 1, len(df_input)):
#                                     current = df_input["elevation"].iloc[idx]
#                                     previous = df_input["elevation"].iloc[idx - 1]
#                                     direction = "UP" if current > previous else "DOWN" if current < previous else "FLAT"
#                                     arcpy.AddMessage(f"Line {line_idx} at idx={idx}: Current={current:.4f}, Previous={previous:.4f}, Direction={direction}")

#                                     # Detect subsequent low points (down or flat to up, or end point)
#                                     if low_left_idx is not None:
#                                         if (direction_prev == "DOWN" or direction_prev == "FLAT") and direction == "UP" and idx - 1 > low_left_idx:
#                                             low_right_idx = idx - 1
#                                             low_right_elev = previous
#                                             arcpy.AddMessage(f"Line {line_idx} at idx={idx}: Low point detected at idx={low_right_idx}, elev={low_right_elev:.4f}")
#                                         elif idx == len(df_input) - 1:
#                                             low_right_idx = idx
#                                             low_right_elev = current
#                                             arcpy.AddMessage(f"Line {line_idx} at idx={idx}: End low point detected at idx={low_right_idx}, elev={low_right_elev:.4f}")

#                                     # Detect peak using a sliding window for wider peaks and narrow peaks
#                                     if idx >= window_size:
#                                         window_start = max(0, idx - window_size)
#                                         window_elevations = df_input["elevation"].iloc[window_start:idx + 1]
#                                         max_elev_idx = window_elevations.idxmax()
#                                         max_elev = window_elevations.iloc[max_elev_idx - window_start]
#                                         if max_elev_idx == idx and max_elev > previous and (max_elev >= current or direction == "DOWN"):
#                                             peak_idx = max_elev_idx
#                                             peak_elev = max_elev
#                                             arcpy.AddMessage(f"Line {line_idx} at idx={idx}: Wider peak detected at idx={peak_idx}, elev={peak_elev:.4f}")
#                                         elif direction_prev == "UP" and direction == "DOWN":
#                                             peak_idx = idx - 1
#                                             peak_elev = previous
#                                             arcpy.AddMessage(f"Line {line_idx} at idx={idx}: Narrow peak detected at idx={peak_idx}, elev={peak_elev:.4f}")
#                                         elif direction_prev == "UP" and direction == "FLAT" and peak_idx is None:
#                                             peak_idx = idx - 1
#                                             peak_elev = previous
#                                             arcpy.AddMessage(f"Line {line_idx} at idx={idx}: Tentative peak detected at idx={peak_idx}, elev={peak_elev:.4f}")

#                                     if low_right_idx is not None and peak_idx is not None and low_left_idx <= peak_idx <= low_right_idx:
#                                         left_height = peak_elev - low_left_elev
#                                         right_height = peak_elev - low_right_elev
#                                         agg_height = min(left_height, right_height)
#                                         agg_width = (low_right_idx - low_left_idx) * dx
#                                         agg_height_inches = agg_height * inches_per_meter

#                                         if agg_height >= min_height_diff:
#                                             res.append([agg_width, low_left_idx, peak_idx, low_right_idx])
#                                             arcpy.AddMessage(f"Line {line_idx} at x={x:.4f}: Detected loose aggregate - Height={agg_height_inches:.4f} inches, Width={agg_width:.4f} m")

#                                         low_left_idx = low_right_idx
#                                         low_left_elev = low_right_elev
#                                         low_right_idx = None
#                                         low_right_elev = None
#                                         peak_idx = None
#                                         peak_elev = None

#                                     direction_prev = direction

#                             arcpy.AddMessage(f"Line {line_idx} at x={x:.4f}: Detected {len(res)} loose aggregate features.")

#                             if res:
#                                 elevation_plots.append({
#                                     "line_x": round(x, 4),
#                                     "sections": df_input["section"].tolist(),
#                                     "elevations": df_input["elevation"].tolist(),
#                                     "aggregates": res
#                                 })

#                             # Calculate mean and standard deviation of section indices for separation
#                             section_indices = df_input["section"].values
#                             mean_idx = np.mean(section_indices) if len(section_indices) > 0 else 0
#                             std_dev = np.std(section_indices) if len(section_indices) > 1 else 0.0
#                             arcpy.AddMessage(f"Line {line_idx}: Mean section index: {mean_idx:.4f}, Standard deviation: {std_dev:.4f}")

#                             # Separate aggregates into sections using standard deviation
#                             section_aggregates = defaultdict(list)
#                             for agg in res:
#                                 agg_width, low_left_idx, peak_idx, low_right_idx = agg
#                                 section_idx = peak_idx  # Use peak index as representative
#                                 if section_idx < (mean_idx - std_dev):
#                                     section_aggregates["left"].append(agg)
#                                 elif (mean_idx - std_dev) <= section_idx <= (mean_idx + std_dev):
#                                     section_aggregates["center"].append(agg)
#                                 else:
#                                     section_aggregates["right"].append(agg)
#                             arcpy.AddMessage(f"Line {line_idx}: Aggregates separated - Left: {len(section_aggregates['left'])}, Center: {len(section_aggregates['center'])}, Right: {len(section_aggregates['right'])}")

#                             # Detect highest severity per section
#                             meters_to_feet = 3.28084
#                             linear_extent_ft = dy * meters_to_feet  # Linear extent as distance between vertical lines
#                             line_data_for_line = []
#                             highest_severity = None
#                             max_height = 0
#                             for section_name, aggregates in section_aggregates.items():
#                                 if aggregates:
#                                     for agg in aggregates:
#                                         agg_width, low_left_idx, peak_idx, low_right_idx = agg
#                                         low_left_elev = df_input["elevation"].iloc[low_left_idx]
#                                         peak_elev = df_input["elevation"].iloc[peak_idx]
#                                         low_right_elev = df_input["elevation"].iloc[low_right_idx]
#                                         left_height = peak_elev - low_left_elev
#                                         right_height = peak_elev - low_right_elev
#                                         agg_height = max(left_height, right_height)
#                                         agg_height_inches = agg_height * inches_per_meter
#                                         if agg_height_inches > max_height:
#                                             max_height = agg_height_inches
#                                         severity = "Low" if agg_height_inches < 2.0 else "Medium" if agg_height_inches <= 4.0 else "High"
#                                         agg_info = {
#                                             "height": round(agg_height_inches, 4),
#                                             "linear_extent_ft": linear_extent_ft,
#                                             "line_x": round(x, 4),
#                                             "section": section_name,
#                                             "severity": severity,
#                                             "low_left_idx": low_left_idx,
#                                             "peak_idx": peak_idx,
#                                             "low_right_idx": low_right_idx
#                                         }
#                                         line_data_for_line.append(agg_info)
#                                         all_results[severity].append(agg_info)
#                                         severity_continuity[(section_name, severity)].append(x)
#                                         arcpy.AddMessage(f"Line {line_idx} Section {section_name}: Severity={severity}, Height={agg_height_inches:.4f} inches, Linear Extent={linear_extent_ft:.4f} ft")
#                                     # Update highest severity for this line
#                                     if severity == "High" or (severity == "Medium" and highest_severity != "High") or (severity == "Low" and highest_severity is None):
#                                         highest_severity = severity

#                             line_data.extend(line_data_for_line)
#                             if highest_severity:
#                                 line_severities[x] = highest_severity

#                     # Create polygonal features for each line with highest severity
#                     arcpy.AddMessage("Creating output polygon feature class...")
#                     arcpy.management.CreateFeatureclass(
#                         os.path.dirname(output_polygons),
#                         os.path.basename(output_polygons),
#                         "POLYGON",
#                         spatial_reference=dem_srs
#                     )
#                     arcpy.management.AddField(output_polygons, "Line_X", "DOUBLE")
#                     arcpy.management.AddField(output_polygons, "Severity", "TEXT")

#                     with arcpy.da.InsertCursor(output_polygons, ["SHAPE@", "Line_X", "Severity"]) as cursor:
#                         for x in x_steps:
#                             severity = line_severities.get(x, "None")
#                             if severity != "None":
#                                 # Create rectangle: x-dy/2 to x+dy/2, min_y to max_y
#                                 poly_coords = [
#                                     (x - dy/2, min_y),
#                                     (x + dy/2, min_y),
#                                     (x + dy/2, max_y),
#                                     (x - dy/2, max_y),
#                                     (x - dy/2, min_y)
#                                 ]
#                                 poly = Polygon(poly_coords)
#                                 clipped_poly = poly.intersection(polygon)
#                                 if not clipped_poly.is_empty:
#                                     # Convert to ArcGIS geometry
#                                     points = [arcpy.Point(pt[0], pt[1]) for pt in clipped_poly.exterior.coords]
#                                     array = arcpy.Array(points)
#                                     arcpy_poly = arcpy.Polygon(array, dem_srs)
#                                     cursor.insertRow([arcpy_poly, x, severity])
#                                     arcpy.AddMessage(f"Created polygon for Line at x={x:.4f} with severity={severity}")

#                     arcpy.AddMessage("Generating enhanced visualization...")
#                     arcpy.AddMessage(f"Total loose aggregates in line_data: {len(line_data)}")

#                     # Calculate Total Severity Linear Extents
#                     severity_lengths = {sev: sum(item["linear_extent_ft"] for item in items) for sev, items in all_results.items()}
#                     length_text = "\n".join([f"{sev}: {severity_lengths.get(sev, 0):.2f} ft" for sev in ["Low", "Medium", "High"]])
#                     arcpy.AddMessage(f"Length text: {length_text}")

#                     num_plots = len(elevation_plots)
#                     if num_plots == 0:
#                         arcpy.AddMessage("No loose aggregates detected. Visualization will show only the table and map.")
#                         fig = make_subplots(
#                             rows=1, cols=2,
#                             subplot_titles=("Details of All Loose Aggregates", "Severity Map"),
#                             specs=[[{'type': 'table'}, {'type': 'xy'}]],
#                             column_widths=[0.7, 0.3]
#                         )
#                         total_height = 600  # Increased height for better visibility
#                         table_row = 1
#                         table_col = 1
#                         map_row = 1
#                         map_col = 2
#                         annotation_x = 0.85
#                         annotation_y = 0.5
#                     else:
#                         cols_per_row = 3
#                         num_rows = math.ceil(num_plots / cols_per_row)
#                         total_rows = num_rows + 1
#                         max_spacing = 1 / (total_rows - 1) if total_rows > 1 else 0.05
#                         vertical_spacing = min(0.02, max_spacing)  # Reduced spacing for larger plots
#                         arcpy.AddMessage(f"Using vertical_spacing={vertical_spacing:.6f} for {total_rows} rows")
#                         specs = [[{'type': 'xy'} if (row * cols_per_row + col) < num_plots else None
#                                   for col in range(cols_per_row)] for row in range(num_rows)]
#                         specs.append([{'type': 'table', 'colspan': 2}, None, {'type': 'xy'}])
#                         row_heights = [0.7 / num_rows] * num_rows + [0.3]
#                         subplot_titles = [f"Elevation at x={plot['line_x']} m" for plot in elevation_plots] + ["Details of All Loose Aggregates", "Severity Map"]
#                         fig = make_subplots(
#                             rows=total_rows,
#                             cols=cols_per_row,
#                             subplot_titles=subplot_titles,
#                             row_heights=row_heights,
#                             specs=specs,
#                             vertical_spacing=vertical_spacing,
#                             column_widths=[0.333, 0.333, 0.333]
#                         )
#                         total_height = 400 * total_rows  # Increased height per row for better visibility
#                         table_row = total_rows
#                         table_col = 1
#                         map_row = total_rows
#                         map_col = 3
#                         annotation_x = 0.85
#                         annotation_y = (1 - sum(row_heights[:-1])) / 2

#                     colors = plotly.colors.qualitative.Plotly
#                     num_colors = len(colors)

#                     for i, plot in enumerate(elevation_plots):
#                         row = (i // cols_per_row) + 1
#                         col = (i % cols_per_row) + 1
#                         line_color = colors[i % num_colors]
#                         fig.add_trace(go.Scatter(
#                             x=plot["sections"],
#                             y=plot["elevations"],
#                             mode='lines',
#                             name=f'Elevation at x={plot["line_x"]} m',
#                             line=dict(color=line_color)
#                         ), row=row, col=col)

#                         # Only visualize aggregates with highest severity per section
#                         for agg_info in line_data:
#                             if agg_info["line_x"] == plot["line_x"]:
#                                 low_left_idx = agg_info["low_left_idx"]
#                                 peak_idx = agg_info["peak_idx"]
#                                 low_right_idx = agg_info["low_right_idx"]
#                                 section_name = agg_info["section"]
#                                 severity = agg_info["severity"]
#                                 try:
#                                     low_left_elev = plot["elevations"][low_left_idx]
#                                     peak_elev = plot["elevations"][peak_idx]
#                                     low_right_elev = plot["elevations"][low_right_idx]
#                                 except IndexError as e:
#                                     arcpy.AddWarning(f"Index error in visualization for Line {line_idx} Section {section_name}: {str(e)}")
#                                     continue
#                                 height = max(peak_elev - low_left_elev, peak_elev - low_right_elev)
#                                 height_inches = height * inches_per_meter
#                                 color = 'green' if severity == "Low" else 'orange' if severity == "Medium" else 'red'
#                                 fig.add_shape(
#                                     type="line",
#                                     x0=peak_idx, x1=peak_idx,
#                                     y0=min(plot["elevations"]), y1=max(plot["elevations"]),
#                                     line=dict(color=color, width=2, dash="dash"),
#                                     row=row, col=col
#                                 )
#                                 fig.add_annotation(
#                                     x=peak_idx,
#                                     y=max(plot["elevations"]),
#                                     text=f"{severity} ({section_name})",
#                                     showarrow=True,
#                                     arrowhead=1,
#                                     arrowsize=1,
#                                     arrowwidth=1,
#                                     arrowcolor=color,
#                                     ax=0,
#                                     ay=-30,
#                                     font=dict(size=10, color=color),
#                                     row=row, col=col
#                                 )

#                     # Add map plot for polygonal features
#                     if line_severities:
#                         poly_x = []
#                         poly_y = []
#                         poly_severity = []
#                         for x, severity in line_severities.items():
#                             poly_coords = [
#                                 (x - dy/2, min_y),
#                                 (x + dy/2, min_y),
#                                 (x + dy/2, max_y),
#                                 (x - dy/2, max_y)
#                             ]
#                             clipped_poly = Polygon(poly_coords).intersection(polygon)
#                             if not clipped_poly.is_empty:
#                                 color = 'green' if severity == "Low" else 'orange' if severity == "Medium" else 'red'
#                                 for pt in clipped_poly.exterior.coords[:-1]:
#                                     poly_x.append(pt[0])
#                                     poly_y.append(pt[1])
#                                     poly_severity.append(severity)
#                                 poly_x.append(None)  # Break between polygons
#                                 poly_y.append(None)
#                                 poly_severity.append(None)
#                                 # Add filled polygon
#                                 fig.add_trace(go.Scatter(
#                                     x=[pt[0] for pt in clipped_poly.exterior.coords[:-1]],
#                                     y=[pt[1] for pt in clipped_poly.exterior.coords[:-1]],
#                                     mode='lines',
#                                     line=dict(color='black', width=1),
#                                     fill='toself',
#                                     fillcolor=color,
#                                     opacity=0.5,
#                                     name=f'Severity {severity}',
#                                     hovertemplate=f'Severity: {severity}<br>X: %{{x:.4f}} m<br>Y: %{{y:.4f}} m<extra></extra>'
#                                 ), row=map_row, col=map_col)

#                         fig.update_xaxes(title_text="X (m)", row=map_row, col=map_col, title_font=dict(size=10))
#                         fig.update_yaxes(title_text="Y (m)", row=map_row, col=map_col, title_font=dict(size=10), scaleanchor=f"x{map_col}", scaleratio=1)

#                     table_data = {
#                         "Line X (m)": [item["line_x"] for item in line_data],
#                         "Height (ft)": [item["height"] / inches_per_meter for item in line_data],  # Convert inches to feet
#                         "Linear Extent (ft)": [item["linear_extent_ft"] for item in line_data],
#                         "Section": [item["section"].capitalize() for item in line_data],
#                         "Severity": [item["severity"] for item in line_data]
#                     }
#                     arcpy.AddMessage(f"Table data: {table_data}")

#                     if not line_data:
#                         fig.add_annotation(
#                             x=0.35, y=0.5, xref="paper", yref="paper",
#                             text="No loose aggregates detected. Insufficient elevation variation in the DEM.",
#                             showarrow=False, font=dict(size=12, color="red"),
#                             align="center"
#                         )

#                     fig.add_trace(go.Table(
#                         header=dict(
#                             values=["Line X (m)", "Height (ft)", "Linear Extent (ft)", "Section", "Severity"],
#                             fill_color='paleturquoise',
#                             align='left',
#                             font=dict(size=12)
#                         ),
#                         cells=dict(
#                             values=[
#                                 table_data["Line X (m)"],
#                                 table_data["Height (ft)"],
#                                 table_data["Linear Extent (ft)"],
#                                 table_data["Section"],
#                                 table_data["Severity"]
#                             ],
#                             fill_color='lavender',
#                             align='left',
#                             font=dict(size=11)
#                         )
#                     ), row=table_row, col=table_col)

#                     for i in range(num_plots):
#                         row = (i // cols_per_row) + 1
#                         col = (i % cols_per_row) + 1
#                         fig.update_yaxes(title_text="Elevation (m)", row=row, col=col, title_font=dict(size=10), automargin=True)
#                         fig.update_xaxes(title_text="Section Index", row=row, col=col, title_font=dict(size=10))

#                     fig.add_annotation(
#                         x=annotation_x, y=annotation_y, xref="paper", yref="paper",
#                         text=f"Total Severity Lengths\n{length_text}",
#                         showarrow=False, align="center", bgcolor="white", bordercolor="black", borderwidth=1,
#                         font=dict(size=14, color="black"),
#                         xanchor="center", yanchor="middle"
#                     )

#                     fig.update_layout(
#                         title="Loose Aggregate Severity Analysis Across Polygon (Linear Feet)",
#                         height=total_height,
#                         showlegend=True,
#                         template="simple_white",
#                         margin=dict(l=60, r=100, t=100, b=60)
#                     )

#                     try:
#                         fig.write_html(output_html)
#                         arcpy.AddMessage(f"Enhanced plot saved to: {output_html}")
#                     except Exception as e:
#                         raise arcpy.ExecuteError(f"Failed to save HTML plot: {str(e)}")

#                     arcpy.AddMessage("Loose aggregate severity detection for polygon complete.")

#             except Exception as e:
#                 arcpy.AddError(f"Failed to process DEM file: {str(e)}")
#                 raise

#         except Exception as e:
#             raise arcpy.ExecuteError(f"Execution failed: {str(e)}")

# class Toolbox(object):
#     def __init__(self):
#         self.label = "TESTING TOOLBOX"
#         self.alias = "testing_toolbox"
#         self.tools = [LooseAggregatePolygonTool]

# class ToolValidator(object):
#     def __init__(self):
#         self.params = None

#     def initializeParameters(self, parameters):
#         self.params = parameters
#         return

#     def updateParameters(self, parameters):
#         if parameters[4].valueAsText:
#             output_path = parameters[4].valueAsText
#             if not output_path.lower().endswith('.html'):
#                 parameters[4].value = output_path + '.html'
#         return

#     def updateMessages(self, parameters):
#         if parameters[0].valueAsText:
#             desc = arcpy.Describe(parameters[0].valueAsText)
#             if desc.shapeType != "Polygon":
#                 parameters[0].setErrorMessage("Input feature must be a polygon.")

#         if parameters[1].valueAsText:
#             dem_path = parameters[1].valueAsText
#             if not os.path.exists(dem_path):
#                 parameters[1].setErrorMessage("DEM file does not exist.")
#             elif not dem_path.lower().endswith(('.tif', '.tiff', '.img')):
#                 parameters[1].setWarningMessage("DEM file should be a raster format (e.g., .tif, .img).")

#         if parameters[2].value is not None and parameters[2].value <= 0:
#             parameters[2].setErrorMessage("Sampling distance must be greater than 0.")

#         if parameters[3].value is not None and parameters[3].value <= 0:
#             parameters[3].setErrorMessage("Aggregate width must be greater than 0.")

#         if parameters[4].valueAsText:
#             output_path = parameters[4].valueAsText
#             if not output_path.lower().endswith('.html'):
#                 parameters[4].setErrorMessage("Output file must have a .html extension.")
#             output_dir = os.path.dirname(output_path)
#             if output_dir and not os.path.exists(output_dir):
#                 parameters[4].setErrorMessage("Output directory does not exist.")

#         if parameters[5].valueAsText:
#             output_fc = parameters[5].valueAsText
#             output_dir = os.path.dirname(output_fc)
#             if output_dir and not os.path.exists(output_dir):
#                 parameters[5].setErrorMessage("Output feature class directory does not exist.")

#         return
