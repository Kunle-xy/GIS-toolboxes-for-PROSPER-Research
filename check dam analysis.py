# -*- coding: utf-8 -*-

import arcpy
import math
import os
import json
from datetime import datetime
import time
import gc


class Toolbox:
    def __init__(self):
        self.label = "Rectangle Padding Toolbox"
        self.alias = "rectpadding"
        self.tools = [RectanglePaddingTool]


class RectanglePaddingTool:
    def __init__(self):
        self.label = "Pad Rectangle and Clip Rasters"
        self.description = "Pads a rectangular polygon, clips rasters, generates lines, and performs elevation analysis."

    def getParameterInfo(self):
        param0 = arcpy.Parameter(
            displayName="Input Rectangle Polygon",
            name="input_polygon",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input")
        param0.filter.list = ["Polygon"]
        
        param1 = arcpy.Parameter(
            displayName="Padding Distance (meters)",
            name="padding_distance",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input")
        param1.value = 5.0
        
        param2 = arcpy.Parameter(
            displayName="DEM Raster",
            name="dem_raster",
            datatype="GPRasterLayer",
            parameterType="Required",
            direction="Input")
        
        param3 = arcpy.Parameter(
            displayName="Orthophoto Raster",
            name="ortho_raster",
            datatype="GPRasterLayer",
            parameterType="Optional",
            direction="Input")
        
        param4 = arcpy.Parameter(
            displayName="Multispectral Raster",
            name="multispectral_raster",
            datatype="GPRasterLayer",
            parameterType="Optional",
            direction="Input")
        
        param5 = arcpy.Parameter(
            displayName="Output Folder",
            name="output_folder",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input")
        
        param6 = arcpy.Parameter(
            displayName="Output Padded Polygon",
            name="output_polygon",
            datatype="DEFeatureClass",
            parameterType="Derived",
            direction="Output")
        
        param7 = arcpy.Parameter(
            displayName="Interval between Parallel Lines (meters)",
            name="line_interval",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input")
        param7.value = 0.5
        
        param8 = arcpy.Parameter(
            displayName="Output Parallel Lines",
            name="output_lines",
            datatype="DEFeatureClass",
            parameterType="Derived",
            direction="Output")
        
        param9 = arcpy.Parameter(
            displayName="Sampling Percentage (e.g., 1 for 1% = 100 points)",
            name="sampling_percentage",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input")
        param9.value = 1.0
        
        return [param0, param1, param2, param3, param4, param5, param6, param7, param8, param9]

    def isLicensed(self):
        return True

    def cleanup_memory_workspace(self):
        """Clean up ALL items in memory workspace to prevent state persistence"""
        try:
            arcpy.env.workspace = "memory"
            # List and delete all feature classes in memory
            fc_list = arcpy.ListFeatureClasses() or []
            for fc in fc_list:
                try:
                    arcpy.management.Delete("memory\\{}".format(fc))
                except:
                    pass
            
            # List and delete all tables in memory
            table_list = arcpy.ListTables() or []
            for table in table_list:
                try:
                    arcpy.management.Delete("memory\\{}".format(table))
                except:
                    pass
            
            # List and delete all rasters in memory
            raster_list = arcpy.ListRasters() or []
            for raster in raster_list:
                try:
                    arcpy.management.Delete("memory\\{}".format(raster))
                except:
                    pass
                    
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            arcpy.AddWarning("Memory cleanup warning: {}".format(str(e)))

    def execute(self, parameters, messages):
        # ============================================
        # INITIAL CLEANUP - Ensure fresh state
        # ============================================
        arcpy.AddMessage("=" * 50)
        arcpy.AddMessage("INITIALIZING NEW RUN...")
        arcpy.AddMessage("=" * 50)
        
        # Clean up any leftover memory items from previous runs
        self.cleanup_memory_workspace()
        
        # Clear any cached results
        arcpy.ClearEnvironment("workspace")
        arcpy.ClearEnvironment("scratchWorkspace")
        
        # Get parameters FRESH
        input_polygon = parameters[0].valueAsText
        padding_distance = float(parameters[1].value)
        dem_raster = parameters[2].valueAsText
        ortho_raster = parameters[3].valueAsText
        multispectral_raster = parameters[4].valueAsText
        output_folder = parameters[5].valueAsText
        line_interval = float(parameters[7].value)
        sampling_percentage = float(parameters[9].value)
        
        num_sample_points = int(100 / sampling_percentage)
        arcpy.AddMessage("Sampling {} points per line ({}%)".format(num_sample_points, sampling_percentage))
        
        arcpy.env.overwriteOutput = True
        
        # Create UNIQUE timestamp with microseconds to ensure uniqueness
        time.sleep(0.1)  # Small delay to ensure unique timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # Create run folder with unique timestamp
        run_folder = os.path.join(output_folder, "Run_{}".format(timestamp))
        
        # Ensure folder doesn't exist (extra safety)
        counter = 0
        original_run_folder = run_folder
        while os.path.exists(run_folder):
            counter += 1
            run_folder = "{}_{}".format(original_run_folder, counter)
        
        os.makedirs(run_folder)
        
        arcpy.AddMessage("=" * 50)
        arcpy.AddMessage("OUTPUT FOLDER: {}".format(run_folder))
        arcpy.AddMessage("Timestamp: {}".format(timestamp))
        arcpy.AddMessage("=" * 50)
        
        # Get spatial reference
        spatial_ref = arcpy.Describe(input_polygon).spatialReference
        
        # Create unique temp polygon name using full timestamp
        temp_poly_name = "padded_poly_{}".format(timestamp.replace("_", ""))
        temp_poly_path = "memory\\{}".format(temp_poly_name)
        
        # Delete if exists (safety cleanup)
        if arcpy.Exists(temp_poly_path):
            arcpy.management.Delete(temp_poly_path)

        arcpy.management.CreateFeatureclass(
            out_path="memory",
            out_name=temp_poly_name,
            geometry_type="POLYGON",
            spatial_reference=spatial_ref
        )
        
        arcpy.AddMessage("Calculating padding...")

        # Initialize all variables fresh
        padded_vertices = None
        original_vertices = None
        is_AB_longer = None
        flow_angle = None
        dist_AB = None
        dist_AD = None

        with arcpy.da.SearchCursor(input_polygon, ["SHAPE@"]) as cursor:
            for row in cursor:
                poly = row[0]
                part = poly.getPart(0)
                vertices = [(p.X, p.Y) for p in part if p]

                if len(vertices) < 4:
                    arcpy.AddError("Input feature is not a valid rectangle.")
                    self.cleanup_memory_workspace()
                    return

                A, B, C, D = vertices[0], vertices[1], vertices[2], vertices[3]
                original_vertices = (A, B, C, D)
                
                dist_AB = math.hypot(B[0]-A[0], B[1]-A[1])
                dist_AD = math.hypot(D[0]-A[0], D[1]-A[1])

                arcpy.AddMessage("Original rectangle: AB={:.2f}m, AD={:.2f}m".format(dist_AB, dist_AD))

                if dist_AB >= dist_AD:
                    is_AB_longer = True
                    flow_angle = math.atan2(D[1]-A[1], D[0]-A[0])
                    flow_cos = math.cos(flow_angle)
                    flow_sin = math.sin(flow_angle)
                    
                    A_p = (A[0] - padding_distance * flow_cos, A[1] - padding_distance * flow_sin)
                    B_p = (B[0] - padding_distance * flow_cos, B[1] - padding_distance * flow_sin)
                    D_p = (D[0] + padding_distance * flow_cos, D[1] + padding_distance * flow_sin)
                    C_p = (C[0] + padding_distance * flow_cos, C[1] + padding_distance * flow_sin)
                else:
                    is_AB_longer = False
                    flow_angle = math.atan2(B[1]-A[1], B[0]-A[0])
                    flow_cos = math.cos(flow_angle)
                    flow_sin = math.sin(flow_angle)
                    
                    A_p = (A[0] - padding_distance * flow_cos, A[1] - padding_distance * flow_sin)
                    D_p = (D[0] - padding_distance * flow_cos, D[1] - padding_distance * flow_sin)
                    B_p = (B[0] + padding_distance * flow_cos, B[1] + padding_distance * flow_sin)
                    C_p = (C[0] + padding_distance * flow_cos, C[1] + padding_distance * flow_sin)

                padded_vertices = (A_p, B_p, C_p, D_p)

                new_array = arcpy.Array([arcpy.Point(*A_p), arcpy.Point(*B_p), 
                                         arcpy.Point(*C_p), arcpy.Point(*D_p)])
                padded_geom = arcpy.Polygon(new_array, spatial_ref)

                with arcpy.da.InsertCursor(temp_poly_path, ["SHAPE@"]) as i_cursor:
                    i_cursor.insertRow([padded_geom])

        # Save padded polygon
        out_shp_path = os.path.join(run_folder, "Padded_Polygon.shp")
        arcpy.management.CopyFeatures(temp_poly_path, out_shp_path)
        parameters[6].value = out_shp_path 
        arcpy.AddMessage("Saved: Padded_Polygon.shp")

        # GENERATE PARALLEL LINES
        arcpy.AddMessage("Generating parallel lines...")
        
        A_p, B_p, C_p, D_p = padded_vertices
        A, B, C, D = original_vertices
        
        if is_AB_longer:
            original_flow_distance = dist_AD
            padded_flow_distance = original_flow_distance + 2 * padding_distance
            line_start_edge = (A_p, B_p)
        else:
            original_flow_distance = dist_AB
            padded_flow_distance = original_flow_distance + 2 * padding_distance
            line_start_edge = (A_p, D_p)
        
        before_end = padding_distance
        within_end = padding_distance + original_flow_distance
        
        # Fresh line position lists
        line_positions = []
        current_position = 0
        while current_position <= padded_flow_distance:
            line_positions.append(current_position)
            current_position += line_interval
        
        lines_before = []
        lines_within = []
        lines_after = []
        epsilon = 0.0001
        
        for position in line_positions:
            if position < before_end - epsilon:
                lines_before.append(position)
            elif position <= within_end + epsilon:
                lines_within.append(position)
            else:
                lines_after.append(position)
        
        arcpy.AddMessage("Lines: Before={}, Within={}, After={}".format(len(lines_before), len(lines_within), len(lines_after)))
        
        lines_fc_path = os.path.join(run_folder, "Parallel_Lines.shp")
        
        arcpy.management.CreateFeatureclass(
            out_path=run_folder,
            out_name="Parallel_Lines.shp",
            geometry_type="POLYLINE",
            spatial_reference=spatial_ref
        )
        
        arcpy.management.AddField(lines_fc_path, "Group", "TEXT", field_length=50)
        arcpy.management.AddField(lines_fc_path, "Line_ID", "LONG")
        arcpy.management.AddField(lines_fc_path, "Position", "DOUBLE")
        
        flow_unit_x = math.cos(flow_angle)
        flow_unit_y = math.sin(flow_angle)
        left_start = line_start_edge[0]
        right_start = line_start_edge[1]
        
        def create_line_at_position(position):
            offset_x = position * flow_unit_x
            offset_y = position * flow_unit_y
            start_point = arcpy.Point(left_start[0] + offset_x, left_start[1] + offset_y)
            end_point = arcpy.Point(right_start[0] + offset_x, right_start[1] + offset_y)
            line_array = arcpy.Array([start_point, end_point])
            return arcpy.Polyline(line_array, spatial_ref)
        
        # Fresh dictionaries
        line_geometries = {"Before": [], "Within": [], "After": []}
        line_positions_dict = {"Before": [], "Within": [], "After": []}
        
        line_id = 1
        with arcpy.da.InsertCursor(lines_fc_path, ["SHAPE@", "Group", "Line_ID", "Position"]) as cursor:
            for position in lines_before:
                line_geom = create_line_at_position(position)
                cursor.insertRow([line_geom, "Before", line_id, position])
                line_geometries["Before"].append(line_geom)
                line_positions_dict["Before"].append(position)
                line_id += 1
            
            for position in lines_within:
                line_geom = create_line_at_position(position)
                cursor.insertRow([line_geom, "Within", line_id, position])
                line_geometries["Within"].append(line_geom)
                line_positions_dict["Within"].append(position)
                line_id += 1
            
            for position in lines_after:
                line_geom = create_line_at_position(position)
                cursor.insertRow([line_geom, "After", line_id, position])
                line_geometries["After"].append(line_geom)
                line_positions_dict["After"].append(position)
                line_id += 1
        
        parameters[8].value = lines_fc_path
        arcpy.AddMessage("Saved: Parallel_Lines.shp")

        # CLIP RASTERS
        arcpy.AddMessage("Clipping rasters...")
        dem_padded_path = None
        
        # Log spatial reference info for debugging
        arcpy.AddMessage("Input polygon spatial reference: {}".format(spatial_ref.name))
        dem_sr = arcpy.Describe(dem_raster).spatialReference
        arcpy.AddMessage("DEM spatial reference: {}".format(dem_sr.name))
        if spatial_ref.name != dem_sr.name:
            arcpy.AddWarning("WARNING: Spatial reference mismatch between polygon and DEM!")
        
        for raster, label in [(dem_raster, "DEM"), (ortho_raster, "Ortho"), (multispectral_raster, "Multispectral")]:
            if raster:
                orig_out = os.path.join(run_folder, "{}_Original.tif".format(label))
                arcpy.management.Clip(raster, "#", orig_out, input_polygon, "#", "ClippingGeometry", "NO_MAINTAIN_EXTENT")
                arcpy.AddMessage("Saved: {}_Original.tif".format(label))
                
                pad_out = os.path.join(run_folder, "{}_Padded.tif".format(label))
                arcpy.management.Clip(raster, "#", pad_out, temp_poly_path, "#", "ClippingGeometry", "NO_MAINTAIN_EXTENT")
                arcpy.AddMessage("Saved: {}_Padded.tif".format(label))
                
                if label == "DEM":
                    dem_padded_path = pad_out
                    # Verify clipped DEM has valid data
                    try:
                        dem_desc = arcpy.Describe(pad_out)
                        arcpy.AddMessage("DEM_Padded extent: {} to {}".format(
                            (dem_desc.extent.XMin, dem_desc.extent.YMin),
                            (dem_desc.extent.XMax, dem_desc.extent.YMax)))
                    except Exception as e:
                        arcpy.AddWarning("Could not read DEM extent: {}".format(str(e)))

        # ELEVATION SAMPLING
        arcpy.AddMessage("=" * 50)
        arcpy.AddMessage("ELEVATION PROFILE ANALYSIS")
        arcpy.AddMessage("=" * 50)        
        if dem_padded_path:
            # Track NoData occurrences for diagnostics
            nodata_count = [0]  # Use list to allow modification in nested function
            
            # Get DEM spatial reference for coordinate transformation
            dem_sr = arcpy.Describe(dem_padded_path).spatialReference
            needs_projection = spatial_ref.name != dem_sr.name
            
            if needs_projection:
                arcpy.AddMessage("Projecting sample points from {} to {}".format(spatial_ref.name, dem_sr.name))
            
            def sample_line_elevations(line_geom, dem_path, num_points):
                elevations = []
                line_length = line_geom.length
                
                for i in range(num_points):
                    fraction = i / (num_points - 1) if num_points > 1 else 0.5
                    distance = fraction * line_length
                    point = line_geom.positionAlongLine(distance)
                    
                    # Project point to DEM's coordinate system if needed
                    if needs_projection:
                        projected_point = point.projectAs(dem_sr)
                        query_x = projected_point.firstPoint.X
                        query_y = projected_point.firstPoint.Y
                    else:
                        query_x = point.firstPoint.X
                        query_y = point.firstPoint.Y
                    
                    result = arcpy.management.GetCellValue(dem_path, "{} {}".format(query_x, query_y))
                    cell_value = result.getOutput(0)
                    
                    try:
                        elevation = float(cell_value)
                        elevations.append(elevation)
                    except (ValueError, TypeError):
                        elevations.append(None)
                        nodata_count[0] += 1
                        # Log first few NoData occurrences for debugging
                        if nodata_count[0] <= 3:
                            arcpy.AddWarning("NoData at point ({:.2f}, {:.2f}) - cell value: '{}'".format(
                                query_x, query_y, cell_value))
                
                return elevations
            
            # ============================================
            # FRONT PROFILE DATA (Cross-sectional view)
            # ============================================
            arcpy.AddMessage("-" * 40)
            arcpy.AddMessage("FRONT PROFILE (Cross-sectional)")
            arcpy.AddMessage("-" * 40)
            
            # Fresh data structures
            front_profile_data = {"Before": [], "Within": [], "After": []}
            
            for segment_name, lines in line_geometries.items():
                arcpy.AddMessage("Sampling {} ({} lines)...".format(segment_name, len(lines)))
                segment_valid_count = 0
                segment_total_count = 0
                for line_geom in lines:
                    elevations = sample_line_elevations(line_geom, dem_padded_path, num_sample_points)
                    front_profile_data[segment_name].append(elevations)
                    # Track valid vs total samples
                    segment_total_count += len(elevations)
                    segment_valid_count += sum(1 for e in elevations if e is not None)
                
                # Warn if no valid data found for this segment
                if segment_valid_count == 0:
                    arcpy.AddWarning("WARNING: No valid elevation data found for '{}' segment! ({} samples attempted)".format(
                        segment_name, segment_total_count))
                elif segment_valid_count < segment_total_count:
                    arcpy.AddMessage("  {}: {}/{} valid samples ({:.1f}%)".format(
                        segment_name, segment_valid_count, segment_total_count, 
                        100.0 * segment_valid_count / segment_total_count))
            
            # Report total NoData occurrences
            if nodata_count[0] > 0:
                arcpy.AddWarning("Total NoData samples encountered: {} (check DEM coverage and spatial reference)".format(nodata_count[0]))
            
            # Calculate front profile statistics
            front_stats = {}
            global_min_elevation = float('inf')
            global_max_elevation = float('-inf')
            
            for segment_name, segment_elevations in front_profile_data.items():
                if segment_elevations:
                    means = []
                    stds = []
                    mins = []
                    maxs = []
                    
                    for point_idx in range(num_sample_points):
                        values = []
                        for line_elevations in segment_elevations:
                            if point_idx < len(line_elevations) and line_elevations[point_idx] is not None:
                                values.append(line_elevations[point_idx])
                        
                        if values:
                            mean_val = sum(values) / len(values)
                            variance = sum((x - mean_val) ** 2 for x in values) / len(values) if len(values) > 1 else 0
                            std_val = math.sqrt(variance)
                            min_val = min(values)
                            max_val = max(values)
                            means.append(mean_val)
                            stds.append(std_val)
                            mins.append(min_val)
                            maxs.append(max_val)
                            
                            if min_val < global_min_elevation:
                                global_min_elevation = min_val
                            if max_val > global_max_elevation:
                                global_max_elevation = max_val
                        else:
                            means.append(None)
                            stds.append(None)
                            mins.append(None)
                            maxs.append(None)
                    
                    front_stats[segment_name] = {
                        "means": means, 
                        "stds": stds,
                        "mins": mins,
                        "maxs": maxs,
                        "num_lines": len(line_geometries[segment_name])
                    }
                    
                    valid_means = [m for m in means if m is not None]
                    if valid_means:
                        arcpy.AddMessage("  {}: avg elevation = {:.2f}m".format(segment_name, sum(valid_means)/len(valid_means)))
            
            # ============================================
            # SIDE PROFILE DATA (Longitudinal view)
            # ============================================
            arcpy.AddMessage("-" * 40)
            arcpy.AddMessage("SIDE PROFILE (Longitudinal)")
            arcpy.AddMessage("-" * 40)
            
            # Fresh data structures
            side_profile_data = {"Before": [], "Within": [], "After": []}
            
            for segment_name, lines in line_geometries.items():
                arcpy.AddMessage("Processing {} ({} lines -> {} points)...".format(segment_name, len(lines), len(lines)))
                
                for idx, line_geom in enumerate(lines):
                    elevations = sample_line_elevations(line_geom, dem_padded_path, num_sample_points)
                    valid_elevations = [e for e in elevations if e is not None]
                    
                    if valid_elevations:
                        line_mean = sum(valid_elevations) / len(valid_elevations)
                        line_variance = sum((x - line_mean) ** 2 for x in valid_elevations) / len(valid_elevations) if len(valid_elevations) > 1 else 0
                        line_std = math.sqrt(line_variance)
                        line_min = min(valid_elevations)
                        line_max = max(valid_elevations)
                        
                        side_profile_data[segment_name].append({
                            "position": line_positions_dict[segment_name][idx],
                            "mean": line_mean,
                            "std": line_std,
                            "min": line_min,
                            "max": line_max
                        })
                        
                        if line_min < global_min_elevation:
                            global_min_elevation = line_min
                        if line_max > global_max_elevation:
                            global_max_elevation = line_max
                    else:
                        side_profile_data[segment_name].append({
                            "position": line_positions_dict[segment_name][idx],
                            "mean": None,
                            "std": None,
                            "min": None,
                            "max": None
                        })
            
            # Build side profile stats with LINE NUMBERS (1 to N)
            side_stats = {}
            
            # Calculate cumulative line numbers
            num_before = len(side_profile_data["Before"])
            num_within = len(side_profile_data["Within"])
            num_after = len(side_profile_data["After"])
            total_lines = num_before + num_within + num_after
            
            for segment_name, data_points in side_profile_data.items():
                if segment_name == "Before":
                    start_line = 1
                elif segment_name == "Within":
                    start_line = num_before + 1
                else:  # After
                    start_line = num_before + num_within + 1
                
                line_numbers = [start_line + i for i in range(len(data_points))]
                means = [d["mean"] for d in data_points]
                stds = [d["std"] for d in data_points]
                mins = [d["min"] for d in data_points]
                maxs = [d["max"] for d in data_points]
                
                side_stats[segment_name] = {
                    "line_numbers": line_numbers,
                    "means": means,
                    "stds": stds,
                    "mins": mins,
                    "maxs": maxs,
                    "num_points": len(data_points)
                }
                
                valid_means = [m for m in means if m is not None]
                if valid_means:
                    arcpy.AddMessage("  {}: {} points, avg elevation = {:.2f}m".format(
                        segment_name, len(data_points), sum(valid_means)/len(valid_means)))
            
            # Calculate y-axis range
            y_axis_min = global_min_elevation - 3
            y_axis_max = global_max_elevation + 2
            
            arcpy.AddMessage("Elevation range: {:.2f}m to {:.2f}m".format(global_min_elevation, global_max_elevation))
            
            # Boundary line numbers for vertical demarcation
            boundary_before_within = num_before + 0.5
            boundary_within_after = num_before + num_within + 0.5
            
            # Save JSON
            stats_json_path = os.path.join(run_folder, "elevation_stats.json")
            json_data = {
                "run_timestamp": timestamp,
                "parameters": {
                    "padding_distance": padding_distance,
                    "line_interval": line_interval,
                    "sampling_percentage": sampling_percentage,
                    "num_sample_points": num_sample_points,
                    "padded_flow_distance": padded_flow_distance
                },
                "elevation_range": {
                    "global_min": global_min_elevation,
                    "global_max": global_max_elevation
                },
                "line_counts": {
                    "before": num_before,
                    "within": num_within,
                    "after": num_after,
                    "total": total_lines
                },
                "front_profile": {},
                "side_profile": {}
            }
            
            for segment_name, segment_stats in front_stats.items():
                json_data["front_profile"][segment_name] = {
                    "means": segment_stats["means"],
                    "stds": segment_stats["stds"],
                    "mins": segment_stats["mins"],
                    "maxs": segment_stats["maxs"],
                    "num_lines": segment_stats["num_lines"]
                }
            
            for segment_name, segment_stats in side_stats.items():
                json_data["side_profile"][segment_name] = {
                    "line_numbers": segment_stats["line_numbers"],
                    "means": segment_stats["means"],
                    "stds": segment_stats["stds"],
                    "mins": segment_stats["mins"],
                    "maxs": segment_stats["maxs"],
                    "num_points": segment_stats["num_points"]
                }
            
            with open(stats_json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2)
            
            arcpy.AddMessage("Saved: elevation_stats.json")
            
            # GENERATE HTML
            arcpy.AddMessage("Generating HTML plots...")
            
            try:
                # Front profile x-values (0-100%)
                front_x_values = [i * 100.0 / (num_sample_points - 1) if num_sample_points > 1 else 50.0 for i in range(num_sample_points)]
                
                colors = {
                    "Before": {"line": "rgb(65, 105, 225)", "fill": "rgba(65, 105, 225, 0.3)"},
                    "Within": {"line": "rgb(34, 139, 34)", "fill": "rgba(34, 139, 34, 0.3)"},
                    "After": {"line": "rgb(220, 20, 60)", "fill": "rgba(220, 20, 60, 0.3)"},
                    "Continuous": {"line": "rgb(75, 0, 130)", "fill": "rgba(75, 0, 130, 0.25)"}
                }
                
                # Helper function to find shaded regions where Within < Before OR Within < After
                def find_shaded_regions(within_vals, before_vals, after_vals, x_values):
                    """Returns list of (x_start, x_end) tuples for regions to shade"""
                    regions = []
                    in_shaded = False
                    start_x = None
                    
                    for i in range(len(x_values)):
                        w = within_vals[i] if i < len(within_vals) else None
                        b = before_vals[i] if i < len(before_vals) else None
                        a = after_vals[i] if i < len(after_vals) else None
                        
                        # Check if Within is lower than Before OR After (should be shaded)
                        should_shade = False
                        if w is not None and b is not None and a is not None:
                            if w < b or w < a:
                                should_shade = True
                        
                        if should_shade and not in_shaded:
                            # Start new shaded region
                            in_shaded = True
                            start_x = x_values[i]
                        elif not should_shade and in_shaded:
                            # End shaded region
                            in_shaded = False
                            end_x = x_values[i - 1] if i > 0 else x_values[i]
                            regions.append((start_x, end_x))
                    
                    # Close final region if still in shaded
                    if in_shaded:
                        regions.append((start_x, x_values[-1]))
                    
                    return regions
                
                # Helper function to calculate area under curve using trapezoidal rule
                def calculate_area_under_curve(x_values, y_values):
                    """Calculate area under curve using trapezoidal integration"""
                    if len(x_values) < 2 or len(y_values) < 2:
                        return 0.0
                    area = 0.0
                    for i in range(1, len(x_values)):
                        if y_values[i] is not None and y_values[i-1] is not None:
                            # dx = x_values[i] - x_values[i-1]
                            dx = 1.0 #1 unit needed for percentage scale
                            avg_y = (y_values[i] + y_values[i-1]) / 2.0
                            area += dx * avg_y
                    return area
                
                # ============================================
                # PLOT 1: Front Profile - Mean ± STD
                # ============================================
                traces1 = []
                
                # Display name mapping for legend
                display_names = {"Before": "Upstream", "Within": "Check Dam", "After": "Downstream"}
                
                for segment_name in ["Before", "Within", "After"]:
                    if segment_name not in front_stats:
                        continue
                    
                    means = front_stats[segment_name]["means"]
                    stds = front_stats[segment_name]["stds"]
                    num_lines_seg = front_stats[segment_name]["num_lines"]
                    
                    valid_indices = [i for i, m in enumerate(means) if m is not None]
                    if not valid_indices:
                        continue
                    
                    x_valid = [front_x_values[i] for i in valid_indices]
                    means_valid = [means[i] for i in valid_indices]
                    stds_valid = [stds[i] if stds[i] is not None else 0 for i in valid_indices]
                    
                    upper_bound = [m + s for m, s in zip(means_valid, stds_valid)]
                    lower_bound = [m - s for m, s in zip(means_valid, stds_valid)]
                    
                    # Upper bound line (invisible, for fill)
                    traces1.append({
                        "x": x_valid,
                        "y": upper_bound,
                        "mode": "lines",
                        "line": {"width": 0},
                        "showlegend": False,
                        "hoverinfo": "skip"
                    })
                    
                    # Lower bound line with fill to upper
                    traces1.append({
                        "x": x_valid,
                        "y": lower_bound,
                        "mode": "lines",
                        "line": {"width": 0},
                        "fill": "tonexty",
                        "fillcolor": colors[segment_name]["fill"],
                        "showlegend": False,
                        "hoverinfo": "skip"
                    })
                    
                    # Mean line
                    traces1.append({
                        "x": x_valid,
                        "y": means_valid,
                        "mode": "lines",
                        "line": {"color": colors[segment_name]["line"], "width": 2.5},
                        "name": display_names[segment_name] + " Mean +/- Std (n=" + str(num_lines_seg) + ")"
                    })
                
                # ============================================
                # PLOT 2: Front Profile - MAX with shaded regions and AUC
                # ============================================
                traces2 = []
                areas_plot2 = {}
                
                for segment_name in ["Before", "Within", "After"]:
                    if segment_name not in front_stats:
                        continue
                        
                    maxs = front_stats[segment_name]["maxs"]
                    num_lines_seg = front_stats[segment_name]["num_lines"]
                    
                    valid_indices = [i for i, m in enumerate(maxs) if m is not None]
                    if not valid_indices:
                        continue
                    
                    x_valid = [front_x_values[i] for i in valid_indices]
                    maxs_valid = [maxs[i] for i in valid_indices]
                    
                    # Calculate area under curve
                    area = calculate_area_under_curve(x_valid, maxs_valid)
                    areas_plot2[segment_name] = area
                    
                    traces2.append({
                        "x": x_valid,
                        "y": maxs_valid,
                        "mode": "lines",
                        "line": {"color": colors[segment_name]["line"], "width": 2.5},
                        "name": display_names[segment_name] + " Max (AUC=" + "{:.1f}".format(area) + ")"
                    })
                
                # Calculate area differences for Plot 2
                diff_before_within_p2 = areas_plot2.get("Before", 0) - areas_plot2.get("Within", 0)
                diff_after_within_p2 = areas_plot2.get("Before", 0) - areas_plot2.get("After", 0)
                
                # Find shaded regions for Plot 2 (MAX comparison)
                shaded_regions_plot2 = find_shaded_regions(
                    front_stats.get("Within", {}).get("maxs", []),
                    front_stats.get("Before", {}).get("maxs", []),
                    front_stats.get("After", {}).get("maxs", []),
                    front_x_values
                )
                
                # ============================================
                # PLOT 3: Front Profile - MIN for Within, MAX for Before/After
                # ============================================
                traces3 = []
                areas_plot3 = {}
                plot3_data = {}  # Store data for shaded region calculation
                
                for segment_name in ["Before", "Within", "After"]:
                    if segment_name not in front_stats:
                        continue
                    
                    num_lines_seg = front_stats[segment_name]["num_lines"]
                    
                    # Use MIN for Within, MAX for Before/After
                    if segment_name == "Within":
                        data_values = front_stats[segment_name]["mins"]
                        label = "Min"
                    else:
                        data_values = front_stats[segment_name]["maxs"]
                        label = "Max"
                    
                    valid_indices = [i for i, m in enumerate(data_values) if m is not None]
                    if not valid_indices:
                        continue
                    
                    x_valid = [front_x_values[i] for i in valid_indices]
                    y_valid = [data_values[i] for i in valid_indices]
                    
                    # Store for shaded region calculation
                    plot3_data[segment_name] = data_values
                    
                    # Calculate area under curve
                    area = calculate_area_under_curve(x_valid, y_valid)
                    areas_plot3[segment_name] = area
                    
                    traces3.append({
                        "x": x_valid,
                        "y": y_valid,
                        "mode": "lines",
                        "line": {"color": colors[segment_name]["line"], "width": 2.5},
                        "name": display_names[segment_name] + " " + label + " (AUC=" + "{:.1f}".format(area) + ")"
                    })
                
                # Calculate area differences for Plot 3
                diff_before_within_p3 = areas_plot3.get("Before", 0) - areas_plot3.get("Within", 0)
                diff_after_within_p3 = areas_plot3.get("Before", 0) - areas_plot3.get("After", 0)
                
                # Find shaded regions for Plot 3 (Within MIN vs Before/After MAX)
                shaded_regions_plot3 = find_shaded_regions(
                    plot3_data.get("Within", []),
                    plot3_data.get("Before", []),
                    plot3_data.get("After", []),
                    front_x_values
                )
                
                # ============================================
                # SIDE PROFILE - Build continuous data arrays
                # ============================================
                continuous_x = []
                continuous_means = []
                continuous_stds = []
                continuous_maxs = []
                continuous_mins = []
                
                for segment_name in ["Before", "Within", "After"]:
                    if segment_name in side_stats:
                        continuous_x.extend(side_stats[segment_name]["line_numbers"])
                        continuous_means.extend(side_stats[segment_name]["means"])
                        continuous_stds.extend(side_stats[segment_name]["stds"])
                        continuous_maxs.extend(side_stats[segment_name]["maxs"])
                        continuous_mins.extend(side_stats[segment_name]["mins"])
                
                # Filter valid indices for continuous data
                valid_continuous = [i for i, m in enumerate(continuous_means) if m is not None]
                x_cont_valid = [continuous_x[i] for i in valid_continuous]
                means_cont_valid = [continuous_means[i] for i in valid_continuous]
                stds_cont_valid = [continuous_stds[i] if continuous_stds[i] is not None else 0 for i in valid_continuous]
                maxs_cont_valid = [continuous_maxs[i] for i in valid_continuous]
                mins_cont_valid = [continuous_mins[i] for i in valid_continuous]
                
                # ============================================
                # PLOT 4: Side Profile - Mean ± STD (Continuous)
                # ============================================
                traces4 = []
                
                upper_bound_cont = [m + s for m, s in zip(means_cont_valid, stds_cont_valid)]
                lower_bound_cont = [m - s for m, s in zip(means_cont_valid, stds_cont_valid)]
                
                # Upper bound line (invisible, for fill)
                traces4.append({
                    "x": x_cont_valid,
                    "y": upper_bound_cont,
                    "mode": "lines",
                    "line": {"width": 0},
                    "showlegend": False,
                    "hoverinfo": "skip"
                })
                
                # Lower bound line with fill to upper
                traces4.append({
                    "x": x_cont_valid,
                    "y": lower_bound_cont,
                    "mode": "lines",
                    "line": {"width": 0},
                    "fill": "tonexty",
                    "fillcolor": colors["Continuous"]["fill"],
                    "showlegend": False,
                    "hoverinfo": "skip"
                })
                
                # Mean line
                traces4.append({
                    "x": x_cont_valid,
                    "y": means_cont_valid,
                    "mode": "lines",
                    "line": {"color": colors["Continuous"]["line"], "width": 2.5},
                    "name": "Mean +/- Std (n=" + str(len(x_cont_valid)) + ")"
                })
                
                # ============================================
                # PLOT 5: Side Profile - MAX (Continuous)
                # ============================================
                traces5 = []
                
                traces5.append({
                    "x": x_cont_valid,
                    "y": maxs_cont_valid,
                    "mode": "lines",
                    "line": {"color": colors["Continuous"]["line"], "width": 2.5},
                    "name": "Max (n=" + str(len(x_cont_valid)) + ")"
                })
                
                # ============================================
                # PLOT 6: Side Profile - MIN (Continuous)
                # ============================================
                traces6 = []
                
                traces6.append({
                    "x": x_cont_valid,
                    "y": mins_cont_valid,
                    "mode": "lines",
                    "line": {"color": colors["Continuous"]["line"], "width": 2.5},
                    "name": "Min (n=" + str(len(x_cont_valid)) + ")"
                })
                
                # Build table rows for front profile
                front_table_rows = ""
                for segment_name in ["Before", "Within", "After"]:
                    if segment_name in front_stats:
                        m_list = [m for m in front_stats[segment_name]["means"] if m is not None]
                        s_list = [s for s in front_stats[segment_name]["stds"] if s is not None]
                        min_list = [m for m in front_stats[segment_name]["mins"] if m is not None]
                        max_list = [m for m in front_stats[segment_name]["maxs"] if m is not None]
                        avg_m = sum(m_list) / len(m_list) if m_list else 0
                        avg_s = sum(s_list) / len(s_list) if s_list else 0
                        avg_min = sum(min_list) / len(min_list) if min_list else 0
                        avg_max = sum(max_list) / len(max_list) if max_list else 0
                        front_table_rows += "<tr><td><strong>{}</strong></td><td>{:.2f}</td><td>+/-{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td></tr>\n".format(
                            segment_name, avg_m, avg_s, avg_min, avg_max)
                
                # Build table rows for side profile
                side_table_rows = ""
                for segment_name in ["Before", "Within", "After"]:
                    if segment_name in side_stats:
                        m_list = [m for m in side_stats[segment_name]["means"] if m is not None]
                        s_list = [s for s in side_stats[segment_name]["stds"] if s is not None]
                        min_list = [m for m in side_stats[segment_name]["mins"] if m is not None]
                        max_list = [m for m in side_stats[segment_name]["maxs"] if m is not None]
                        avg_m = sum(m_list) / len(m_list) if m_list else 0
                        avg_s = sum(s_list) / len(s_list) if s_list else 0
                        avg_min = sum(min_list) / len(min_list) if min_list else 0
                        avg_max = sum(max_list) / len(max_list) if max_list else 0
                        num_pts = side_stats[segment_name]["num_points"]
                        side_table_rows += "<tr><td><strong>{}</strong></td><td>{}</td><td>{:.2f}</td><td>+/-{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td></tr>\n".format(
                            segment_name, num_pts, avg_m, avg_s, avg_min, avg_max)
                
                traces1_json = json.dumps(traces1)
                traces2_json = json.dumps(traces2)
                traces3_json = json.dumps(traces3)
                traces4_json = json.dumps(traces4)
                traces5_json = json.dumps(traces5)
                traces6_json = json.dumps(traces6)
                
                # Convert shaded regions to JSON for JavaScript
                shaded_regions_plot2_json = json.dumps(shaded_regions_plot2)
                shaded_regions_plot3_json = json.dumps(shaded_regions_plot3)
                
                html_str = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Elevation Profile Analysis</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
.container { max-width: 1400px; margin: auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
h1 { text-align: center; color: #333; }
h2 { text-align: center; color: #555; margin-top: 40px; }
h3 { color: #666; margin-top: 30px; border-bottom: 2px solid #ddd; padding-bottom: 10px; }
.plot-container { width: 100%; height: 500px; margin-bottom: 30px; }
table { width: 80%; margin: 20px auto; border-collapse: collapse; }
th, td { padding: 12px; border: 1px solid #ddd; text-align: center; }
th { background: #f0f0f0; }
.legend-box { display: flex; justify-content: center; gap: 30px; margin: 20px 0; }
.legend-item { display: flex; align-items: center; gap: 8px; }
.legend-color { width: 25px; height: 15px; border-radius: 3px; }
.info-box { background: #f9f9f9; padding: 15px; border-radius: 5px; margin: 20px 0; }
.info-box p { margin: 5px 0; }
.section-divider { border-top: 3px solid #007bff; margin: 50px 0; padding-top: 20px; }
</style>
</head>
<body>
<div class="container">
<h1>Elevation Profile Analysis</h1>
<p style="text-align:center;color:#666">Flow Direction: Upstream (Before) --&gt; Check Dam (Within) --&gt; Downstream (After)</p>

<div class="info-box">
<p><strong>Run Timestamp:</strong> """ + timestamp + """</p>
<p><strong>Parameters:</strong> Padding=""" + str(padding_distance) + """m, Line Interval=""" + str(line_interval) + """m, Sample Points=""" + str(num_sample_points) + """</p>
<p><strong>Line Counts:</strong> Before=""" + str(num_before) + """, Within=""" + str(num_within) + """, After=""" + str(num_after) + """ (Total=""" + str(total_lines) + """)</p>
</div>

<div class="legend-box">
    <div class="legend-item"><div class="legend-color" style="background:rgba(65,105,225,0.5)"></div><span>Before (Upstream)</span></div>
    <div class="legend-item"><div class="legend-color" style="background:rgba(34,139,34,0.5)"></div><span>Within (Check Dam)</span></div>
    <div class="legend-item"><div class="legend-color" style="background:rgba(220,20,60,0.5)"></div><span>After (Downstream)</span></div>
    <div class="legend-item"><div class="legend-color" style="background:rgba(255,165,0,0.15)"></div><span>Within Lower (Shaded)</span></div>
</div>

<!-- FRONT PROFILE SECTION -->
<div class="section-divider">
<h2>FRONT PROFILE (Cross-Sectional View)</h2>
</div>

<h3>Plot 1: Mean Elevation +/- Standard Deviation (Cross-Sectional)</h3>
<div id="plot1" class="plot-container"></div>

<h3>Plot 2: Maximum Elevation (Cross-Sectional)</h3>
<p style="text-align:center;color:#888;font-size:0.9em;">Shaded regions indicate where Within (Check Dam) is lower than Before (Upstream) or After (Downstream)</p>
<p style="text-align:center;color:#555;font-size:0.95em;"><strong>AUC Differences:</strong> Before - Within = """ + "{:.1f}".format(diff_before_within_p2) + """ | After - Within = """ + "{:.1f}".format(diff_after_within_p2) + """</p>
<div id="plot2" class="plot-container"></div>

<h3>Plot 3: Terrain Envelope (Within: Min, Before/After: Max)</h3>
<p style="text-align:center;color:#888;font-size:0.9em;">Shaded regions indicate where Within (Check Dam) is lower than Before (Upstream) or After (Downstream)</p>
<p style="text-align:center;color:#555;font-size:0.95em;"><strong>AUC Differences:</strong> Before - Within = """ + "{:.1f}".format(diff_before_within_p3) + """ | After - Within = """ + "{:.1f}".format(diff_after_within_p3) + """</p>
<div id="plot3" class="plot-container"></div>

<h3>Front Profile Summary Statistics</h3>
<table>
<tr><th>Segment</th><th>Mean (m)</th><th>Std Dev (m)</th><th>Avg Min (m)</th><th>Avg Max (m)</th></tr>
""" + front_table_rows + """
</table>

<!-- SIDE PROFILE SECTION -->
<div class="section-divider">
<h2>SIDE PROFILE (Longitudinal View - Continuous)</h2>
</div>

<h3>Plot 4: Mean Elevation +/- Standard Deviation (Longitudinal)</h3>
<div id="plot4" class="plot-container"></div>

<h3>Plot 5: Maximum Elevation (Longitudinal)</h3>
<div id="plot5" class="plot-container"></div>

<h3>Plot 6: Minimum Elevation (Longitudinal)</h3>
<div id="plot6" class="plot-container"></div>

<h3>Side Profile Summary Statistics</h3>
<table>
<tr><th>Segment</th><th>Points</th><th>Mean (m)</th><th>Std Dev (m)</th><th>Avg Min (m)</th><th>Avg Max (m)</th></tr>
""" + side_table_rows + """
</table>

</div>

<script>
// Shaded regions data
var shadedRegionsPlot2 = """ + shaded_regions_plot2_json + """;
var shadedRegionsPlot3 = """ + shaded_regions_plot3_json + """;

// Build shapes for shaded regions (very transparent orange)
function buildShadedShapes(regions, yMin, yMax) {
    var shapes = [];
    for (var i = 0; i < regions.length; i++) {
        shapes.push({
            type: 'rect',
            x0: regions[i][0],
            x1: regions[i][1],
            y0: yMin,
            y1: yMax,
            fillcolor: 'rgba(255, 165, 0, 0.08)',
            line: { width: 0 }
        });
    }
    return shapes;
}

// Plot 1: Front Profile - Mean +/- Std
var traces1 = """ + traces1_json + """;
var layout1 = {
    title: 'Cross-Sectional Elevation Profile (Mean +/- Std)',
    xaxis: { title: 'Position Along Cross-Section (%)', range: [0, 100] },
    yaxis: { title: 'Elevation (m)', range: [""" + str(y_axis_min) + """, """ + str(y_axis_max) + """] },
    hovermode: 'closest',
    showlegend: true,
    legend: { x: 0.02, y: 0.98 }
};
Plotly.newPlot('plot1', traces1, layout1);

// Plot 2: Front Profile - MAX with shaded regions
var traces2 = """ + traces2_json + """;
var shapes2 = buildShadedShapes(shadedRegionsPlot2, """ + str(y_axis_min) + """, """ + str(y_axis_max) + """);
var layout2 = {
    title: 'Cross-Sectional Elevation Profile (Maximum)',
    xaxis: { title: 'Position Along Cross-Section (%)', range: [0, 100] },
    yaxis: { title: 'Elevation (m)', range: [""" + str(y_axis_min) + """, """ + str(y_axis_max) + """] },
    hovermode: 'closest',
    showlegend: true,
    legend: { x: 0.02, y: 0.98 },
    shapes: shapes2
};
Plotly.newPlot('plot2', traces2, layout2);

// Plot 3: Front Profile - Within MIN, Before/After MAX with shaded regions
var traces3 = """ + traces3_json + """;
var shapes3 = buildShadedShapes(shadedRegionsPlot3, """ + str(y_axis_min) + """, """ + str(y_axis_max) + """);
var layout3 = {
    title: 'Terrain Envelope (Within: Min, Before/After: Max)',
    xaxis: { title: 'Position Along Cross-Section (%)', range: [0, 100] },
    yaxis: { title: 'Elevation (m)', range: [""" + str(y_axis_min) + """, """ + str(y_axis_max) + """] },
    hovermode: 'closest',
    showlegend: true,
    legend: { x: 0.02, y: 0.98 },
    shapes: shapes3
};
Plotly.newPlot('plot3', traces3, layout3);

// Side profile boundary shapes (vertical dashed lines)
var sideShapes = [
    {
        type: 'line',
        x0: """ + str(boundary_before_within) + """,
        x1: """ + str(boundary_before_within) + """,
        y0: """ + str(y_axis_min) + """,
        y1: """ + str(y_axis_max) + """,
        line: { color: 'gray', width: 2, dash: 'dash' }
    },
    {
        type: 'line',
        x0: """ + str(boundary_within_after) + """,
        x1: """ + str(boundary_within_after) + """,
        y0: """ + str(y_axis_min) + """,
        y1: """ + str(y_axis_max) + """,
        line: { color: 'gray', width: 2, dash: 'dash' }
    }
];

var sideAnnotations = [
    {
        x: """ + str(num_before / 2 + 0.5) + """,
        y: """ + str(y_axis_max - 0.3) + """,
        text: 'Before',
        showarrow: false,
        font: { color: 'rgb(65, 105, 225)', size: 12 }
    },
    {
        x: """ + str(num_before + num_within / 2 + 0.5) + """,
        y: """ + str(y_axis_max - 0.3) + """,
        text: 'Within',
        showarrow: false,
        font: { color: 'rgb(34, 139, 34)', size: 12 }
    },
    {
        x: """ + str(num_before + num_within + num_after / 2 + 0.5) + """,
        y: """ + str(y_axis_max - 0.3) + """,
        text: 'After',
        showarrow: false,
        font: { color: 'rgb(220, 20, 60)', size: 12 }
    }
];

// Plot 4: Side Profile - Mean +/- Std (Continuous)
var traces4 = """ + traces4_json + """;
var layout4 = {
    title: 'Longitudinal Elevation Profile (Mean +/- Std)',
    xaxis: { title: 'Line Number', range: [0.5, """ + str(total_lines + 0.5) + """], dtick: Math.ceil(""" + str(total_lines) + """ / 20) },
    yaxis: { title: 'Elevation (m)', range: [""" + str(y_axis_min) + """, """ + str(y_axis_max) + """] },
    hovermode: 'closest',
    showlegend: true,
    legend: { x: 0.02, y: 0.98 },
    shapes: sideShapes,
    annotations: sideAnnotations
};
Plotly.newPlot('plot4', traces4, layout4);

// Plot 5: Side Profile - MAX (Continuous)
var traces5 = """ + traces5_json + """;
var layout5 = {
    title: 'Longitudinal Elevation Profile (Maximum)',
    xaxis: { title: 'Line Number', range: [0.5, """ + str(total_lines + 0.5) + """], dtick: Math.ceil(""" + str(total_lines) + """ / 20) },
    yaxis: { title: 'Elevation (m)', range: [""" + str(y_axis_min) + """, """ + str(y_axis_max) + """] },
    hovermode: 'closest',
    showlegend: true,
    legend: { x: 0.02, y: 0.98 },
    shapes: sideShapes,
    annotations: sideAnnotations
};
Plotly.newPlot('plot5', traces5, layout5);

// Plot 6: Side Profile - MIN (Continuous)
var traces6 = """ + traces6_json + """;
var layout6 = {
    title: 'Longitudinal Elevation Profile (Minimum)',
    xaxis: { title: 'Line Number', range: [0.5, """ + str(total_lines + 0.5) + """], dtick: Math.ceil(""" + str(total_lines) + """ / 20) },
    yaxis: { title: 'Elevation (m)', range: [""" + str(y_axis_min) + """, """ + str(y_axis_max) + """] },
    hovermode: 'closest',
    showlegend: true,
    legend: { x: 0.02, y: 0.98 },
    shapes: sideShapes,
    annotations: sideAnnotations
};
Plotly.newPlot('plot6', traces6, layout6);
</script>
</body>
</html>"""
                
                html_path = os.path.join(run_folder, "elevation_profile.html")
                
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(html_str)
                
                arcpy.AddMessage("Saved: elevation_profile.html")
                
                if os.path.exists(html_path):
                    arcpy.AddMessage("HTML verified! Size: {} bytes".format(os.path.getsize(html_path)))
                    
            except Exception as e:
                arcpy.AddError("HTML generation error: {}".format(str(e)))
                import traceback
                arcpy.AddError(traceback.format_exc())
        
        # ============================================
        # FINAL CLEANUP - Ensure no state persists
        # ============================================
        arcpy.AddMessage("-" * 40)
        arcpy.AddMessage("Performing final cleanup...")
        
        # Delete specific temp polygon
        if arcpy.Exists(temp_poly_path):
            try:
                arcpy.management.Delete(temp_poly_path)
                arcpy.AddMessage("Deleted: {}".format(temp_poly_name))
            except:
                pass
        
        # Full memory cleanup
        self.cleanup_memory_workspace()
        
        # Clear all local variables that might hold references
        del front_profile_data
        del side_profile_data
        del front_stats
        del side_stats
        del line_geometries
        del line_positions_dict
        
        # Force garbage collection
        gc.collect()
        
        arcpy.AddMessage("Cleanup complete!")
        
        arcpy.AddMessage("=" * 50)
        arcpy.AddMessage("RUN COMPLETE!")
        arcpy.AddMessage("Output folder: {}".format(run_folder))
        arcpy.AddMessage("Timestamp: {}".format(timestamp))
        arcpy.AddMessage("=" * 50)
        
        # List all files created
        arcpy.AddMessage("Files created:")
        for f in os.listdir(run_folder):
            arcpy.AddMessage("  - {}".format(f))

    def postExecute(self, parameters):
        """Called after execute - perform additional cleanup"""
        try:
            # Additional cleanup after execution
            gc.collect()
        except:
            pass
        return
