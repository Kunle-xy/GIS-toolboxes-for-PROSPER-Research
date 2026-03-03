# -*- coding: utf-8 -*-
import arcpy
import os
import plotly.graph_objects as go

class Toolbox:
    def __init__(self):
        self.label = "Terrain Analysis Toolbox"
        self.alias = "TerrainToolbox"
        self.tools = [ProfileTool]

class ProfileTool:
    def __init__(self):
        self.label = "Generate Line Profile Plot"
        self.description = "Creates points along a line, samples elevation from a DEM, and exports a plot."

    def getParameterInfo(self):
        """Define the tool parameters."""
        # Parameter 0: Input Line Feature
        param0 = arcpy.Parameter(
            displayName="Input Line Feature",
            name="in_line",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input")
        param0.filter.list = ["Polyline"]

        # Parameter 1: Input DEM Raster
        param1 = arcpy.Parameter(
            displayName="Input DEM Raster",
            name="in_dem",
            datatype="GPRasterLayer",
            parameterType="Required",
            direction="Input")

        # Parameter 2: Percentage Spacing
        param2 = arcpy.Parameter(
            displayName="Percentage Spacing (0.1-100)",
            name="spacing_pct",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input")
        param2.value = 0.1

        # Parameter 3: Output HTML Plot File
        param3 = arcpy.Parameter(
            displayName="Output HTML Plot File",
            name="out_html",
            datatype="DEFile",
            parameterType="Required",
            direction="Output")
        param3.filter.list = ["html"]

        return [param0, param1, param2, param3]

    def execute(self, parameters, messages):
        """The source code of the tool."""
        in_line = parameters[0].valueAsText
        in_dem = parameters[1].valueAsText
        spacing_pct = parameters[2].value
        out_html = parameters[3].valueAsText

        # Use memory workspace for intermediate data
        temp_points = "memory\\sampled_points"
        
        # 1. Generate points along lines based on percentage spacing
        arcpy.management.GeneratePointsAlongLines(
            Input_Features=in_line,
            Output_Feature_Class=temp_points,
            Point_Placement="PERCENTAGE",
            Percentage=spacing_pct
        )
        arcpy.AddMessage("Points generated along line.")

        # 2. Sample DEM values at the points
        # Sample creates a table; we use the 'Sample' tool from Image Analyst/Spatial Analyst
        temp_table = "memory\\elevation_table"
        arcpy.sa.Sample(
            in_rasters=[in_dem],
            in_location_data=temp_points,
            out_table=temp_table,
            resampling_type="NEAREST"
        )
        arcpy.AddMessage("DEM sampled at point locations.")

        # 3. Extract Data for Plotting
        x_vals = [] # Distance/Order
        y_vals = [] # Elevation
        
        # Identify the elevation field name (usually matches the raster name or 'Band_1')
        fields = [f.name for f in arcpy.ListFields(temp_table)]
        elev_field = fields[-1] # Simple heuristic: last field added by Sample

        with arcpy.da.SearchCursor(temp_table, ["OID@", elev_field]) as cursor:
            for row in cursor:
                x_vals.append(row[0])
                y_vals.append(row[1])

        # 4. Generate Interactive HTML Plot using Plotly
        self.create_html_plot(out_html, x_vals, y_vals)
        arcpy.AddMessage(f"Plot saved to: {out_html}")

    def create_html_plot(self, file_path, x_data, y_data):
        """Creates an interactive HTML plot using Plotly."""
        if not y_data:
            return
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode='lines',
            name='Elevation',
            line=dict(color='blue', width=2),
            hovertemplate='Point: %{x}<br>Elevation: %{y:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Elevation Profile',
            xaxis_title='Sample Point Sequence',
            yaxis_title='Elevation',
            template='plotly_white',
            hovermode='x unified'
        )
        
        fig.write_html(file_path, include_plotlyjs=True, full_html=True)
