# -*- coding: utf-8 -*-

import os
import numpy as np
import arcpy
from arcpy.sa import Raster, Con, IsNull, SquareRoot, ExtractByMask, Float

arcpy.CheckOutExtension("Spatial")


class Toolbox:
    def __init__(self):
        self.label = "Vegetation Index Toolbox"
        self.alias = "vegindex"
        self.tools = [VegetationIndexTool]


class VegetationIndexTool:
    def __init__(self):
        self.label = "Vegetation Index from MS Ortho"
        self.description = (
            "Compute NDVI, SAVI, or MSAVI from a multispectral ortho "
            "within a boundary polygon. Outputs both the vegetation index "
            "raster and a binary classification raster based on threshold."
        )

    def getParameterInfo(self):
        # Input multispectral raster
        ms_raster = arcpy.Parameter(
            displayName="Multispectral Ortho Raster",
            name="ms_raster",
            datatype="Raster Layer",
            parameterType="Required",
            direction="Input"
        )

        # Red band selection (dynamically populated)
        red_band = arcpy.Parameter(
            displayName="Red Band",
            name="red_band",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        red_band.filter.type = "ValueList"
        red_band.filter.list = []
        red_band.enabled = False

        # NIR band selection (dynamically populated)
        nir_band = arcpy.Parameter(
            displayName="NIR Band",
            name="nir_band",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        nir_band.filter.type = "ValueList"
        nir_band.filter.list = []
        nir_band.enabled = False

        # Boundary polygon
        boundary = arcpy.Parameter(
            displayName="Boundary Polygon",
            name="boundary_polygon",
            datatype="Feature Layer",
            parameterType="Required",
            direction="Input"
        )

        # Vegetation index method
        method = arcpy.Parameter(
            displayName="Vegetation Index Method",
            name="veg_method",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        method.filter.type = "ValueList"
        method.filter.list = ["NDVI", "SAVI", "MSAVI"]

        # Vegetation threshold for binary classification
        threshold = arcpy.Parameter(
            displayName="Vegetation Threshold",
            name="veg_threshold",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input"
        )
        threshold.value = 0.3  # Default threshold

        # Output raster (base name - will generate _index and _classified)
        output = arcpy.Parameter(
            displayName="Output Raster (Base Name)",
            name="output_raster",
            datatype="Raster Dataset",
            parameterType="Required",
            direction="Output"
        )

        return [ms_raster, red_band, nir_band, boundary, method, threshold, output]

    def isLicensed(self):
        try:
            if arcpy.CheckExtension("Spatial") != "Available":
                return False
            return True
        except Exception:
            return False

    def updateParameters(self, parameters):
        """Dynamically populate band dropdowns based on input raster."""
        ms_raster_param = parameters[0]
        red_band_param = parameters[1]
        nir_band_param = parameters[2]

        if ms_raster_param.valueAsText and ms_raster_param.altered:
            try:
                # Get full catalog path using Describe
                desc = arcpy.Describe(ms_raster_param.valueAsText)
                full_path = desc.catalogPath

                # Get raster properties
                raster_obj = arcpy.Raster(full_path)
                band_count = raster_obj.bandCount

                # Build list of band names
                band_list = [f"Band_{i}" for i in range(1, band_count + 1)]

                # Populate both dropdowns
                red_band_param.filter.list = band_list
                nir_band_param.filter.list = band_list
                red_band_param.enabled = True
                nir_band_param.enabled = True

            except Exception:
                # If raster can't be read, disable band selection
                red_band_param.filter.list = []
                nir_band_param.filter.list = []
                red_band_param.enabled = False
                nir_band_param.enabled = False
        else:
            # No raster selected, disable band dropdowns
            if not ms_raster_param.valueAsText:
                red_band_param.filter.list = []
                nir_band_param.filter.list = []
                red_band_param.enabled = False
                nir_band_param.enabled = False

        return

    def updateMessages(self, parameters):
        """Validate that Red and NIR bands are different."""
        red_band_param = parameters[1]
        nir_band_param = parameters[2]
        threshold_param = parameters[5]

        if (red_band_param.valueAsText and nir_band_param.valueAsText and
                red_band_param.valueAsText == nir_band_param.valueAsText):
            nir_band_param.setErrorMessage("NIR band must be different from Red band.")

        # Validate threshold is within reasonable range
        if threshold_param.value is not None:
            if threshold_param.value < -1 or threshold_param.value > 1:
                threshold_param.setWarningMessage(
                    "Threshold is typically between -1 and 1 for vegetation indices."
                )

        return

    def execute(self, parameters, messages):
        ms_raster_input = parameters[0].valueAsText
        red_band_name = parameters[1].valueAsText
        nir_band_name = parameters[2].valueAsText
        boundary = parameters[3].valueAsText
        method = parameters[4].valueAsText
        threshold = parameters[5].value
        output_raster = parameters[6].valueAsText

        # Set environment
        arcpy.env.overwriteOutput = True

        # Get full catalog path using Describe
        desc = arcpy.Describe(ms_raster_input)
        ms_raster_path = desc.catalogPath

        messages.addMessage(f"Raster full path: {ms_raster_path}")

        # Extract band numbers from "Band_X" strings
        red_band_num = int(red_band_name.split("_")[1])
        nir_band_num = int(nir_band_name.split("_")[1])

        messages.addMessage(f"Red band number: {red_band_num}")
        messages.addMessage(f"NIR band number: {nir_band_num}")
        messages.addMessage(f"Vegetation threshold: {threshold}")

        # Create single-band raster layers using MakeRasterLayer
        red_layer_name = "red_band_layer"
        nir_layer_name = "nir_band_layer"

        arcpy.management.MakeRasterLayer(
            ms_raster_path,
            red_layer_name,
            band_index=red_band_num
        )
        arcpy.management.MakeRasterLayer(
            ms_raster_path,
            nir_layer_name,
            band_index=nir_band_num
        )

        red = Raster(red_layer_name)
        nir = Raster(nir_layer_name)

        # Clip bands to boundary polygon
        messages.addMessage("Extracting bands by mask...")
        red_clip = ExtractByMask(red, boundary)
        nir_clip = ExtractByMask(nir, boundary)

        # Convert to float for proper division
        red_float = Float(red_clip)
        nir_float = Float(nir_clip)

        # Create valid pixel mask
        valid_pixels = (~IsNull(red_float)) & (~IsNull(nir_float))

        # Avoid division by zero
        denominator_check = (nir_float + red_float) != 0

        messages.addMessage(f"Computing {method}...")

        # Vegetation index computation
        if method == "NDVI":
            # NDVI = (NIR - Red) / (NIR + Red)
            veg_index = Con(
                denominator_check,
                (nir_float - red_float) / (nir_float + red_float),
                0
            )

        elif method == "SAVI":
            # SAVI = ((NIR - Red) * (1 + L)) / (NIR + Red + L), L = 0.5
            L = 0.5
            denom = nir_float + red_float + L
            denom_valid = denom != 0
            veg_index = Con(
                denom_valid,
                ((nir_float - red_float) * (1 + L)) / denom,
                0
            )

        elif method == "MSAVI":
            # MSAVI = (2 * NIR + 1 - sqrt((2 * NIR + 1)^2 - 8 * (NIR - Red))) / 2
            inner_term = (2 * nir_float + 1) ** 2 - 8 * (nir_float - red_float)
            inner_term_safe = Con(inner_term >= 0, inner_term, 0)
            veg_index = (2 * nir_float + 1 - SquareRoot(inner_term_safe)) / 2

        # Apply valid pixel mask to vegetation index
        final_raster = Con(valid_pixels, veg_index)

        # Create binary classification raster
        # 1 = vegetation (>= threshold), 0 = not vegetation (< threshold)
        messages.addMessage("Creating binary classification raster...")
        binary_raster = Con(valid_pixels, Con(veg_index >= threshold, 1, 0))

        # Generate output paths from single output parameter
        base_path, extension = os.path.splitext(output_raster)
        
        # Handle geodatabase rasters (no extension)
        if not extension:
            index_output = f"{base_path}_index"
            classified_output = f"{base_path}_classified"
        else:
            index_output = f"{base_path}_index{extension}"
            classified_output = f"{base_path}_classified{extension}"

        # Save both rasters
        messages.addMessage("Saving output rasters...")
        
        final_raster.save(index_output)
        messages.addMessage(f"Vegetation index raster saved: {index_output}")
        
        binary_raster.save(classified_output)
        messages.addMessage(f"Binary classification raster saved: {classified_output}")

        # Build raster attribute table for classified raster
        messages.addMessage("Building raster attribute table...")
        arcpy.management.BuildRasterAttributeTable(classified_output, "Overwrite")

        # Calculate vegetation statistics from binary raster
        messages.addMessage("Calculating vegetation statistics...")
        
        # Convert binary raster to numpy array for pixel counting
        # Use nodata_to_value to identify nodata pixels
        binary_array = arcpy.RasterToNumPyArray(
            binary_raster, 
            nodata_to_value=-9999
        )
        
        # Count pixels
        total_pixels = binary_array.size
        nodata_pixels = np.sum(binary_array == -9999)
        total_valid = total_pixels - nodata_pixels
        veg_pixels = np.sum(binary_array == 1)
        non_veg_pixels = np.sum(binary_array == 0)
        
        # Calculate percentages
        if total_valid > 0:
            veg_percent = (veg_pixels / total_valid) * 100
            non_veg_percent = (non_veg_pixels / total_valid) * 100
        else:
            veg_percent = 0
            non_veg_percent = 0

        # Add Percent and ClassName fields to the raster attribute table
        messages.addMessage("Adding percentage and class name to raster attribute table...")
        arcpy.management.AddField(classified_output, "Percent", "DOUBLE")
        arcpy.management.AddField(classified_output, "ClassName", "TEXT", field_length=50)
        
        # Update the attribute table with percentages and class names
        with arcpy.da.UpdateCursor(classified_output, ["Value", "Percent", "ClassName"]) as cursor:
            for row in cursor:
                if row[0] == 0:
                    row[1] = non_veg_percent
                    row[2] = "Non-Vegetation"
                elif row[0] == 1:
                    row[1] = veg_percent
                    row[2] = "Vegetation"
                cursor.updateRow(row)

        # Get cell size for area calculation
        cell_size_x = binary_raster.meanCellWidth
        cell_size_y = binary_raster.meanCellHeight
        cell_area = cell_size_x * cell_size_y  # square units (likely sq meters)
        
        # Calculate areas
        veg_area = veg_pixels * cell_area
        non_veg_area = non_veg_pixels * cell_area
        total_area = total_valid * cell_area

        # Clean up temporary layers
        arcpy.management.Delete(red_layer_name)
        arcpy.management.Delete(nir_layer_name)

        # Display results summary
        messages.addMessage("")
        messages.addMessage("=" * 60)
        messages.addMessage("                 CLASSIFICATION SUMMARY")
        messages.addMessage("=" * 60)
        messages.addMessage(f"  Method:                    {method}")
        messages.addMessage(f"  Threshold:                 {threshold}")
        messages.addMessage("-" * 60)
        messages.addMessage("  PIXEL COUNTS:")
        messages.addMessage(f"    Total valid pixels:      {total_valid:,}")
        messages.addMessage(f"    Vegetation pixels:       {veg_pixels:,}")
        messages.addMessage(f"    Non-vegetation pixels:   {non_veg_pixels:,}")
        messages.addMessage("-" * 60)
        messages.addMessage("  PERCENTAGES:")
        messages.addMessage(f"    Vegetation:              {veg_percent:.2f}%")
        messages.addMessage(f"    Non-vegetation:          {non_veg_percent:.2f}%")
        messages.addMessage("-" * 60)
        messages.addMessage("  AREA (map units squared):")
        messages.addMessage(f"    Cell size:               {cell_size_x:.4f} x {cell_size_y:.4f}")
        messages.addMessage(f"    Total area:              {total_area:,.2f}")
        messages.addMessage(f"    Vegetation area:         {veg_area:,.2f}")
        messages.addMessage(f"    Non-vegetation area:     {non_veg_area:,.2f}")
        messages.addMessage("=" * 60)
        messages.addMessage("")
        messages.addMessage("OUTPUT FILES:")
        messages.addMessage(f"  Index raster:        {index_output}")
        messages.addMessage(f"  Classification:      {classified_output}")
        messages.addMessage("=" * 60)

        return

    def postExecute(self, parameters):
        return
