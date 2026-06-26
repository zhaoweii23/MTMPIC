#!/bin/bash

# Input directory (contains .smi files)
INPUT_DIR="ligprep_input"
# Output directory (generated .sdf files will be saved here)
OUTPUT_DIR="SDF"

# Create output directory if it does not exist
mkdir -p "$OUTPUT_DIR"

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' does not exist."
    exit 1
fi

# Iterate over all .smi files in the input directory
for smi_file in "$INPUT_DIR"/*.smi; do
    # If no .smi files are found, the literal string "$INPUT_DIR/*.smi" remains
    if [ ! -f "$smi_file" ]; then
        echo "No .smi files found in $INPUT_DIR"
        break
    fi

    # Extract the base filename without path and extension
    base_name=$(basename "$smi_file" .smi)
    # Output SDF file path
    sdf_file="$OUTPUT_DIR/$base_name.sdf"

    echo "Processing: $smi_file -> $sdf_file"
    # Run LigPrep conversion
    ligprep -ismi "$smi_file" -osd "$sdf_file"

    # Check the exit status of the command
    if [ $? -ne 0 ]; then
        echo "Warning: LigPrep failed for $smi_file" >&2
    fi
done

echo "Batch processing completed."
