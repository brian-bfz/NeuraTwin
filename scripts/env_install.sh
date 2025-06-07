#!/bin/bash

# This script automates the complete setup for the NeuraTwin environment.
# It should be run from the root directory of the NeuraTwin project.
#
# Usage:
# cd /path/to/NeuraTwin
# bash scripts/env_install.sh

set -e # Exit immediately if a command exits with a non-zero status.

ENV_NAME="NeuraTwin"
YAML_FILE="env.yaml" # Using the filename you specified

# STAGE 2: Executed inside the conda environment
if [ "$1" == "--post" ]; then
    echo ""
    echo "======================================================"
    echo "   STAGE 2: RUNNING PROCEDURAL INSTALLS INSIDE ENV    "
    echo "======================================================"
    echo ""
    echo "✅ Now executing inside the '$CONDA_DEFAULT_ENV' environment."
    
    # --- Install PyTorch3D explicitly (Moved from YAML) ---
    echo "🚀 Installing PyTorch3D..."
    pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt240/download.html
    echo "✅ PyTorch3D installation complete."
    echo ""
    
    # --- Install TRELLIS from within the PhysTwin directory ---
    echo "🚀 Installing TRELLIS..."
    if [ ! -d "PhysTwin/data_process/TRELLIS" ]; then
        echo "Cloning TRELLIS repository..."
        git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git PhysTwin/data_process/TRELLIS
    else
        echo "TRELLIS directory already exists. Skipping clone."
    fi
    
    cd PhysTwin/data_process/TRELLIS
    echo "Running TRELLIS setup script..."
    . ./setup.sh --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast
    cd ../../../ # Return to the NeuraTwin root directory
    echo "✅ TRELLIS installation complete."
    echo ""

    # --- Install gaussian_splatting submodules ---
    echo "🚀 Installing gaussian_splatting submodules..."
    cd PhysTwin/gaussian_splatting/
    pip install submodules/diff-gaussian-rasterization/
    pip install submodules/simple-knn/
    cd ../../ # Return to the NeuraTwin root directory
    echo "✅ gaussian_splatting submodules installed."
    
    exit 0
fi


# STAGE 1: Create the base environment
echo ""
echo "======================================================"
echo "    STAGE 1: CREATING CONDA ENV 'NeuraTwin'           "
echo "======================================================"
echo ""

if ! conda env list | grep -q "$ENV_NAME"; then
    conda env create -f "$YAML_FILE"
    if [ $? -ne 0 ]; then
        echo "❌ Error: Conda environment creation failed."
        exit 1
    fi
    echo "✅ Conda environment '$ENV_NAME' created successfully."
else
    echo "✅ Conda environment '$ENV_NAME' already exists. Skipping creation."
fi

# Run Stage 2 of this script inside the conda environment
conda run -n $ENV_NAME bash scripts/env_install.sh --post

echo ""
echo "======================================================"
echo "      SETUP COMPLETE!                             "
echo "======================================================"
echo "✅ The NeuraTwin environment is fully installed."
echo "To activate it in your terminal, run:"
echo ""
echo "conda activate $ENV_NAME"
echo ""