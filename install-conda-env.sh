#!/bin/bash
# install-conda-python-deps.sh

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
CONDA_DIR="$HOME/miniconda"
ENV_NAME="ocr-dataset-builder"
PYTHON_VERSION="3.11"
MINICONDA_INSTALL_SCRIPT="miniconda.sh"
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"

# --- Miniconda Installation ---
# Check if conda command exists, if not, install Miniconda
if ! command -v conda &> /dev/null; then
    echo "⬇️ Conda not found. Downloading Miniconda..."
    wget --quiet "$MINICONDA_URL" -O "$MINICONDA_INSTALL_SCRIPT"

    echo "📦 Installing Miniconda to $CONDA_DIR..."
    /bin/bash "$MINICONDA_INSTALL_SCRIPT" -b -p "$CONDA_DIR"

    echo "🧹 Cleaning up Miniconda installer..."
    rm "$MINICONDA_INSTALL_SCRIPT"

    # --- Add Conda to PATH for this script's execution ---
    # Note: For persistent PATH changes, you'd modify shell config files like .bashrc
    export PATH="$CONDA_DIR/bin:$PATH"
    echo " PATH updated for script execution."

    # --- Conda Initialization for Shells ---
    echo "⚙️ Initializing Conda for bash..."
    conda init bash

    # Check if fish shell exists and initialize if it does
    if command -v fish &> /dev/null; then
        echo "⚙️ Fish shell detected. Initializing Conda for fish..."
        conda init fish
    else
        echo "🐟 Fish shell not found, skipping Conda initialization for fish."
    fi
else
    echo "✅ Conda is already installed."
    # Ensure Conda is in PATH even if pre-installed
    if [[ ":$PATH:" != *":$CONDA_DIR/bin:"* ]]; then
        export PATH="$CONDA_DIR/bin:$PATH"
        echo " PATH updated for script execution (pre-installed conda)."
    fi
fi

# --- Conda Environment Creation & UV Installation ---
echo "🐍 Checking/Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
# Check if environment exists
if conda env list | grep -q "^$ENV_NAME\\s"; then
    echo " Env '$ENV_NAME' already exists. Updating..."
    # Update environment with Python version and ffmpeg if it exists
    conda install -n "$ENV_NAME" python="$PYTHON_VERSION" ffmpeg -c conda-forge -y
else
    # Create environment with Python and ffmpeg
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" ffmpeg -c conda-forge -y
fi

# Install uv using pip within the environment
echo "✨ Installing uv package manager using pip into '$ENV_NAME'..."
conda run -n "$ENV_NAME" pip install uv

# --- Project Dependency Installation ---
echo "🚀 Installing project dependencies using uv from pyproject.toml into '$ENV_NAME'..."
# Ensure pyproject.toml is in the current directory where this script is run
if [ ! -f pyproject.toml ]; then
    echo "❌ Error: pyproject.toml not found in the current directory."
    exit 1
fi
# Install core, develop, and test dependencies
conda run -n "$ENV_NAME" uv pip install .[dev,test]

echo "✅ Installation/Update complete for environment '$ENV_NAME'."
echo "   To activate the environment, run:"
echo "   conda activate $ENV_NAME"
echo ""
echo "   If the 'conda activate' command is not found after a fresh Miniconda install,"
echo "   you may need to close and reopen your terminal or source your shell config file first:"
echo "     - For bash: source ~/.bashrc"
echo "     - For fish: source ~/.config/fish/config.fish" 