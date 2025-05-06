# Use a slim Python 3.11 base image
FROM python:3.11-slim-bookworm

# Set non-interactive frontend for apt-get to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install basic dependencies (git for potential VCS needs, build-essential for packages, wget/curl for downloads)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    build-essential \
    wget \
    curl \
    ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda (needed for the environment setup even if not using GPU base)
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh

# Add conda to PATH
ENV PATH=$CONDA_DIR/bin:$PATH

# Create the conda environment, install ffmpeg, and install uv via pip
ENV ENV_NAME=ocr-dataset-builder
RUN conda create -n $ENV_NAME python=3.11 -y && \
    echo "conda activate $ENV_NAME" >> ~/.bashrc && \
    conda run -n $ENV_NAME conda install -c conda-forge ffmpeg -y && \
    conda run -n $ENV_NAME pip install uv

# Copy just the files needed for dependency installation first
COPY pyproject.toml README.md ./

# Install project dependencies using uv within the environment
# This leverages Docker layer caching if only source code changes later
RUN conda run -n $ENV_NAME uv pip install .[develop,test] # Install core, develop, and test dependencies

# Copy the rest of the application code
COPY . .

# Make RUN commands use the conda environment
SHELL ["conda", "run", "-n", "ocr-dataset-builder", "/bin/bash", "-c"]

# Ensure the main package is importable
ENV PYTHONPATH=/app

# Set the default command to run tests (example)
# You might change this or override it when running the container
# CMD ["poe", "test"]

# Default command: Run the main script (adjust if needed)
# CMD ["python", "youtube_collector/main.py"] # Assuming main.py moves here
CMD ["/bin/bash"] # Default to bash for interactive use

# Optional: Expose ports if needed later (e.g., for a web service)
# EXPOSE 8000 