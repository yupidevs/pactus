"""
General configurations
"""
import os
from pathlib import Path

# -----------------------------------------------------------------------------
# Download configs
# -----------------------------------------------------------------------------
DOWNLOAD_CHUNCK_SIZE = 4096
PROGRESS_BAR_LENGTH = 50

# -----------------------------------------------------------------------------
# Dataset configs
# -----------------------------------------------------------------------------
#
# Dataset version: (doens't need to match the project version)
#   [major].[minor].[patch]
#      │       │       └── Change when minors changes are made in the code
#      │       │           related to the datasets. (no need for redownload
#      │       │           or regeneration of yupi trajectories)
#      │       │
#      │       └── Change when the yupi data changes (needs to be regenerated)
#      │
#      └── Change when the original dataset changes (needs redownload)
#
# Dataset structure: (mainly for downloadable datasets)
#  src
#  └── datasets
#      └── [dataset_name]
#          ├── raw_data
#          │   └── [unzipped dataset files]
#          ├── yupi_data
#          │   └── [serialized yupi trajectories]
#          └── metadata.json
#
CACHE_PATH = (
    str(Path(__file__).parent / ".cache")
    if os.environ["YUCA_CACHE_PATH"] == ""
    else os.environ["YUCA_CACHE_PATH"]
)
DS_BASE_DIR = CACHE_PATH + "/datasets"
DS_DIR = DS_BASE_DIR + "/{0}"
DS_METADATA_FILE = DS_DIR + "/metadata.json"
DS_RAW_DIR = DS_DIR + "/raw_data"
DS_YUPI_DIR = DS_DIR + "/yupi_data"
DS_FEATS_DIR = DS_DIR + "/features"
DS_EVALS_DIR = DS_DIR + "/evaluations"
