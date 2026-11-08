"""
General configurations
"""

import os
from pathlib import Path

# -----------------------------------------------------------------------------
# Download configs
# -----------------------------------------------------------------------------
DOWNLOAD_CHUNCK_SIZE = 4096
DOWNLOAD_TIMEOUT_SECONDS = 60
PROGRESS_BAR_LENGTH = 50

# -----------------------------------------------------------------------------
# Dataset configs
# -----------------------------------------------------------------------------
#
CACHE_PATH = os.environ.get("PACTUS_CACHE_PATH", str(Path(__file__).parent / ".cache"))
DS_BASE_DIR = CACHE_PATH + "/datasets"
DS_DIR = DS_BASE_DIR + "/{0}"
DS_FEATS_DIR = DS_DIR + "/features"
DS_EVALS_DIR = DS_DIR + "/evaluations"

# -----------------------------------------------------------------------------
# Model configs
# -----------------------------------------------------------------------------
MASK_VALUE = -10_000
