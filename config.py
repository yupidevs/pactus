"""
General configurations
"""

__version__ = "0.1.0"

# -----------------------------------------------------------------------------
#  Download urls
# -----------------------------------------------------------------------------
GEOLIFE_DATASET_URL = (
    "https://download.microsoft.com/download/F/4/8/"
    "F4894AA5-FDBC-481E-9285-D5F8C4C4F039/Geolife%20Trajectories%201.3.zip"
)
MNIST_DATASET_URL = "http://yann.lecun.com/exdb/mnist/"

# -----------------------------------------------------------------------------
# Download configs
# -----------------------------------------------------------------------------
DOWNLOAD_CHUNCK_SIZE = 4096
PROGRESS_BAR_LENGTH = 50

# -----------------------------------------------------------------------------
# Dataset configs
# -----------------------------------------------------------------------------
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
DS_VERSION = "v1"
DS_BASE_DIR = "datasets"
DS_DIR = DS_BASE_DIR + "/{0}"
DS_METADATA_FILE = DS_DIR + "/metadata.json"
DS_RAW_DIR = DS_DIR + "/raw_data"
DS_YUPI_DIR = DS_DIR + "/yupi_data"
