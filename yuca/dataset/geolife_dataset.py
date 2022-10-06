# Dataset metadata
NAME = "geolife"
VERSION = "0.1.0" # See version description in config.py
DOWNLOAD_URL = (
    "https://download.microsoft.com/download/F/4/8/"
    "F4894AA5-FDBC-481E-9285-D5F8C4C4F039/Geolife%20Trajectories%201.3.zip"
)

class GeoLifeDataset(Dataset): 
    
    def __init__(self, redownload: bool = False, reyupify: bool = False):
        super().__init__(NAME, VERSION, redownload, reyupify)
       
    def _fetch(self):
        ...

    def _yupify(self):
        ...