"""
This is not intended to be used directly, but as a template for creating new
datasets.
"""

from typing import Any, List, Tuple

from yupi import Trajectory

from pactus.dataset import Dataset

# Dataset metadata
NAME = "template"  # Name of the dataset
VERSION = "0.1.0"  # See version description in config.py
# Sometimes the dataset needs to be downloaded from a url. If that is the case,
# this is also a good place to put the url.
URL = "https://example.com/dataset.zip"

# The name of the class should be the same as the name of the dataset, followed
# by "Dataset".
class TemplateDataset(Dataset):
    """Template dataset."""

    # Call the super class constructor with dataset info
    # The refetch and reyupify arguments are used to force the dataset to be
    # redownloaded or reconverted to yupi trajectories.
    def __init__(self, refetch: bool = False, reyupify: bool = False):
        super().__init__(NAME, VERSION, refetch, reyupify)

    # 'fetch' is the method you need to override to download the dataset
    # if needed. The self.raw_dir variable is the path to the folder
    # where the dataset should be stored.
    #
    # If the dataset is generated, this method can be left empty.
    # Use the 'yupify' method instead.
    def fetch(self) -> None:
        """Downloads the dataset in case needed"""

    # 'yupify' is the method you need to override to parse or generate the
    # dataset and convert it to yupi trajectories. The method should return
    # a tuple containing a list of yupi trajectories and a list of labels.
    #
    # The labels can be anything, but they must be the same length as the
    # list of trajectories and the values must be storable in a json file.
    # (e.g., strings or numbers usually)
    def yupify(self) -> Tuple[List[Trajectory], List[Any]]:
        """Parses or generates the dataset and convert it to yupi trajectories"""
        raise NotImplementedError
