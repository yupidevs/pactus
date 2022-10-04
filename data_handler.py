import json

import numpy as np
from yupi import Trajectory

from pipeline import PipelineStep


@PipelineStep.build("metadata loader")
def load_trajs_metadata(metadata_file: str) -> list[dict]:
    """
    Loads all the data from the metadata file.

    Input: metada file path.
    Output: list of dicts containing the metadata info.
    """
    with open(metadata_file, "r", encoding="utf-8") as md_file:
        metadata: list[dict] = json.load(md_file)
    return metadata


@PipelineStep.build("trajectory loader")
def load_trajs_data(metadata: list[dict]) -> list[dict]:
    """
    Loads all the data from the trajectories file.

    Input: metadata dict.
    Output: metada dict with the trajectries loaded.
    """
    for i, traj_md in enumerate(metadata):
        print(f"{(i+1)/len(metadata):.2%}", end="\r")
        traj = Trajectory(points=np.loadtxt(traj_md["file_path"], ndmin=2))
        traj_md["traj_data"] = traj
    return metadata


@PipelineStep.build("trajectory filter")
def get_selected_data(metadata: list[dict]) -> list[dict]:
    """
    Filters the trajectories to be used

    Input: metadata dict
    Output: metadata dict with the filtered trajectories.
    """
    classes = {"car", "taxi", "bus", "walk", "bike", "subway", "train"}
    filter_data = [
        traj
        for traj in metadata
        if traj["class"] in classes
        if traj["class"] in classes and traj["length"] > 30 and traj["mean_dt"] <= 8
    ]
    final_data = []
    for traj_md in filter_data:
        # Join similar classes
        if traj_md["class"] == "taxi" or traj_md["class"] == "bus":
            traj_md["class"] = "bus-taxi"
        final_data.append(traj_md)
    return final_data


@PipelineStep.build("traj-class builder")
def traj_class_builder(metadata: list[dict]) -> tuple[list[Trajectory], np.ndarray]:
    """
    Extracts the trajectories and their classes from the metadata
    dictionary. This prepares the data for the feature extraction
    process.

    Input: metadata dictionary
    Output: list of traj-class tuples
    """
    return [d["traj_data"] for d in metadata], np.array([d["class"] for d in metadata])
