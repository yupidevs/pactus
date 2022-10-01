"""
This contains necessary functions to handle the data.
"""
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
    data = [traj_md for traj_md in metadata if traj_md["class"] in classes]
    final_data = []
    for traj_md in data:
        # Filter trajs: dt <= 3s and len >= 100
        if traj_md["mean_dt"] > 3 or traj_md["length"] < 100:
            continue

        # Join similar classes
        if traj_md["class"] == "taxi":
            traj_md["class"] = "car"
        if traj_md["class"] == "subway":
            traj_md["class"] = "train"
        final_data.append(traj_md)
    final_data = load_trajs_data(final_data)
    return final_data


@PipelineStep.build("traj-class builder")
def traj_class_builder(metadata: list[dict]) -> tuple[list[Trajectory], list[str]]:
    """
    Extracts the trajectories and their classes from the metadata
    dictionary. This prepares the data for the feature extraction
    process.

    Input: metadata dictionary
    Output: list of traj-class tuples
    """
    return [d["traj_data"] for d in metadata], [d["class"] for d in metadata]
