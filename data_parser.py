"""
Parses the data, saves the trajectories and creates a metadata json file.

A trajectory is a Nx3 matrix, where N is the number of points and the 3
columns are the lat, lon and time (in seconds).

The json metadata file contains a list of dictionaries (the metada of
each trajectory) with the following keys:
    - id: the trajectory id
    - file_path: the path to the trajectory file
    - class: the class of the trajectory
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, NamedTuple, Tuple

import numpy as np

from pipeline import PipelineStep

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

metadata: List[dict] = []


LabelData = NamedTuple(
    "LabelData", [("start_dt", datetime), ("end_dt", datetime), ("clsf", str)]
)


Register = Tuple[float, float, datetime]
"""Register data (GPS point and time) in the form: lat, lon, time"""


def process_usr_trajs(usr_folder: Path) -> None:
    """
    Processes the trajectories of a user.

    Parameters
    ----------
    usr_folder : Path
        The path to the user folder.
    """
    labels_file = usr_folder / "labels.txt"
    if not labels_file.exists():
        return

    labels = load_labels(labels_file)
    all_regs = load_registers(usr_folder)

    logging.info("Processing trajectories")
    dest_folder = Path(f"./trajectories/{usr_folder.name}")
    dest_folder.mkdir(parents=True, exist_ok=True)

    label_idx = 0
    regs_idx = 0
    label = labels[label_idx]
    traj: List[List[float]] = []
    while regs_idx < len(all_regs):
        print(f"{(regs_idx + 1) / len(all_regs):.2%}", end="\r")
        lat, long, reg_dt = all_regs[regs_idx]

        # Register inside label bounds
        if label.start_dt <= reg_dt <= label.end_dt:
            time = (reg_dt - label.start_dt).seconds
            if not traj or time != traj[-1][-1]:
                traj.append([lat, long, time])

        # Register outside label bounds
        elif label.start_dt < reg_dt:
            # Save the trajectory if it has at least 2 points
            if len(traj) > 1:
                traj_id = f"{usr_folder.name}_{label_idx}"
                file_path = str(dest_folder / f"{label_idx}_{label.clsf}.txt")
                npy_traj = np.array(traj)
                np.savetxt(file_path, npy_traj)
                metadata.append(
                    {
                        "id": traj_id,
                        "file_path": file_path,
                        "class": label.clsf,
                        "mean_dt": np.mean(np.diff(npy_traj[:, 2])),
                        "length": len(traj),
                    }
                )
                traj = []

            # Move to the next label
            label_idx += 1
            if label_idx >= len(labels):
                break
            label = labels[label_idx]
            continue  # Don't increment regs_idx yet
        regs_idx += 1


def load_registers(usr_folder: Path) -> List[Register]:
    """
    Loads all the registers of a user.

    Parameters
    ----------
    usr_folder : Path
        The path to the user folder.

    Returns
    -------
    List[Register]
        The list of registers (all together).
    """
    logging.info("Loading registers")
    all_regs = []
    trajs_folder = usr_folder / "Trajectory"
    sorted_paths = sorted(trajs_folder.iterdir())
    for i, plt in enumerate(sorted_paths):
        print(f"{(i + 1) / len(sorted_paths):.2%}", end="\r")
        with open(plt, "r", encoding="utf-8") as reg:
            registers = [parse_register(line) for line in reg.readlines()[6:]]
            all_regs += registers
    return all_regs


def parse_register(line: str) -> Register:
    """
    Parses a register line.

    Parameters
    ----------
    line : str
        The line to parse.

    Returns
    -------
    Register
        The parsed register.
    """
    reg = line.strip().split(",")
    lat, lon = float(reg[0]), float(reg[1])
    _dt = datetime.strptime(f"{reg[5]} {reg[6]}", "%Y-%m-%d %H:%M:%S")
    return lat, lon, _dt


def load_labels(labels_file: Path) -> List[LabelData]:
    """
    Loads the labels of a user.

    Parameters
    ----------
    labels_file : Path
        The path to the labels file.

    Returns
    -------
    List[LabelData]
        The list of labels.
    """
    logging.info("Loading labels")
    with open(labels_file, "r", encoding="utf-8") as l_file:
        labels = [parse_label(line) for line in l_file.readlines()[1:]]
    return labels


def parse_label(line: str) -> LabelData:
    """
    Parses a label line.

    Parameters
    ----------
    line : str
        The line to parse.

    Returns
    -------
    LabelData
        The parsed label.
    """
    items = line.split()
    start_dt = datetime.strptime(f"{items[0]} {items[1]}", "%Y/%m/%d %H:%M:%S")
    end_dt = datetime.strptime(f"{items[2]} {items[3]}", "%Y/%m/%d %H:%M:%S")
    clsf = items[4]
    return LabelData(start_dt, end_dt, clsf)


def _parse_data(dataset_folder: Path) -> str:
    if not dataset_folder.exists():
        raise FileNotFoundError(f"Dataset folder not found. Path: '{dataset_folder}'")

    usr_folders = list(sorted(dataset_folder.iterdir()))
    for i, usr in enumerate(usr_folders):
        logging.info(
            "Processing user: %s - %s", usr.name, f"{(i + 1) / len(usr_folders):.2%}"
        )
        process_usr_trajs(usr)

    logging.info("Saving metadata")
    metadata_file = Path("./trajectories/metadata.json")
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_file, "w", encoding="utf-8") as doc:
        json.dump(metadata, doc, indent=4, ensure_ascii=False)
    logging.info("Done")
    return str(metadata_file)


@PipelineStep.build("geolife dataset parser", cache=True)
def parse_data(_none: None) -> str:
    """
    Parses the GeoLife dataset and saves the trajectories.

    Input: None
    Output: The path to the metadata file.
    """
    assert _none is None, "This step doesn't take any input"
    return _parse_data(Path("./geolife_dataset"))


def main():
    """Main function. Processes all the users."""
    _parse_data(Path("./geolife_dataset"))


if __name__ == "__main__":
    main()
