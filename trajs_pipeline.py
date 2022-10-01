from data_handler import (
    get_selected_data,
    load_trajs_data,
    load_trajs_metadata,
    traj_class_builder,
)
from data_parser import parse_data
from pipeline import Pipeline

GEOLIFE_DATASET = 0
LANGEVIN_GENERATED = 1
DIFF_DIFF_GENERATED = 2


def get_traj_extraction_pl(origin: int = GEOLIFE_DATASET) -> Pipeline:
    if origin == GEOLIFE_DATASET:
        return Pipeline(
            "Geolife data loader",
            parse_data,
            load_trajs_metadata,
            load_trajs_data,
            get_selected_data,
            traj_class_builder,
        )
    if origin == LANGEVIN_GENERATED:
        raise NotImplementedError
    if origin == DIFF_DIFF_GENERATED:
        raise NotImplementedError
    raise ValueError("Invalid origin")
