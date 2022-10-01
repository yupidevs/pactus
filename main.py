import features as feats
from feat_pipeline import get_feat_extraction_pl
from ml_pipeline import get_ml_classifier_pl
from pipeline import Pipeline
from trajs_pipeline import get_traj_extraction_pl

FEATURES = feats.ALL_FEATS
STOP_RATE_THRESHOLD = 1
VEL_CHANGE_RATE_THRESHOLD = 1

main_pl = Pipeline(
    "Main",
    get_traj_extraction_pl(),
    get_feat_extraction_pl(
        features=FEATURES,
        stop_rate_threshold=STOP_RATE_THRESHOLD,
        vel_change_rate_threshold=VEL_CHANGE_RATE_THRESHOLD,
    ),
    get_ml_classifier_pl(),
)

main_pl.show(verbose=False)
