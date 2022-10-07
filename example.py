import logging

from yupi import DiffMethod, Trajectory, WindowType
from yuca import LangevinDataset, RandomForestModel, Featurizer, features

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)

Trajectory.global_diff_method(
    method=DiffMethod.LINEAR_DIFF, window_type=WindowType.FORWARD
)


# Load Dataset
dataset = LangevinDataset()

# Select the desired features to be extracted from the trajectories
featurizer = Featurizer(selected=features.ALL)

# Defining models
model = RandomForestModel(
    featurizer=featurizer
    max_features = 16
    n_estimators = 200
    bootstrap = False,
    random_state = None,
    warm_start = True,
    n_jobs = 6,
)

# Spliting dataset
train, test = dataset.split(0.8)

# Train the model
model.train(data=train, cross_validation=5)

# Evaluate the model on a test dataset
evaluation = model.evaluate(test)

# Print the evaluation
evaluation.show()



# import yuca.features as feats
# from yuca.feat_pipeline import get_feat_extraction_pl
# from yuca.ml_pipeline import get_ml_classifier_pl
# from yuca.pipeline import Pipeline
# from yuca.trajs_pipeline import get_traj_extraction_pl
# from yuca.transf_pipeline import get_transformer_classifier_pl


# FEATURES = feats.ALL_FEATS
# STOP_RATE_THRESHOLD = 5
# VEL_CHANGE_RATE_THRESHOLD = 3

# feat_pl = Pipeline(
#     "Main",
#     get_traj_extraction_pl(3),
#     get_feat_extraction_pl(
#         features=FEATURES,
#         stop_rate_threshold=STOP_RATE_THRESHOLD,
#         vel_change_rate_threshold=VEL_CHANGE_RATE_THRESHOLD,
#     ),
#     get_ml_classifier_pl(
#         method=0,
#         test_size=0.3,
#         random_state=0,
#     ),
# )

# tranf_pl = Pipeline(
#     "Main transf",
#     get_traj_extraction_pl(3),
#     get_transformer_classifier_pl(),
# )

# tranf_pl.show()
# tranf_pl.run()
