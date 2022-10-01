import numpy as np
from yupi.trajectory import Trajectory, Vector

FEAT_DICT = {}
__FEAT_VAL = 1


def __new_feat(feat_name: str | None = None) -> int:
    global __FEAT_VAL  # pylint: disable=global-statement
    FEAT_DICT[__FEAT_VAL] = feat_name
    feat_val = __FEAT_VAL
    __FEAT_VAL <<= 1
    return feat_val


def __all_feat_val():
    return __new_feat() - 1


DISTANCE = __new_feat("DISTANCE")
VEL_MEAN = __new_feat("VEL_MEAN")
VEL_MEDIAN = __new_feat("VEL_MEDIAN")
VEL_MIN = __new_feat("VEL_MIN")
VEL_MAX = __new_feat("VEL_MAX")
VEL_STD = __new_feat("VEL_STD")
VEL_VAR = __new_feat("VEL_VAR")
VEL_COEF_VAR = __new_feat("VEL_COEF_VAR")
VEL_IQR = __new_feat("VEL_IQR")
STOP_RATE = __new_feat("STOP_RATE")
VEL_CHANGE_RATE = __new_feat("VEL_CHANGE_RATE")
ACC_MEAN = __new_feat("ACC_MEAN")
ACC_MEDIAN = __new_feat("ACC_MEDIAN")
ACC_MIN = __new_feat("ACC_MIN")
ACC_MAX = __new_feat("ACC_MAX")
ACC_STD = __new_feat("ACC_STD")
ACC_VAR = __new_feat("ACC_VAR")
ACC_COEF_VAR = __new_feat("ACC_COEF_VAR")
ACC_IQR = __new_feat("ACC_IQR")
ACC_CHANGE_RATE_MEAN = __new_feat("ACC_CHANGE_RATE_MEAN")
ACC_CHANGE_RATE_MEDIAN = __new_feat("ACC_CHANGE_RATE_MEDIAN")
ACC_CHANGE_RATE_MIN = __new_feat("ACC_CHANGE_RATE_MIN")
ACC_CHANGE_RATE_MAX = __new_feat("ACC_CHANGE_RATE_MAX")
ACC_CHANGE_RATE_STD = __new_feat("ACC_CHANGE_RATE_STD")
ACC_CHANGE_RATE_VAR = __new_feat("ACC_CHANGE_RATE_VAR")
ACC_CHANGE_RATE_COEF_VAR = __new_feat("ACC_CHANGE_RATE_COEF_VAR")
ACC_CHANGE_RATE_IQR = __new_feat("ACC_CHANGE_RATE_IQR")
ANGLE_MEAN = __new_feat("ANGLE_MEAN")
ANGLE_MEDIAN = __new_feat("ANGLE_MEDIAN")
ANGLE_MIN = __new_feat("ANGLE_MIN")
ANGLE_MAX = __new_feat("ANGLE_MAX")
ANGLE_STD = __new_feat("ANGLE_STD")
ANGLE_VAR = __new_feat("ANGLE_VAR")
ANGLE_COEF_VAR = __new_feat("ANGLE_COEF_VAR")
ANGLE_IQR = __new_feat("ANGLE_IQR")
TURNING_ANGLE_MEAN = __new_feat("TURNING_ANGLE_MEAN")
TURNING_ANGLE_MEDIAN = __new_feat("TURNING_ANGLE_MEDIAN")
TURNING_ANGLE_MIN = __new_feat("TURNING_ANGLE_MIN")
TURNING_ANGLE_MAX = __new_feat("TURNING_ANGLE_MAX")
TURNING_ANGLE_STD = __new_feat("TURNING_ANGLE_STD")
TURNING_ANGLE_VAR = __new_feat("TURNING_ANGLE_VAR")
TURNING_ANGLE_COEF_VAR = __new_feat("TURNING_ANGLE_COEF_VAR")
TURNING_ANGLE_IQR = __new_feat("TURNING_ANGLE_IQR")
TURNING_ANGLE_CHANGE_RATE_MEAN = __new_feat("TURNING_ANGLE_CHANGE_RATE_MEAN")
TURNING_ANGLE_CHANGE_RATE_MEDIAN = __new_feat("TURNING_ANGLE_CHANGE_RATE_MEDIAN")
TURNING_ANGLE_CHANGE_RATE_MIN = __new_feat("TURNING_ANGLE_CHANGE_RATE_MIN")
TURNING_ANGLE_CHANGE_RATE_MAX = __new_feat("TURNING_ANGLE_CHANGE_RATE_MAX")
TURNING_ANGLE_CHANGE_RATE_STD = __new_feat("TURNING_ANGLE_CHANGE_RATE_STD")
TURNING_ANGLE_CHANGE_RATE_VAR = __new_feat("TURNING_ANGLE_CHANGE_RATE_VAR")
TURNING_ANGLE_CHANGE_RATE_COEF_VAR = __new_feat("TURNING_ANGLE_CHANGE_RATE_COEF_VAR")
TURNING_ANGLE_CHANGE_RATE_IQR = __new_feat("TURNING_ANGLE_CHANGE_RATE_IQR")

ALL_FEATS = __all_feat_val()

VEL_FEATS = (
    VEL_MEAN
    | VEL_MEDIAN
    | VEL_MIN
    | VEL_MAX
    | VEL_STD
    | VEL_VAR
    | VEL_COEF_VAR
    | VEL_IQR
    | STOP_RATE
    | VEL_CHANGE_RATE
)

ACC_FEATS = (
    ACC_MEAN
    | ACC_MEDIAN
    | ACC_MIN
    | ACC_MAX
    | ACC_STD
    | ACC_VAR
    | ACC_COEF_VAR
    | ACC_IQR
)

ACC_CHANGE_RATE_FEATS = (
    ACC_CHANGE_RATE_MEAN
    | ACC_CHANGE_RATE_MEDIAN
    | ACC_CHANGE_RATE_MIN
    | ACC_CHANGE_RATE_MAX
    | ACC_CHANGE_RATE_STD
    | ACC_CHANGE_RATE_VAR
    | ACC_CHANGE_RATE_COEF_VAR
    | ACC_CHANGE_RATE_IQR
)

ANGLE_FEATS = (
    ANGLE_MEAN
    | ANGLE_MEDIAN
    | ANGLE_MIN
    | ANGLE_MAX
    | ANGLE_STD
    | ANGLE_VAR
    | ANGLE_COEF_VAR
    | ANGLE_IQR
)

TURNING_ANGLE_FEATS = (
    TURNING_ANGLE_MEAN
    | TURNING_ANGLE_MEDIAN
    | TURNING_ANGLE_MIN
    | TURNING_ANGLE_MAX
    | TURNING_ANGLE_STD
    | TURNING_ANGLE_VAR
    | TURNING_ANGLE_COEF_VAR
    | TURNING_ANGLE_IQR
)

TURNING_ANGLE_CHANGE_RATE_FEATS = (
    TURNING_ANGLE_CHANGE_RATE_MEAN
    | TURNING_ANGLE_CHANGE_RATE_MEDIAN
    | TURNING_ANGLE_CHANGE_RATE_MIN
    | TURNING_ANGLE_CHANGE_RATE_MAX
    | TURNING_ANGLE_CHANGE_RATE_STD
    | TURNING_ANGLE_CHANGE_RATE_VAR
    | TURNING_ANGLE_CHANGE_RATE_COEF_VAR
    | TURNING_ANGLE_CHANGE_RATE_IQR
)

MEAN_FEATS = (
    VEL_MEAN
    | ACC_MEAN
    | ACC_CHANGE_RATE_MEAN
    | ANGLE_MEAN
    | TURNING_ANGLE_MEAN
    | TURNING_ANGLE_CHANGE_RATE_MEAN
)

MEDIAN_FEATS = (
    VEL_MEDIAN
    | ACC_MEDIAN
    | ACC_CHANGE_RATE_MEDIAN
    | ANGLE_MEDIAN
    | TURNING_ANGLE_MEDIAN
    | TURNING_ANGLE_CHANGE_RATE_MEDIAN
)

MIN_FEATS = (
    VEL_MIN
    | ACC_MIN
    | ACC_CHANGE_RATE_MIN
    | ANGLE_MIN
    | TURNING_ANGLE_MIN
    | TURNING_ANGLE_CHANGE_RATE_MIN
)

MAX_FEATS = (
    VEL_MAX
    | ACC_MAX
    | ACC_CHANGE_RATE_MAX
    | ANGLE_MAX
    | TURNING_ANGLE_MAX
    | TURNING_ANGLE_CHANGE_RATE_MAX
)

STD_FEATS = (
    VEL_STD
    | ACC_STD
    | ACC_CHANGE_RATE_STD
    | ANGLE_STD
    | TURNING_ANGLE_STD
    | TURNING_ANGLE_CHANGE_RATE_STD
)

VAR_FEATS = (
    VEL_VAR
    | ACC_VAR
    | ACC_CHANGE_RATE_VAR
    | ANGLE_VAR
    | TURNING_ANGLE_VAR
    | TURNING_ANGLE_CHANGE_RATE_VAR
)

COEF_VAR_FEATS = (
    VEL_COEF_VAR
    | ACC_COEF_VAR
    | ACC_CHANGE_RATE_COEF_VAR
    | ANGLE_COEF_VAR
    | TURNING_ANGLE_COEF_VAR
    | TURNING_ANGLE_CHANGE_RATE_COEF_VAR
)

IQR_FEATS = (
    VEL_IQR
    | ACC_IQR
    | ACC_CHANGE_RATE_IQR
    | ANGLE_IQR
    | TURNING_ANGLE_IQR
    | TURNING_ANGLE_CHANGE_RATE_IQR
)

FEAT_NAMES = list(FEAT_DICT.values())
FEAT_VALUES = list(FEAT_DICT.keys())


def get_feat(traj: Trajectory, feat: int, **kwargs) -> float:
    """Get feature value from trajectory.

    Parameters
    ----------
    traj : Trajectory
        Trajectory.
    feat : int
        Feature.

    Returns
    -------
    float
        Feature value.

    """

    if feat == DISTANCE:
        norm = traj.r.delta.norm
        assert isinstance(norm, Vector)
        return sum(norm)

    if feat == STOP_RATE:
        vel = traj.v.delta.norm
        assert "stop_rate_threshold" in kwargs
        threshold = kwargs["stop_rate_threshold"]
        return np.sum(vel < threshold) / get_feat(traj, DISTANCE)

    if feat == VEL_CHANGE_RATE:
        vel = traj.v.delta.norm
        assert isinstance(vel, Vector)
        subs = np.abs(np.diff(vel))
        vel[vel == 0] = np.inf
        v_rate = subs / vel[:-1]
        assert "vel_change_rate_threshold" in kwargs
        threshold = kwargs["vel_change_rate_threshold"]
        return np.sum(v_rate > threshold) / get_feat(traj, DISTANCE)

    values = None

    if feat & VEL_FEATS:
        values = traj.v.delta.norm
        assert isinstance(values, Vector)
    elif feat & ACC_FEATS:
        values = traj.a.delta.norm
        assert isinstance(values, Vector)
    elif feat & ACC_CHANGE_RATE_FEATS:
        acc = traj.a.delta.norm
        assert isinstance(acc, Vector)
        delta_t = traj.t.delta
        values = np.diff(acc) / delta_t[1:]
    elif feat & ANGLE_FEATS:
        values = traj.turning_angles(accumulate=True)
    elif feat & TURNING_ANGLE_FEATS:
        values = traj.turning_angles(accumulate=False)
    elif feat & TURNING_ANGLE_CHANGE_RATE_FEATS:
        angle = traj.turning_angles(accumulate=False)
        assert isinstance(angle, Vector)
        delta_t = traj.t.delta
        values = np.diff(angle) / delta_t[2:]

    assert values is not None, "Feature not found"

    if feat & MEAN_FEATS:
        return float(np.mean(values))
    if feat & MEDIAN_FEATS:
        med = values[len(values) // 2]
        assert isinstance(med, float)
        return float(med)
    if feat & MIN_FEATS:
        return float(np.min(values))
    if feat & MAX_FEATS:
        return float(np.max(values))
    if feat & STD_FEATS:
        return float(np.std(values))
    if feat & VAR_FEATS:
        return float(np.var(values))
    if feat & COEF_VAR_FEATS:
        mean = np.mean(values)
        return float(np.std(values) / mean if mean != 0 else 0)
    if feat & IQR_FEATS:
        val = np.percentile(values, 75) - np.percentile(values, 25)
        assert isinstance(val, float)
        return float(val)

    raise ValueError("Feature not found")


def get_feat_vector(traj: Trajectory, feats: int, **kwargs) -> list[float]:
    """Get feature vector from trajectory.

    Parameters
    ----------
    traj : Trajectory
        Trajectory.
    feats : int
        Features.

    Returns
    -------
    list[float]
        Feature vector.
    """
    return [get_feat(traj, feat, **kwargs) for feat in FEAT_VALUES if feat & feats]
