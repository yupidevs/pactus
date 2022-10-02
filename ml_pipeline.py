import matplotlib.pyplot as plt
import numpy as np
import sklearn.pipeline as skp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import features as feats
from pipeline import Pipeline, PipelineStep

FeatClassTuple = tuple[list[float], str]
TrainTestTuples = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
FeatureImportances = list[tuple[str, float]]

RANDOM_FOREST = 0
SUPPORT_VECTOR_MACHINE = 1


@PipelineStep.build("Confusion matrix builder")
def confusion_matrix_builder(classes: list, y_val: np.ndarray, pred_val: np.ndarray):
    """
    Shows the confusion matrix

    Input: tuple of three elements (classes, y_val, pred_val)
    Output: None
    """
    print("Confusion matrix")
    cm = confusion_matrix(y_val, pred_val, labels=classes)

    # estimate the precision
    precision = [np.max(row) / np.sum(row) for row in cm]

    # Normalize the confusion matrix by columns
    cm = cm.astype("float")
    for i in range(cm.shape[0]):
        cm[:, i] /= cm[:, i].sum()

    # Round the values
    # cm = np.round(cm, decimals=2)
    classes = list(classes) + ["precision"]
    print(*[f"{c:>10}" for c in classes], sep=" ")
    print("-" * 10 * (len(classes) + 1))
    for i, row in enumerate(cm):
        # get the precision
        row = np.append(row, precision[i])
        print(*[f"{round(c * 100, 2):>10}" for c in row], sep=" ")


def get_train_test_splitter(test_size: float = 0.2, random_state: int | None = None):
    """
    Creates a pipeline step that splits the data into two
    sets: train and test.

    Parameters
    ----------
    test_size : float
        Size of the test size from 0 to 1.
    random_state : int
        Seed
    """

    @PipelineStep.build("train test splitter")
    def train_test_splitter(
        feat_vecs: list[list[float]], clss: list[str]
    ) -> TrainTestTuples:
        """
        Splits the data into two sets: train and test.

        Input: list of feature vectors, classes
        Output: tuple of np.ndarray (X_train, y_train, X_test, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            feat_vecs,
            clss,
            stratify=clss,
            random_state=random_state,
            test_size=test_size,
        )

        return (X_train, y_train, X_test, y_test)

    return train_test_splitter


def __get_random_forest_pl(splitter: PipelineStep, **rfc_kwargs):
    @PipelineStep.build("random forest trainer")
    def random_forest_classifier(
        splitted_data: TrainTestTuples,
    ) -> tuple[TrainTestTuples, GridSearchCV]:
        """
        Takes the splitted data and trains a RandomForestClassifier.

        Input: Splitted data.
        Output: tuple of two elements (splitted data and a trained
                instance of a Random Forest Classifier).
        """
        X, y = splitted_data[0:2]
        X = np.array(X)
        y = np.array(y)
        rfc_kwargs["max_features"] = rfc_kwargs.get("max_features", 16)
        rfc_kwargs["n_estimators"] = rfc_kwargs.get("n_estimators", 200)
        rfc_kwargs["bootstrap"] = rfc_kwargs.get("bootstrap", False)
        rfc_kwargs["random_state"] = rfc_kwargs.get("random_state", None)

        rfc_kwargs["warm_start"] = True
        rfc_kwargs["n_jobs"] = 6

        rfc = RandomForestClassifier(**rfc_kwargs)
        rfc = GridSearchCV(rfc, {}, verbose=3)
        rfc.fit(X, y)
        return splitted_data, rfc

    @PipelineStep.build("score")
    def score(
        splitted_data: TrainTestTuples, rfc: GridSearchCV
    ) -> tuple[list, np.ndarray, np.ndarray]:
        """
        Shows the random forest classifier score

        Input: tuple of two elements (splitted data and a trained
               instance of a Random Forest Classifier).
        Output: tuple of three elements (classes, y_val, pred_val)
        """
        X_val, y_val = splitted_data[2:4]
        accuracy = float(rfc.score(X_val, y_val))
        # importances = list(zip(feats.FEAT_NAMES, rfc.feature_importances_))
        # importances = sorted(importances, key=lambda x: -x[1])[:10]
        # print("Feature importances")
        # print(*[f"{name}: {importance}" for name, importance in importances], sep="\n")

        # plt.title("10 most important features")
        # plt.xlabel("Features")
        # plt.ylabel("Importances")
        # plt.xticks(rotation=-90)
        # plt.bar(names, values)
        # plt.grid()
        # plt.show()

        print("Random forest accuracy", accuracy)
        pred_val = rfc.predict(X_val)
        return rfc.classes_, y_val, pred_val

    return Pipeline(
        "Random forest classifier",
        splitter,
        random_forest_classifier,
        score,
        confusion_matrix_builder,
    )


def __get_svm_pl(splitter: PipelineStep, **svc_kwargs):
    @PipelineStep.build("SVM trainer")
    def random_forest_classifier(
        splitted_data: TrainTestTuples,
    ) -> tuple[TrainTestTuples, skp.Pipeline]:
        """
        Takes the splitted data and trains a RandomForestClassifier.

        Input: Splitted data.
        Output: tuple of two elements (splitted data and a trained
                instance of a Random Forest Classifier).
        """
        X_train, y_train = splitted_data[0:2]
        svc_kwargs["kernel"] = svc_kwargs.get("kernel", "linear")
        svc_kwargs["gamma"] = svc_kwargs.get("gamma", 7)
        svc_kwargs["C"] = svc_kwargs.get("C", 8)
        svc_kwargs["random_state"] = svc_kwargs.get("random_state", None)

        svc = skp.make_pipeline(StandardScaler(), SVC(**svc_kwargs))
        svc.fit(X_train, y_train)
        return splitted_data, svc

    @PipelineStep.build("score")
    def score(
        splitted_data: TrainTestTuples, svc: skp.Pipeline
    ) -> tuple[list, np.ndarray, np.ndarray]:
        """
        Shows the SVM classifier score

        Input: tuple of two elements (splitted data and a trained
               instance of a SVM classifier).
        Output: tuple of three elements (classes, y_val, pred_val)
        """
        X_val, y_val = splitted_data[2:4]
        accuracy = float(svc.score(X_val, y_val))
        print("SVM accuracy", accuracy)
        pred_val = svc.predict(X_val)
        return svc.classes_, y_val, pred_val

    return Pipeline(
        "Random forest classifier",
        splitter,
        random_forest_classifier,
        score,
        confusion_matrix_builder,
    )


def get_ml_classifier_pl(
    method: int = RANDOM_FOREST,
    test_size: float = 0.2,
    random_state: int | None = None,
    **kwargs,
) -> Pipeline:
    """
    Creates a classification pipeline

    Parameters
    ----------
    method : int
        Classification method, by default 0 (Random forest)
    """

    splitter = get_train_test_splitter(test_size=test_size, random_state=random_state)

    kwargs["random_state"] = random_state

    if method == RANDOM_FOREST:
        return __get_random_forest_pl(splitter, **kwargs)
    if method == SUPPORT_VECTOR_MACHINE:
        return __get_svm_pl(splitter, **kwargs)

    raise ValueError("Unknown method")
