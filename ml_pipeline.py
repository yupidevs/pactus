import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import features as feats
from pipeline import Pipeline, PipelineStep

FeatClassTuple = tuple[list[float], str]
TrainTestTuples = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
FeatureImportances = list[tuple[str, float]]

RANDOM_FOREST = 0


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
    ) -> tuple[TrainTestTuples, RandomForestClassifier]:
        """
        Takes the splitted data and trains a RandomForestClassifier.

        Input: Splitted data.
        Output: tuple of two elements (splitted data and a trained
                instance of a Random Forest Classifier).
        """
        X_train, y_train = splitted_data[0:2]
        rfc_kwargs["criterion"] = rfc_kwargs.get("criterion", "entropy")
        rfc_kwargs["max_features"] = rfc_kwargs.get("max_features", "log2")
        rfc_kwargs["bootstrap"] = rfc_kwargs.get("bootstrap", False)
        rfc_kwargs["random_state"] = rfc_kwargs.get("random_state", None)
        rfc = RandomForestClassifier(**rfc_kwargs)
        rfc.fit(X_train, y_train)
        return splitted_data, rfc

    @PipelineStep.build("results visualizer")
    def results_visualizer(
        splitted_data: TrainTestTuples, rfc: RandomForestClassifier
    ) -> None:
        """
        Shows the random forest classifier results

        Input: tuple of two elements (splitted data and a trained
               instance of a Random Forest Classifier).
        Output: None
        """
        X_val, y_val = splitted_data[2:4]
        accuracy = float(rfc.score(X_val, y_val))
        importances = list(zip(feats.FEAT_NAMES, rfc.feature_importances_))
        importances = sorted(importances, key=lambda x: -x[1])[:10]

        names = [name for name, _ in importances]
        values = [imp for _, imp in importances]

        print("Random forest accuracy", accuracy)

        plt.title("10 most important features")
        plt.xlabel("Features")
        plt.ylabel("Importances")
        plt.xticks(rotation=-90)
        plt.bar(names, values)
        plt.grid()
        plt.show()

    return Pipeline(
        "Random forest classifier",
        splitter,
        random_forest_classifier,
        results_visualizer,
    )


def get_ml_classifier_pl(
    method: int = RANDOM_FOREST,
    test_size: float = 0.2,
    random_state: int | None = None,
    **kwargs
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

    raise ValueError("Unknown method")
