"""Proyecto 4Geeks: clasificador de vinos con K-Vecinos mas Cercanos.

El script reproduce el flujo principal del notebook:
1. Carga el dataset de vinos tintos.
2. Crea la variable `label` a partir de `quality`.
3. Entrena y optimiza un modelo KNN.
4. Evalua modelos de referencia.
5. Guarda los datos procesados, resultados y modelos entrenados.
"""

from pathlib import Path
import pickle
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_URL = (
    "https://raw.githubusercontent.com/4GeeksAcademy/"
    "k-nearest-neighbors-project-tutorial/refs/heads/main/winequality-red.csv"
)
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "winequality-red.csv"
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed" / "winequality_red_labeled.csv"
K_RESULTS_PATH = ROOT_DIR / "data" / "processed" / "knn_k_results.csv"
MODEL_RESULTS_PATH = ROOT_DIR / "data" / "processed" / "model_results.csv"
KNN_MODEL_PATH = ROOT_DIR / "models" / "wine_quality_knn.pkl"
BEST_MODEL_PATH = ROOT_DIR / "models" / "wine_quality_best_model.pkl"


FEATURE_COLUMNS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

LABEL_NAMES = {
    0: "baja",
    1: "media",
    2: "alta",
}


def load_data() -> pd.DataFrame:
    """Load the local CSV, falling back to the public URL."""
    if RAW_DATA_PATH.exists():
        return pd.read_csv(RAW_DATA_PATH, sep=";")

    return pd.read_csv(DATA_URL, sep=";")


def quality_to_label(quality: int) -> int:
    """Map original UCI quality scores into three classes."""
    if quality <= 4:
        return 0
    if quality <= 6:
        return 1
    return 2


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Create the target label and keep the useful columns."""
    clean_df = df.copy()
    clean_df = clean_df.drop_duplicates().reset_index(drop=True)
    clean_df["label"] = clean_df["quality"].apply(quality_to_label)
    return clean_df


def get_metrics(y_true, y_pred) -> dict[str, float]:
    """Return multiclass metrics for the positive and minority classes."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def evaluate_model(name: str, model, x_train, x_test, y_train, y_test) -> dict[str, Any]:
    """Fit a model and return its test metrics."""
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return {"model": name, **get_metrics(y_test, predictions)}


def predict_wine_quality(values: list[float], model_path: Path = KNN_MODEL_PATH) -> str:
    """Predict wine quality from the 11 chemical measurements."""
    with model_path.open("rb") as model_file:
        model = pickle.load(model_file)

    prediction = int(model.predict(pd.DataFrame([values], columns=FEATURE_COLUMNS))[0])
    return f"Este vino probablemente sea de calidad {LABEL_NAMES[prediction]}."


def main() -> None:
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    KNN_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = load_data()
    clean_df = prepare_data(df)
    clean_df.to_csv(PROCESSED_DATA_PATH, index=False)

    x = clean_df[FEATURE_COLUMNS]
    y = clean_df["label"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    k_results = []
    for k in range(1, 21):
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=k)),
            ]
        )
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        k_results.append(
            {
                "k": k,
                **get_metrics(y_test, predictions),
            }
        )

    k_results_df = pd.DataFrame(k_results)
    k_results_df.to_csv(K_RESULTS_PATH, index=False)
    best_k = int(k_results_df.sort_values(["accuracy", "f1_weighted"], ascending=False).iloc[0]["k"])

    knn_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=best_k)),
        ]
    )

    knn_distance_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=best_k, weights="distance")),
        ]
    )

    reference_models = {
        f"KNN k={best_k}": knn_model,
        f"KNN k={best_k} distance": knn_distance_model,
        "SVC RBF": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVC(class_weight="balanced", random_state=42)),
            ]
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
    }

    model_results = []
    fitted_models = {}
    for name, model in reference_models.items():
        model_results.append(evaluate_model(name, model, x_train, x_test, y_train, y_test))
        fitted_models[name] = model

    model_results_df = pd.DataFrame(model_results).sort_values(
        ["accuracy", "f1_weighted"],
        ascending=False,
    )
    model_results_df.to_csv(MODEL_RESULTS_PATH, index=False)

    with KNN_MODEL_PATH.open("wb") as model_file:
        pickle.dump(knn_distance_model, model_file)

    best_model_name = str(model_results_df.iloc[0]["model"])
    with BEST_MODEL_PATH.open("wb") as model_file:
        pickle.dump(fitted_models[best_model_name], model_file)

    example = [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]

    print("Dataset procesado guardado en:", PROCESSED_DATA_PATH)
    print("Resultados por k guardados en:", K_RESULTS_PATH)
    print("Comparativa de modelos guardada en:", MODEL_RESULTS_PATH)
    print("Mejor k uniforme:", best_k)
    print(model_results_df.round(4).to_string(index=False))
    print("Modelo KNN guardado en:", KNN_MODEL_PATH)
    print("Mejor modelo global guardado en:", BEST_MODEL_PATH)
    print("Prediccion de ejemplo:", predict_wine_quality(example))


if __name__ == "__main__":
    main()
