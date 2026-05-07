"""Einfaches Machine Learning für ImmoInsight."""

# Wir importieren Bibliotheken für Modell-Cache, Zahlen, Tabellen und scikit-learn.
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Wir importieren Projektmodule und Konstanten für gleiche Spaltennamen.
import datenbank
from konfiguration import FEATURE_COLUMNS, FEATURE_LABELS, MODEL_DIR, MODEL_NAMES, MODEL_PATH, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE


def plotly_design_anwenden(fig):
    """
    Zweck: Wendet das App-Design auf jeden Plotly-Chart an.
    Parameter: fig – das Plotly-Figure-Objekt
    Rückgabe: fig – das angepasste Figure-Objekt
    """
    # Hintergrund des Charts transparent (passt sich der Karte an)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#F7F8F6",
        font=dict(color="#1A3B1E", family="sans-serif", size=12),
        title_font=dict(color="#1A3B1E", size=14, weight=500),
        # Rahmen und Gitter dezent halten
        xaxis=dict(showgrid=True, gridcolor="#E2E8E3", gridwidth=0.5, linecolor="#D1D9D2", tickfont=dict(color="#7A8C7C", size=11)),
        yaxis=dict(showgrid=True, gridcolor="#E2E8E3", gridwidth=0.5, linecolor="#D1D9D2", tickfont=dict(color="#7A8C7C", size=11)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#1A3B1E", size=11)),
        # Etwas Innenabstand
        margin=dict(l=16, r=16, t=36, b=16)
    )
    # Balken und Linien in der App-Akzentfarbe
    fig.update_traces(marker_color="#2D7A40", selector=dict(type="bar"))
    return fig


def load_or_train(properties: pd.DataFrame) -> dict[str, object]:
    """Zweck: Lädt oder trainiert Modelle. Parameter: Immobilien. Rückgabe: Bundle."""
    # Der Cache verhindert Training bei jedem Seitenwechsel.
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return train_and_save(properties, True)


def train_and_save(properties: pd.DataFrame, save_runs: bool) -> dict[str, object]:
    """Zweck: Trainiert alle Modelle. Parameter: Daten und Speicherflag. Rückgabe: Bundle."""
    # Wir nutzen Daten aus SQLite, damit auch neue Nutzereingaben einfliessen.
    data = datenbank.get_training_data()
    x_data = data[FEATURE_COLUMNS].copy()
    y_data = data[TARGET_COLUMN].copy()
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    scaler = StandardScaler()
    scaler.fit(x_train)
    models = fit_models(scaler.transform(x_train), y_train)
    metrics = evaluate_models(models, scaler.transform(x_test), y_test, save_runs)
    bundle = make_bundle(models, scaler, metrics, len(x_train), len(x_test))
    save_bundle(bundle)
    return bundle


def fit_models(x_train: np.ndarray, y_train: pd.Series) -> dict[str, object]:
    """Zweck: Passt drei Modelle an. Parameter: Trainingsdaten. Rückgabe: Modelle."""
    # Alle Modelle erhalten dieselben skalierten Daten für einen fairen Vergleich.
    models: dict[str, object] = {}
    models["Lineare Regression"] = LinearRegression().fit(x_train, y_train)
    models["Random Forest"] = RandomForestRegressor(n_estimators=180, random_state=RANDOM_STATE).fit(x_train, y_train)
    models["Gradient Boosting"] = GradientBoostingRegressor(random_state=RANDOM_STATE).fit(x_train, y_train)
    return models


def evaluate_models(models: dict[str, object], x_test: np.ndarray, y_test: pd.Series, save_runs: bool) -> dict[str, dict[str, float | str]]:
    """Zweck: Bewertet Modelle. Parameter: Modelle und Testdaten. Rückgabe: Metriken."""
    # Testdaten simulieren unbekannte Immobilien.
    metrics: dict[str, dict[str, float | str]] = {}
    for model_name in MODEL_NAMES:
        row = evaluate_one(models[model_name], x_test, y_test)
        metrics[model_name] = row
        if save_runs is True:
            datenbank.save_model_run(model_name, float(row["mae"]), float(row["rmse"]), float(row["r2"]), str(row["trained_at"]))
    return metrics


def evaluate_one(model: object, x_test: np.ndarray, y_test: pd.Series) -> dict[str, float | str]:
    """Zweck: Bewertet ein Modell. Parameter: Modell und Testdaten. Rückgabe: Zeile."""
    # MAE und RMSE messen Fehler in CHF, R-Quadrat misst Erklärungskraft.
    prediction = model.predict(x_test)
    mae = float(mean_absolute_error(y_test, prediction))
    rmse = float(np.sqrt(mean_squared_error(y_test, prediction)))
    r2 = float(r2_score(y_test, prediction))
    residual = float(np.std(y_test - prediction))
    return {"mae": mae, "rmse": rmse, "r2": r2, "residual": residual, "trained_at": datenbank.now_text()}


def make_bundle(models: dict[str, object], scaler: StandardScaler, metrics: dict[str, dict[str, float | str]], train_size: int, test_size: int) -> dict[str, object]:
    """Zweck: Bündelt ML-Objekte. Parameter: Modelle und Daten. Rückgabe: Bundle."""
    # Ein Dictionary ist einfacher erklärbar als eine Klasse.
    bundle: dict[str, object] = {}
    bundle["models"] = models
    bundle["scaler"] = scaler
    bundle["metrics"] = metrics
    bundle["train_size"] = train_size
    bundle["test_size"] = test_size
    return bundle


def save_bundle(bundle: dict[str, object]) -> None:
    """Zweck: Speichert Modellcache. Parameter: Bundle. Rückgabe: Keine."""
    # Der Ordner entsteht automatisch, damit der erste Start funktioniert.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)


def make_feature_dictionary(inputs: dict[str, float | int | str], stat_row: pd.Series) -> dict[str, float]:
    """Zweck: Baut Modellfeatures. Parameter: Eingaben und Statistik. Rückgabe: Features."""
    # Die PLZ ist ein numerisches Feature und ersetzt Stadt-One-Hot-Encoding.
    features: dict[str, float] = {}
    for column in FEATURE_COLUMNS:
        features[column] = get_feature_value(column, inputs, stat_row)
    return features


def get_feature_value(column: str, inputs: dict[str, float | int | str], stat_row: pd.Series) -> float:
    """Zweck: Holt einen Feature-Wert. Parameter: Spalte, Eingaben, Statistik. Rückgabe: Zahl."""
    # Standortwerte kommen aus der Statistik, Objektwerte aus dem Formular.
    if column == "population_density":
        return float(stat_row["population_density"])
    if column == "average_income":
        return float(stat_row["average_income"])
    return float(inputs[column])


def predict_price(bundle: dict[str, object], model_name: str, feature_row: dict[str, float]) -> dict[str, float]:
    """Zweck: Sagt Preis voraus. Parameter: Bundle, Modell, Features. Rückgabe: Preisbereich."""
    # Vorhersagedaten müssen dieselbe Spaltenreihenfolge wie Trainingsdaten haben.
    frame = pd.DataFrame([feature_row])
    frame = frame[FEATURE_COLUMNS]
    scaled = bundle["scaler"].transform(frame)
    price = float(bundle["models"][model_name].predict(scaled)[0])
    width = interval_width(bundle, model_name, price)
    return {"price": price, "lower": max(0.0, price - width), "upper": price + width}


def interval_width(bundle: dict[str, object], model_name: str, price: float) -> float:
    """Zweck: Schätzt Unsicherheit. Parameter: Bundle, Modell, Preis. Rückgabe: Breite."""
    # Der Bereich ist eine einfache Projektnäherung, keine Garantie.
    residual = float(bundle["metrics"][model_name]["residual"])
    minimum = price * 0.08
    return max(minimum, residual * 1.1)


def metrics_to_dataframe(bundle: dict[str, object]) -> pd.DataFrame:
    """Zweck: Baut Metriktabelle. Parameter: Bundle. Rückgabe: DataFrame."""
    # Deutsche Spalten machen den Modellvergleich präsentationsfähig.
    rows = []
    for model_name in MODEL_NAMES:
        metric = bundle["metrics"][model_name]
        rows.append(metric_row(model_name, metric))
    return pd.DataFrame(rows)


def metric_row(model_name: str, metric: dict[str, float | str]) -> dict[str, float | str]:
    """Zweck: Baut eine Metrikzeile. Parameter: Name und Metrik. Rückgabe: Zeile."""
    # Runde Zahlen sind im Unterricht leichter lesbar.
    row: dict[str, float | str] = {}
    row["Modell"] = model_name
    row["MAE"] = round(float(metric["mae"]), 0)
    row["RMSE"] = round(float(metric["rmse"]), 0)
    row["R-Quadrat"] = round(float(metric["r2"]), 3)
    row["Trainingszeit"] = str(metric["trained_at"])
    return row


def get_feature_importance(bundle: dict[str, object], model_name: str) -> pd.DataFrame:
    """Zweck: Berechnet Wichtigkeit. Parameter: Bundle und Modell. Rückgabe: DataFrame."""
    # Bäume nutzen feature_importances, lineare Regression nutzt Koeffizienten.
    model = bundle["models"][model_name]
    values = raw_importance(model)
    rows = []
    total = float(values.sum())
    for index in range(len(FEATURE_COLUMNS)):
        rows.append(importance_row(index, values, total))
    return pd.DataFrame(rows)


def raw_importance(model: object) -> np.ndarray:
    """Zweck: Holt Rohwerte. Parameter: Modell. Rückgabe: Array."""
    # Absolute Werte vermeiden negative Balken bei linearer Regression.
    if hasattr(model, "feature_importances_"):
        return np.array(model.feature_importances_)
    if hasattr(model, "coef_"):
        return np.abs(np.array(model.coef_))
    return np.zeros(len(FEATURE_COLUMNS))


def importance_row(index: int, values: np.ndarray, total: float) -> dict[str, float | str]:
    """Zweck: Baut Wichtigkeitszeile. Parameter: Index, Werte, Summe. Rückgabe: Zeile."""
    # Normierung macht Werte zwischen Modellen vergleichbar.
    column = FEATURE_COLUMNS[index]
    value = float(values[index] / total) if total > 0 else 0.0
    return {"Merkmal": FEATURE_LABELS[column], "Wichtigkeit": value}
