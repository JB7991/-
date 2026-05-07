"""Konstanten für die deutsche ImmoInsight-App."""

# Pfade werden zentral definiert, damit alle Module dieselben Dateien nutzen.
from pathlib import Path

APP_TITLE = "ImmoInsight - Schweizer Immobilienpreisschätzer"
APP_PASSWORT = "immo2026"
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
DATABASE_PATH = DATA_DIR / "immoinsight_deutsch.db"
MODEL_PATH = MODEL_DIR / "immoinsight_deutsche_modelle.joblib"

# Reproduzierbarkeit ist wichtig, damit Präsentation und Tests gleiche Ergebnisse liefern.
RANDOM_STATE = 42
SEED_ROWS = 520
TEST_SIZE = 0.2

# Das Farbsystem wird zentral gehalten, damit CSS und Plotly konsistent bleiben.
FARBEN = {
    "sidebar_hintergrund": "#1C2B1E",
    "sidebar_nav_aktiv": "#2D5A35",
    "sidebar_text": "#FFFFFF",
    "sidebar_text_gedimmt": "rgba(255,255,255,0.50)",
    "seiten_hintergrund": "#F7F8F6",
    "karte_hintergrund": "#FFFFFF",
    "karte_rahmen": "#E2E8E3",
    "text_haupt": "#1A3B1E",
    "text_neben": "#7A8C7C",
    "feld_hintergrund": "#F7F8F6",
    "feld_rahmen": "#D1D9D2",
    "feld_rahmen_fokus": "#2D7A40",
    "akzent": "#2D7A40",
    "akzent_hell": "#E8F5EA",
    "akzent_text": "#1A6B3C",
    "ergebnis_hintergrund": "#1C2B1E",
    "ergebnis_text": "#FFFFFF",
    "ergebnis_akzent": "#7FCF8A",
}

# Die Seiten sind ohne Emojis benannt und erscheinen in der Sidebar.
PAGE_NAMES = ["Preisschätzung", "Marktübersicht", "Daten-Explorer", "ML-Modell", "Über die App"]
MODEL_NAMES = ["Lineare Regression", "Random Forest", "Gradient Boosting"]

# Alle Features sind numerisch, damit das Modell im ersten Semester erklärbar bleibt.
FEATURE_COLUMNS = ["postcode", "area_m2", "rooms", "floor", "has_parking", "has_garden", "year_built", "population_density", "average_income"]
TARGET_COLUMN = "estimated_price"

# API-Konstanten stehen zentral, damit Endpunkte nicht im App-Code verteilt sind.
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OPENDATA_URL = "https://opendata.swiss/api/3/action/package_search"
API_TIMEOUT_SECONDS = 8
HTTP_USER_AGENT = "ImmoInsight Studienprojekt 2026"

# Die Lookup-Tabelle enthält PLZ, Ort, Kanton, Koordinaten, Dichte, Einkommen und Basispreis.
POSTCODE_LOOKUP = {
    8001: ("Zürich", "ZH", 47.3769, 8.5417, 4770.0, 89200.0, 14800.0),
    3001: ("Bern", "BE", 46.9480, 7.4474, 2600.0, 76800.0, 9300.0),
    4001: ("Basel", "BS", 47.5596, 7.5886, 7200.0, 82500.0, 10600.0),
    1201: ("Genf", "GE", 46.2044, 6.1432, 12700.0, 86600.0, 13700.0),
    1003: ("Lausanne", "VD", 46.5197, 6.6323, 3400.0, 79600.0, 11600.0),
    9000: ("St. Gallen", "SG", 47.4245, 9.3767, 1900.0, 72400.0, 7200.0),
    6900: ("Lugano", "TI", 46.0037, 8.9511, 850.0, 69500.0, 8500.0),
    8400: ("Winterthur", "ZH", 47.4988, 8.7237, 1700.0, 81200.0, 9000.0),
    5000: ("Aarau", "AG", 47.3917, 8.0446, 3300.0, 76500.0, 8300.0),
    2501: ("Biel", "BE", 47.1368, 7.2468, 2600.0, 70500.0, 7100.0),
    6003: ("Luzern", "LU", 47.0502, 8.3093, 2900.0, 77800.0, 10800.0),
    7000: ("Chur", "GR", 46.8508, 9.5320, 1100.0, 71400.0, 7800.0),
    3900: ("Brig", "VS", 46.3160, 7.9870, 720.0, 68200.0, 6500.0),
    1700: ("Freiburg", "FR", 46.8065, 7.1619, 4100.0, 73500.0, 8200.0),
    4500: ("Solothurn", "SO", 47.2088, 7.5323, 2700.0, 74200.0, 7600.0),
    2000: ("Neuenburg", "NE", 46.9918, 6.9310, 2100.0, 71800.0, 7900.0),
    6500: ("Bellinzona", "TI", 46.1946, 9.0244, 980.0, 68400.0, 6900.0),
    8200: ("Schaffhausen", "SH", 47.6973, 8.6349, 1200.0, 74500.0, 7400.0),
    1800: ("Vevey", "VD", 46.4628, 6.8419, 7800.0, 77000.0, 11200.0),
    3600: ("Thun", "BE", 46.7512, 7.6217, 2100.0, 72800.0, 7600.0),
}

# Die sortierte Liste wird für Selectboxen und Schleifen genutzt.
POSTCODE_LIST = sorted(POSTCODE_LOOKUP.keys())

# Deutsche Feature-Namen erscheinen in der Modell-Erklärung.
FEATURE_LABELS = {
    "postcode": "PLZ",
    "area_m2": "Wohnfläche",
    "rooms": "Zimmer",
    "floor": "Etage",
    "has_parking": "Parkplatz",
    "has_garden": "Garten",
    "year_built": "Baujahr",
    "population_density": "Bevölkerungsdichte",
    "average_income": "Durchschnittseinkommen",
}
