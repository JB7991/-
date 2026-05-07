"""SQLite-Funktionen für ImmoInsight."""

# Wir importieren die nötigen Bibliotheken für Datenbank, Zeit und Datenarbeit.
import sqlite3
from datetime import datetime
import numpy as np
import pandas as pd

# Wir importieren zentrale Konstanten, damit die Datenbank keine eigenen Listen pflegt.
from konfiguration import DATABASE_PATH, DATA_DIR, POSTCODE_LIST, POSTCODE_LOOKUP, RANDOM_STATE, SEED_ROWS

PROPERTY_SQL = "CREATE TABLE IF NOT EXISTS properties (id INTEGER PRIMARY KEY AUTOINCREMENT, postcode INTEGER, place TEXT, canton TEXT, area_m2 REAL, rooms REAL, floor INTEGER, has_parking INTEGER, has_garden INTEGER, year_built INTEGER, estimated_price REAL, source TEXT, created_at TEXT)"
STATS_SQL = "CREATE TABLE IF NOT EXISTS postcode_stats (postcode INTEGER PRIMARY KEY, place TEXT, canton TEXT, lat REAL, lon REAL, population_density REAL, average_income REAL, base_price_m2 REAL, updated_at TEXT)"
RUNS_SQL = "CREATE TABLE IF NOT EXISTS model_runs (id INTEGER PRIMARY KEY AUTOINCREMENT, model_name TEXT, mae REAL, rmse REAL, r2 REAL, trained_at TEXT)"


def now_text() -> str:
    """Zweck: Liefert Zeitstempel. Parameter: Keine. Rückgabe: Text."""
    # Einheitliche Zeitstempel erleichtern Sortierung und Anzeige.
    return datetime.now().isoformat(timespec="seconds")


def get_connection() -> sqlite3.Connection:
    """Zweck: Öffnet SQLite. Parameter: Keine. Rückgabe: Verbindung."""
    # Der Ordner wird automatisch erstellt, damit der erste Start funktioniert.
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database() -> None:
    """Zweck: Erstellt Tabellen und Seed-Daten. Parameter: Keine. Rückgabe: Keine."""
    # Initialisierung macht die App komplett ohne manuelle Vorbereitung lauffähig.
    with get_connection() as connection:
        connection.execute(PROPERTY_SQL)
        connection.execute(STATS_SQL)
        connection.execute(RUNS_SQL)
        seed_stats(connection)
        connection.execute("UPDATE properties SET source = 'Startdaten' WHERE source = 'Seed'")
        if count_rows(connection, "properties") == 0:
            seed_properties(connection)
        connection.commit()


def count_rows(connection: sqlite3.Connection, table_name: str) -> int:
    """Zweck: Zählt Tabellenzeilen. Parameter: Verbindung und Tabelle. Rückgabe: Anzahl."""
    # Die Zählung entscheidet, ob Seed-Daten fehlen.
    row = connection.execute("SELECT COUNT(*) AS amount FROM " + table_name).fetchone()
    return int(row["amount"])


def seed_stats(connection: sqlite3.Connection) -> None:
    """Zweck: Speichert PLZ-Statistiken. Parameter: Verbindung. Rückgabe: Keine."""
    # Lokale Statistiken sind der Offline-Fallback für API-Probleme.
    for postcode in POSTCODE_LIST:
        value = POSTCODE_LOOKUP[postcode]
        data = (postcode, value[0], value[1], value[2], value[3], value[4], value[5], value[6], now_text())
        connection.execute("INSERT OR IGNORE INTO postcode_stats VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", data)


def seed_properties(connection: sqlite3.Connection) -> None:
    """Zweck: Erstellt synthetische Immobilien. Parameter: Verbindung. Rückgabe: Keine."""
    # Synthetische Daten ermöglichen eine vollständige Demo ohne bezahlte Marktdaten.
    generator = np.random.default_rng(RANDOM_STATE)
    for number in range(SEED_ROWS):
        data = create_seed_property(generator, number)
        insert_property_row(connection, data)


def create_seed_property(generator: np.random.Generator, number: int) -> dict[str, float | int | str]:
    """Zweck: Erstellt eine Seed-Immobilie. Parameter: Generator und Nummer. Rückgabe: Dictionary."""
    # Die PLZ rotiert, damit alle Orte vertreten sind.
    postcode = POSTCODE_LIST[number % len(POSTCODE_LIST)]
    value = POSTCODE_LOOKUP[postcode]
    area = float(np.clip(generator.normal(92, 35), 28, 260))
    rooms = float(np.clip(round(area / 25) * 0.5 + generator.integers(1, 5) * 0.5, 1.0, 8.0))
    floor = int(np.clip(generator.poisson(3), 0, 18))
    year = int(generator.integers(1905, 2026))
    parking = int(generator.random() < 0.55)
    garden = int(generator.random() < 0.20)
    price = calculate_price(generator, value[6], area, rooms, floor, year, parking, garden)
    return make_property(postcode, value, area, rooms, floor, year, parking, garden, price, "Startdaten")


def calculate_price(generator: np.random.Generator, base: float, area: float, rooms: float, floor: int, year: int, parking: int, garden: int) -> float:
    """Zweck: Berechnet Seed-Preis. Parameter: Merkmale. Rückgabe: Preis."""
    # Der Preis kombiniert Lage, Fläche, Alter, Ausstattung und Zufallsrauschen.
    age_factor = 1.0 + (year - 1985) * 0.0015
    floor_factor = 1.0 + min(floor, 10) * 0.005
    room_factor = 1.0 + (rooms - 3.5) * 0.015
    feature_factor = 1.0 + parking * 0.035 + garden * 0.04
    noise = float(generator.lognormal(0.0, 0.14))
    price = area * base * age_factor * floor_factor * room_factor * feature_factor * noise
    return float(np.clip(price, 220000, 6000000))


def make_property(postcode: int, value: tuple, area: float, rooms: float, floor: int, year: int, parking: int, garden: int, price: float, source: str) -> dict[str, float | int | str]:
    """Zweck: Baut Objekt-Dictionary. Parameter: Werte. Rückgabe: Dictionary."""
    # Einheitliche Schlüssel verhindern Fehler zwischen Seed, Formular und SQL.
    data: dict[str, float | int | str] = {}
    data["postcode"] = postcode
    data["place"] = value[0]
    data["canton"] = value[1]
    data["area_m2"] = round(area, 1)
    data["rooms"] = rooms
    data["floor"] = floor
    data["year_built"] = year
    data["has_parking"] = parking
    data["has_garden"] = garden
    data["estimated_price"] = round(price, 0)
    data["source"] = source
    return data


def insert_property(inputs: dict[str, float | int | str], price: float, source: str) -> None:
    """Zweck: Speichert Formularobjekt. Parameter: Eingaben, Preis, Quelle. Rückgabe: Keine."""
    # Die Datenbankfunktion hält SQL aus der Oberfläche heraus.
    value = POSTCODE_LOOKUP[int(inputs["postcode"])]
    data = make_property(int(inputs["postcode"]), value, float(inputs["area_m2"]), float(inputs["rooms"]), int(inputs["floor"]), int(inputs["year_built"]), int(inputs["has_parking"]), int(inputs["has_garden"]), price, source)
    with get_connection() as connection:
        insert_property_row(connection, data)
        connection.commit()


def insert_property_row(connection: sqlite3.Connection, data: dict[str, float | int | str]) -> None:
    """Zweck: Führt Insert aus. Parameter: Verbindung und Daten. Rückgabe: Keine."""
    # Parameterisierte SQL-Werte sind verständlich und sicherer als Textverkettung.
    values = (data["postcode"], data["place"], data["canton"], data["area_m2"], data["rooms"], data["floor"], data["has_parking"], data["has_garden"], data["year_built"], data["estimated_price"], data["source"], now_text())
    connection.execute("INSERT INTO properties (postcode, place, canton, area_m2, rooms, floor, has_parking, has_garden, year_built, estimated_price, source, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", values)


def get_properties() -> pd.DataFrame:
    """Zweck: Lädt Immobilien. Parameter: Keine. Rückgabe: DataFrame."""
    # Neueste Einträge erscheinen zuerst.
    with get_connection() as connection:
        data = pd.read_sql_query("SELECT * FROM properties ORDER BY id DESC", connection)
    return data


def get_postcode_stats() -> pd.DataFrame:
    """Zweck: Lädt PLZ-Statistiken. Parameter: Keine. Rückgabe: DataFrame."""
    # Statistikdaten werden für Karte und Modell verwendet.
    with get_connection() as connection:
        data = pd.read_sql_query("SELECT * FROM postcode_stats ORDER BY postcode", connection)
    return data


def get_training_data() -> pd.DataFrame:
    """Zweck: Lädt Trainingsdaten. Parameter: Keine. Rückgabe: DataFrame."""
    # Der Join ergänzt jede Immobilie mit Standortmerkmalen.
    query = "SELECT p.*, s.population_density, s.average_income FROM properties p JOIN postcode_stats s ON p.postcode = s.postcode"
    with get_connection() as connection:
        data = pd.read_sql_query(query, connection)
    return data


def filter_properties(postcodes: list[int], price_range: tuple[int, int], area_range: tuple[float, float]) -> pd.DataFrame:
    """Zweck: Filtert Explorer-Daten. Parameter: Filterwerte. Rückgabe: DataFrame."""
    # Pandas-Filter sind für Anfänger leichter erklärbar als dynamisches SQL.
    data = get_properties()
    data = data[data["postcode"].isin(postcodes)]
    data = data[data["estimated_price"] >= price_range[0]]
    data = data[data["estimated_price"] <= price_range[1]]
    data = data[data["area_m2"] >= area_range[0]]
    data = data[data["area_m2"] <= area_range[1]]
    return data


def get_average_price_for_postcode(postcode: int) -> float:
    """Zweck: Berechnet PLZ-Durchschnitt. Parameter: PLZ. Rückgabe: Preis."""
    # Der Durchschnitt ist die Referenz im Tachometer.
    data = get_properties()
    part = data[data["postcode"] == postcode]
    return float(part["estimated_price"].mean())


def save_model_run(model_name: str, mae: float, rmse: float, r2: float, trained_at: str) -> None:
    """Zweck: Speichert Modelllauf. Parameter: Kennzahlen. Rückgabe: Keine."""
    # Die Historie zeigt, wie sich Modellqualität verändert.
    with get_connection() as connection:
        connection.execute("INSERT INTO model_runs (model_name, mae, rmse, r2, trained_at) VALUES (?, ?, ?, ?, ?)", (model_name, mae, rmse, r2, trained_at))
        connection.commit()


def get_model_runs() -> pd.DataFrame:
    """Zweck: Lädt Modellläufe. Parameter: Keine. Rückgabe: DataFrame."""
    # Die Trainingsseite zeigt diese Daten als Verlauf.
    with get_connection() as connection:
        data = pd.read_sql_query("SELECT * FROM model_runs ORDER BY trained_at", connection)
    return data


def update_coordinates(postcode: int, lat: float, lon: float) -> None:
    """Zweck: Speichert Koordinaten. Parameter: PLZ, Lat, Lon. Rückgabe: Keine."""
    # API-Koordinaten verbessern die Karte ohne neue Immobilienwerte.
    with get_connection() as connection:
        connection.execute("UPDATE postcode_stats SET lat = ?, lon = ?, updated_at = ? WHERE postcode = ?", (lat, lon, now_text(), postcode))
        connection.commit()


def get_signature() -> str:
    """Zweck: Erstellt Cache-Signatur. Parameter: Keine. Rückgabe: Text."""
    # Die Signatur ändert sich bei neuen Immobilien oder Modellläufen.
    with get_connection() as connection:
        properties = count_rows(connection, "properties")
        runs = count_rows(connection, "model_runs")
    return str(properties) + "-" + str(runs)
