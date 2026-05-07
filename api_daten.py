"""API-Funktionen für Nominatim und opendata.swiss."""

# Wir nutzen requests für einfache HTTP-Anfragen.
import requests

# Die Datenbank speichert erfolgreiche Koordinaten dauerhaft.
import datenbank
from konfiguration import API_TIMEOUT_SECONDS, HTTP_USER_AGENT, NOMINATIM_URL, OPENDATA_URL, POSTCODE_LIST, POSTCODE_LOOKUP


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


def make_headers() -> dict[str, str]:
    """Zweck: Baut HTTP-Header. Parameter: Keine. Rückgabe: Header."""
    # Ein User-Agent erklärt öffentlichen APIs den Zweck der Anfrage.
    headers: dict[str, str] = {}
    headers["User-Agent"] = HTTP_USER_AGENT
    headers["Accept"] = "application/json"
    return headers


def fetch_coordinates(postcode: int, place: str) -> tuple[float, float]:
    """Zweck: Holt Koordinaten. Parameter: PLZ und Ort. Rückgabe: Lat und Lon."""
    # Nominatim liefert freie OpenStreetMap-Koordinaten für die Karte.
    params = {"q": str(postcode) + " " + place + ", Schweiz", "format": "jsonv2", "limit": 1}
    response = requests.get(NOMINATIM_URL, params=params, headers=make_headers(), timeout=API_TIMEOUT_SECONDS)
    response.raise_for_status()
    payload = response.json()
    if len(payload) == 0:
        raise RuntimeError("Nominatim hat keine Koordinate gefunden.")
    return float(payload[0]["lat"]), float(payload[0]["lon"])


def refresh_all_coordinates() -> list[str]:
    """Zweck: Aktualisiert alle Koordinaten. Parameter: Keine. Rückgabe: Warnungen."""
    # Jeder Fehler wird gesammelt, damit lokale Fallback-Werte bestehen bleiben.
    warnings = []
    for postcode in POSTCODE_LIST:
        warning = refresh_one_coordinate(postcode)
        if warning != "":
            warnings.append(warning)
    return warnings


def refresh_one_coordinate(postcode: int) -> str:
    """Zweck: Aktualisiert eine PLZ. Parameter: PLZ. Rückgabe: Warnung oder leerer Text."""
    # Fehler werden nicht geworfen, damit die App weiterläuft.
    place = POSTCODE_LOOKUP[postcode][0]
    try:
        lat, lon = fetch_coordinates(postcode, place)
        datenbank.update_coordinates(postcode, lat, lon)
        return ""
    except Exception as error:
        return "Koordinaten für PLZ " + str(postcode) + " konnten nicht geladen werden: " + str(error)


def search_housing_datasets() -> dict[str, object]:
    """Zweck: Sucht Wohn-Datensätze. Parameter: Keine. Rückgabe: Ergebnis."""
    # opendata.swiss katalogisiert öffentliche Schweizer Datensätze zu Wohnen und Immobilien.
    params = {"q": "Immobilien Wohnen Mietpreise", "rows": 0}
    try:
        response = requests.get(OPENDATA_URL, params=params, headers=make_headers(), timeout=API_TIMEOUT_SECONDS)
        response.raise_for_status()
        return parse_opendata(response.json())
    except Exception:
        return api_warning()


def parse_opendata(payload: dict[str, object]) -> dict[str, object]:
    """Zweck: Verarbeitet API-Antwort. Parameter: JSON. Rückgabe: Ergebnis."""
    # Der CKAN-Endpunkt gibt die Trefferzahl in result.count zurück.
    result = payload.get("result", {})
    count = 0
    if isinstance(result, dict):
        count = int(result.get("count", 0))
    return {"ok": True, "count": count, "message": "Datensätze wurden gefunden."}


def api_warning() -> dict[str, object]:
    """Zweck: Erstellt Ausfallmeldung. Parameter: Keine. Rückgabe: Ergebnis."""
    # Lokale Daten reichen für den Kern der App, deshalb ist der API-Ausfall nicht kritisch.
    message = "opendata.swiss ist momentan nicht erreichbar. Die App fährt mit lokalen Daten fort."
    return {"ok": False, "count": 0, "message": message}
