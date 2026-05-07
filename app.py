"""Deutsche Streamlit-App für ImmoInsight."""
# Wir importieren Standard- und Datenbibliotheken für Oberfläche, Tabellen und Diagramme.
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
# Wir trennen API, Datenbank, Konfiguration und ML, damit die App erklärbar bleibt.
import api_daten
import datenbank
import ml_modell
from konfiguration import APP_PASSWORT, APP_TITLE, PAGE_NAMES, POSTCODE_LIST, POSTCODE_LOOKUP
st.set_page_config(
    page_title="ImmoInsight",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>

/* =============================================
   GLOBALE TEXT-FARBEN – verhindert weiss auf weiss
   Streamlit setzt intern var(--text-color) auf weiss im Dark-Mode.
   Wir überschreiben alle bekannten Text-Selektoren.
   ============================================= */

/* Alle normalen Textelemente im Hauptbereich */
.stApp, .stApp * {
    color: #1A3B1E;
}

/* Paragraphen und Markdown-Text */
.stApp p,
.stApp span,
.stApp div,
.stMarkdown p,
.stMarkdown span,
.stMarkdown li,
.stMarkdown h1,
.stMarkdown h2,
.stMarkdown h3 {
    color: #1A3B1E !important;
}

/* =============================================
   SIDEBAR-TEXTE – hell auf dunkel
   Die Sidebar ist dunkel (#1C2B1E), also muss
   der Text dort weiss sein – explizit setzen.
   ============================================= */

[data-testid="stSidebar"],
[data-testid="stSidebar"] * {
    color: rgba(255, 255, 255, 0.90) !important;
}

/* Sidebar-Titel und Untertitel bleiben explizit weiss auf dunklem Grün */
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: rgba(255, 255, 255, 0.90) !important;
}

/* Navigationstexte werden vollständig weiss gesetzt */
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stRadio label *,
[data-testid="stSidebar"] [data-baseweb="radio"] label,
[data-testid="stSidebar"] [data-baseweb="radio"] label * {
    color: #FFFFFF !important;
}

/* Der Auswahlpunkt der Radio-Navigation wird weiss statt grün dargestellt */
[data-testid="stSidebar"] [data-baseweb="radio"] div,
[data-testid="stSidebar"] [data-baseweb="radio"] span,
[data-testid="stSidebar"] input[type="radio"] + div,
[data-testid="stSidebar"] input[type="radio"] + div * {
    color: #FFFFFF !important;
    border-color: #FFFFFF !important;
}

/* Sidebar-Hintergrund */
[data-testid="stSidebar"] {
    background-color: #1C2B1E !important;
}

/* =============================================
   EINGABEFELD-LABELS – sind oft weiss auf weiss
   Streamlit rendert Labels als eigene spans,
   die separat gefärbt werden müssen.
   ============================================= */

/* Labels über Textfeldern, Zahlenfeldern, Selectboxen */
.stTextInput label,
.stNumberInput label,
.stSelectbox label,
.stMultiSelect label,
.stSlider label,
.stRadio label,
.stCheckbox label,
.stDateInput label,
.stTextArea label,
.stFileUploader label {
    color: #7A8C7C !important;
    font-size: 0.8rem !important;
}

/* =============================================
   EINGABEFELD-INHALTE – Text der eingetippt wird
   ============================================= */

.stTextInput input,
.stNumberInput input,
.stTextArea textarea {
    color: #1A3B1E !important;
    background-color: #F7F8F6 !important;
    border: 0.5px solid #D1D9D2 !important;
    border-radius: 7px !important;
}

/* Placeholder-Text in Eingabefeldern */
.stTextInput input::placeholder,
.stNumberInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #7A8C7C !important;
}

/* Fokus-Zustand des Eingabefeldes */
.stTextInput input:focus,
.stNumberInput input:focus {
    border-color: #2D7A40 !important;
    box-shadow: 0 0 0 2px rgba(45, 122, 64, 0.15) !important;
}

/* =============================================
   SELECTBOX – Dropdown-Text ist oft weiss
   ============================================= */

/* Der angezeigte Wert in der Selectbox */
.stSelectbox [data-baseweb="select"] span,
.stSelectbox [data-baseweb="select"] div {
    color: #1A3B1E !important;
    background-color: #F7F8F6 !important;
}

/* Dropdown-Optionen in der ausgeklappten Liste */
[data-baseweb="popover"] li,
[data-baseweb="menu"] li,
[data-baseweb="option"] {
    color: #1A3B1E !important;
    background-color: #FFFFFF !important;
}
[data-baseweb="option"]:hover {
    background-color: #E8F5EA !important;
}

/* MultiSelect Tags */
.stMultiSelect [data-baseweb="tag"] {
    background-color: #E8F5EA !important;
    color: #1A6B3C !important;
}

/* =============================================
   SLIDER – Werte und Labels
   ============================================= */

.stSlider [data-testid="stTickBar"] span,
.stSlider [data-baseweb="slider"] [role="slider"] {
    color: #1A3B1E !important;
}
/* Akzentfarbe für den Slider-Balken */
.stSlider [data-baseweb="slider"] div[data-testid="stSliderTrackFill"] {
    background-color: #2D7A40 !important;
}

/* =============================================
   RADIO BUTTONS – Optionstext
   ============================================= */

.stRadio [data-baseweb="radio"] label {
    color: #1A3B1E !important;
}

/* Radio in der Sidebar bleibt weiss */
[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] label {
    color: rgba(255, 255, 255, 0.75) !important;
}

/* =============================================
   CHECKBOXEN – Beschriftung
   ============================================= */

.stCheckbox [data-baseweb="checkbox"] label {
    color: #1A3B1E !important;
}
/* Angehakt: Akzentgrün */
.stCheckbox [data-baseweb="checkbox"] [data-checked="true"] {
    background-color: #2D7A40 !important;
    border-color: #2D7A40 !important;
}

/* =============================================
   METRIC CARDS (st.metric)
   ============================================= */

[data-testid="stMetric"] {
    background-color: #FFFFFF !important;
    border: 0.5px solid #E2E8E3 !important;
    border-radius: 10px !important;
    padding: 16px 18px !important;
}
[data-testid="stMetricLabel"] p {
    color: #7A8C7C !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.4px;
}
[data-testid="stMetricValue"] {
    color: #1A3B1E !important;
}
[data-testid="stMetricDelta"] {
    color: #2D7A40 !important;
}

/* =============================================
   DATAFRAME / TABELLEN
   ============================================= */

[data-testid="stDataFrame"] th {
    background-color: #F7F8F6 !important;
    color: #7A8C7C !important;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.4px;
}
[data-testid="stDataFrame"] td {
    color: #1A3B1E !important;
}

/* =============================================
   ALERT-BOXEN – success / warning / info / error
   Diese haben oft weissen Text intern
   ============================================= */

[data-testid="stAlert"] p,
[data-testid="stAlert"] span {
    color: inherit !important;
}
[data-testid="stAlert"][data-baseweb="notification"][kind="positive"],
div[data-testid="stAlert"] > div[class*="success"] {
    background-color: #E8F5EA !important;
    border-left: 3px solid #2D7A40 !important;
    color: #1A6B3C !important;
}
[data-testid="stAlert"][kind="warning"] {
    background-color: #FFF8E8 !important;
    border-left: 3px solid #B8860B !important;
    color: #7A5800 !important;
}
[data-testid="stAlert"][kind="info"] {
    background-color: #EAF3FB !important;
    border-left: 3px solid #1A6BAA !important;
    color: #0D3F6B !important;
}
[data-testid="stAlert"][kind="error"] {
    background-color: #FDECEA !important;
    border-left: 3px solid #C0392B !important;
    color: #7B241C !important;
}

/* =============================================
   EXPANDER
   ============================================= */

[data-testid="stExpander"] {
    background-color: #FFFFFF !important;
    border: 0.5px solid #E2E8E3 !important;
    border-radius: 8px !important;
}
[data-testid="stExpander"] summary p {
    color: #1A3B1E !important;
    font-weight: 500;
}
[data-testid="stExpander"] summary svg {
    fill: #1A3B1E !important;
}

/* =============================================
   TABS
   ============================================= */

.stTabs [data-baseweb="tab"] p {
    color: #7A8C7C !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] p {
    color: #2D7A40 !important;
}
.stTabs [data-baseweb="tab-border"] {
    background-color: #2D7A40 !important;
}

/* =============================================
   BUTTONS
   ============================================= */

.stButton > button {
    border-radius: 7px !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    transition: transform 0.1s !important;
}
.stButton > button[kind="primary"] {
    background-color: #2D7A40 !important;
    color: #FFFFFF !important;
    border: none !important;
}
.stButton > button[kind="primary"]:hover {
    background-color: #225F32 !important;
    transform: translateY(-1px);
}
.stButton > button[kind="secondary"] {
    background-color: #FFFFFF !important;
    color: #2D7A40 !important;
    border: 1px solid #2D7A40 !important;
}

/* =============================================
   HINTERGRÜNDE – verhindert graue Boxen
   ============================================= */

.stApp {
    background-color: #F7F8F6 !important;
}
/* Container-Hintergrund (nicht sidebar) */
[data-testid="stVerticalBlock"] {
    background-color: transparent;
}
/* Hauptinhalt-Bereich */
.main .block-container {
    background-color: #F7F8F6 !important;
    padding-top: 2rem;
}

/* =============================================
   PLOTLY CHARTS – kein weisser Kasten
   ============================================= */

.js-plotly-plot,
.js-plotly-plot .plotly {
    background: transparent !important;
}

/* =============================================
   TRENNLINIEN
   ============================================= */

hr {
    border: none !important;
    border-top: 0.5px solid #E2E8E3 !important;
    margin: 16px 0 !important;
}

/* =============================================
   BENUTZERDEFINIERTE KLASSEN
   (werden via st.markdown(..., unsafe_allow_html=True) genutzt)
   ============================================= */

.ergebnis-kasten {
    background-color: #1C2B1E;
    border-radius: 10px;
    padding: 20px 24px;
    margin: 12px 0;
}
.ergebnis-kasten * {
    color: #FFFFFF !important;
}
.ergebnis-kasten .preis {
    font-size: 2rem;
    font-weight: 500;
}
.ergebnis-kasten .konfidenz {
    color: #7FCF8A !important;
    font-size: 0.8rem;
    margin-top: 4px;
}
.ergebnis-kasten .label {
    color: rgba(255,255,255,0.55) !important;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
}

.badge-gruen {
    display: inline-block;
    background-color: #E8F5EA;
    color: #1A6B3C !important;
    font-size: 0.75rem;
    padding: 3px 10px;
    border-radius: 20px;
    font-weight: 500;
}

</style>
""", unsafe_allow_html=True)


def configure_page() -> None:
    """Zweck: Konfiguriert Seite. Parameter: Keine. Rückgabe: Keine."""
    # Die Seitenkonfiguration und das CSS stehen bewusst ganz oben in der Datei.
    return
def check_login() -> bool:
    """Zweck: Prüft Projektpasswort. Parameter: Keine. Rückgabe: Loginstatus."""
    # Dies ist kein sicheres Login-System für Produktionsumgebungen, aber ausreichend für dieses Projekt.
    if st.session_state.get("eingeloggt") is True:
        return True
    login_seite_anzeigen()
    return False


def login_seite_anzeigen() -> None:
    """
    Zweck: Zeigt eine saubere, zentrierte Login-Seite an.
    Rückgabe: nichts
    """
    # Drei Spalten für Zentrierung: leer | Inhalt | leer
    links, mitte, rechts = st.columns([1, 1.5, 1])
    with mitte:
        # Etwas Abstand nach oben
        st.markdown("<div style='height: 80px;'></div>", unsafe_allow_html=True)
        # Login-Karte
        with st.container(border=True):
            # Titel
            st.markdown("""
            <div style="text-align: center; margin-bottom: 24px;">
                <div style="font-size: 1.4rem; font-weight: 500;
                     color: #1A3B1E;">ImmoInsight</div>
                <div style="font-size: 0.8rem; color: #7A8C7C;
                     margin-top: 4px;">
                     Schweizer Immobilienpreisschätzung
                </div>
            </div>
            """, unsafe_allow_html=True)
            # Passwortfeld
            passwort = st.text_input("Passwort", type="password", placeholder="Passwort eingeben...")
            # Login-Button
            if st.button("Anmelden", type="primary", use_container_width=True):
                if passwort == APP_PASSWORT:
                    st.session_state["eingeloggt"] = True
                    st.rerun()
                else:
                    st.error("Falsches Passwort. Bitte erneut versuchen.")
            # Hinweis
            st.markdown("""
            <div style="text-align: center; margin-top: 16px;
                 font-size: 0.7rem; color: #7A8C7C;">
                Universitätsprojekt - University of St. Gallen
            </div>
            """, unsafe_allow_html=True)


def sidebar_aufbauen() -> str:
    """
    Zweck: Erstellt die Navigations-Sidebar mit dunklem Design.
    Rückgabe: string – der Name der gewählten Seite
    """
    with st.sidebar:
        # App-Logo und Titel
        st.markdown("""
        <div style="padding-bottom: 16px; border-bottom: 0.5px solid
             rgba(255,255,255,0.15); margin-bottom: 16px;">
            <div style="font-size: 1.1rem; font-weight: 500; color: #fff;">
                ImmoInsight
            </div>
            <div style="font-size: 0.7rem; color: rgba(255,255,255,0.45);
                 margin-top: 2px;">
                Immobilienpreise Schweiz
            </div>
        </div>
        """, unsafe_allow_html=True)
        # Navigation als Radio-Buttons
        seite = st.radio("Navigation", options=PAGE_NAMES, label_visibility="collapsed")
        coordinate_refresh()
        # Abstand nach unten
        st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
        # Abmelden-Button am unteren Rand
        if st.button("Abmelden", type="secondary", use_container_width=True):
            st.session_state["eingeloggt"] = False
            st.rerun()
        return seite
@st.cache_data
def load_data(signature: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Zweck: Lädt Daten gecacht. Parameter: Signatur. Rückgabe: Drei DataFrames."""
    # st.cache_data speichert reine Daten und lädt erst nach Datenbankänderungen neu.
    properties = datenbank.get_properties()
    stats = datenbank.get_postcode_stats()
    runs = datenbank.get_model_runs()
    return properties, stats, runs
def clear_cache() -> None:
    """Zweck: Leert Cache. Parameter: Keine. Rückgabe: Keine."""
    # Nach Schreiboperationen müssen Diagramme neue Daten sehen.
    load_data.clear()
def show_shell() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object], str]:
    """Zweck: Lädt Grunddaten. Parameter: Keine. Rückgabe: Daten, Modell und Seite."""
    # Die Reihenfolge verhindert Zugriffe auf nicht erstellte Tabellen.
    datenbank.initialize_database()
    page = sidebar_aufbauen()
    st.title(APP_TITLE)
    st.caption("Deutsche Projekt-App zur Schätzung von Schweizer Immobilienpreisen.")
    properties, stats, runs = load_data(datenbank.get_signature())
    bundle = ml_modell.load_or_train(properties)
    runs = datenbank.get_model_runs()
    return properties, stats, runs, bundle, page
def coordinate_refresh() -> None:
    """Zweck: Bietet Koordinaten-Refresh. Parameter: Keine. Rückgabe: Keine."""
    # API-Aufrufe passieren nur auf Klick, weil die App offline funktionieren soll.
    with st.expander("Koordinaten aktualisieren"):
        st.markdown("Nominatim wird nur bei Klick aufgerufen.")
        if st.button("Koordinaten über API laden"):
            warnings = api_daten.refresh_all_coordinates()
            clear_cache()
            show_coordinate_message(warnings)
def show_coordinate_message(warnings: list[str]) -> None:
    """Zweck: Meldet API-Ergebnis. Parameter: Warnungen. Rückgabe: Keine."""
    # Deutsche Meldungen zeigen klar, ob die API funktioniert hat.
    if len(warnings) == 0:
        st.success("Koordinaten wurden aktualisiert.")
    else:
        st.warning("Einige Koordinaten konnten nicht aktualisiert werden.")
def postcode_fields(key: str) -> tuple[int, str, str]:
    """Zweck: Wählt PLZ und zeigt Ort. Parameter: Widget-Schlüssel. Rückgabe: PLZ, Ort, Kanton."""
    # Eine feste PLZ-Liste verhindert Tippfehler und hält das Modell stabil.
    postcode = st.selectbox("PLZ", POSTCODE_LIST, key=key + "_plz")
    place = POSTCODE_LOOKUP[postcode][0]
    canton = POSTCODE_LOOKUP[postcode][1]
    st.text_input("Ort", value=place, disabled=True, key=key + "_ort")
    st.info("Ausgewählt: " + place + ", Kanton " + canton)
    return postcode, place, canton
def make_inputs(prefix: str) -> dict[str, float | int | str]:
    """Zweck: Sammelt Objektdaten. Parameter: Prefix. Rückgabe: Eingabe-Dictionary."""
    # Drei Spalten machen das Formular ruhig und übersichtlich.
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        postcode, place, canton = postcode_fields(prefix)
        area = st.slider("Wohnfläche in Quadratmetern", 25.0, 300.0, 90.0, 1.0, key=prefix + "_area")
    with col_b:
        rooms = st.slider("Zimmer", 1.0, 8.0, 3.5, 0.5, key=prefix + "_rooms")
        floor = st.slider("Etage", 0, 25, 3, 1, key=prefix + "_floor")
    with col_c:
        year = st.slider("Baujahr", 1900, 2026, 2000, 1, key=prefix + "_year")
        parking = st.checkbox("Parkplatz vorhanden", key=prefix + "_parking")
        garden = st.checkbox("Garten vorhanden", key=prefix + "_garden")
    return pack_inputs(postcode, place, canton, area, rooms, floor, year, parking, garden)
def pack_inputs(postcode: int, place: str, canton: str, area: float, rooms: float, floor: int, year: int, parking: bool, garden: bool) -> dict[str, float | int | str]:
    """Zweck: Bündelt Formularwerte. Parameter: Werte. Rückgabe: Dictionary."""
    # Gleiche Schlüssel werden für Datenbank und Modell genutzt.
    data: dict[str, float | int | str] = {}
    data["postcode"] = postcode
    data["place"] = place
    data["canton"] = canton
    data["area_m2"] = float(area)
    data["rooms"] = float(rooms)
    data["floor"] = int(floor)
    data["year_built"] = int(year)
    data["has_parking"] = int(parking)
    data["has_garden"] = int(garden)
    return data
def card(label: str, value: str, note: str) -> str:
    """Zweck: Baut Metrikkarte. Parameter: Texte. Rückgabe: HTML."""
    # Einheitliche Karten machen Kennzahlen schneller vergleichbar.
    html = "<div class='metric-card'><div class='metric-label'>" + label + "</div>"
    html = html + "<div class='metric-value'>" + value + "</div>"
    html = html + "<div class='metric-note'>" + note + "</div></div>"
    return html
def chf(value: float) -> str:
    """Zweck: Formatiert CHF. Parameter: Zahl. Rückgabe: Text."""
    # Apostrophe sind in der Schweiz übliche Tausendertrennzeichen.
    return "CHF " + f"{value:,.0f}".replace(",", "'")


def preis_anzeigen(preis: float, konfidenz_intervall: float, modell_name: str) -> None:
    """
    Zweck: Zeigt den geschätzten Preis in einer auffälligen dunklen Box an.
    Parameter: preis – float, der geschätzte Preis in CHF
               konfidenz_intervall – float, +/- Bereich
               modell_name – string, Name des verwendeten Modells
    Rückgabe: nichts (direkte Streamlit-Ausgabe)
    """
    # Zahlen in lesbares Schweizer Format umwandeln (z.B. 1'250'000)
    preis_formatiert = f"CHF {preis:,.0f}".replace(",", "'")
    konfidenz_formatiert = f"+/- CHF {konfidenz_intervall:,.0f}".replace(",", "'")
    # HTML-Box ausgeben
    st.markdown(f"""
    <div class="ergebnis-kasten">
        <div class="label">Geschätzter Kaufpreis</div>
        <div class="preis">{preis_formatiert}</div>
        <div class="konfidenz">{konfidenz_formatiert} Konfidenzbereich</div>
        <div style="margin-top: 12px; font-size: 0.75rem; color: rgba(255,255,255,0.45);">
            Modell: {modell_name}
        </div>
    </div>
    """, unsafe_allow_html=True)


def karte_anzeigen(titel: str, inhalt_funktion) -> None:
    """
    Zweck: Zeigt einen Inhaltsbereich in einer weissen Karte an.
    Parameter: titel – string, Kartentitel
               inhalt_funktion – eine Funktion die den Inhalt rendert
    Rückgabe: nichts
    """
    # Container mit weissem Hintergrund und Rahmen
    with st.container(border=True):
        # Grauer Trennstrich unter dem Titel
        st.markdown(f"**{titel}**")
        st.markdown("<hr>", unsafe_allow_html=True)
        # Eigentlichen Inhalt rendern
        inhalt_funktion()
def estimator_page(bundle: dict[str, object], stats: pd.DataFrame) -> None:
    """Zweck: Zeigt Preisschätzung. Parameter: Modell und Statistiken. Rückgabe: Keine."""
    # Diese Seite ist der wichtigste Nutzerfluss.
    st.subheader("Preisschätzung")
    model_name = st.radio("Modell auswählen", ml_modell.MODEL_NAMES, horizontal=True)
    inputs = make_inputs("estimate")
    features = features_from_inputs(inputs, stats)
    prediction = ml_modell.predict_price(bundle, model_name, features)
    prediction_cards(prediction, inputs, model_name)
    price_gauge(prediction, int(inputs["postcode"]))
    if st.button("Diese Schätzung speichern"):
        datenbank.insert_property(inputs, float(prediction["price"]), "Schätzung")
        clear_cache()
        st.success("Die Schätzung wurde gespeichert.")
def features_from_inputs(inputs: dict[str, float | int | str], stats: pd.DataFrame) -> dict[str, float]:
    """Zweck: Ergänzt Standortfeatures. Parameter: Eingaben und Statistiken. Rückgabe: Features."""
    # Das Modell braucht zur PLZ auch Einkommen und Bevölkerungsdichte.
    postcode = int(inputs["postcode"])
    row = stats[stats["postcode"] == postcode].iloc[0]
    return ml_modell.make_feature_dictionary(inputs, row)
def prediction_cards(prediction: dict[str, float], inputs: dict[str, float | int | str], model_name: str) -> None:
    """Zweck: Zeigt Ergebnis-Karten. Parameter: Prognose, Eingaben, Modell. Rückgabe: Keine."""
    # Preis, Quadratmeterpreis und Bereich sind die drei wichtigsten Aussagen.
    price = float(prediction["price"])
    area = float(inputs["area_m2"])
    interval = max(float(prediction["upper"]) - price, price - float(prediction["lower"]))
    preis_anzeigen(price, interval, model_name)
    columns = st.columns(2)
    columns[0].markdown(card("Preis pro Quadratmeter", chf(price / area), "Aus Modellpreis berechnet"), unsafe_allow_html=True)
    columns[1].markdown(card("Unsicherheitsbereich", chf(prediction["lower"]) + " bis " + chf(prediction["upper"]), "Vereinfachte Schätzung"), unsafe_allow_html=True)
def price_gauge(prediction: dict[str, float], postcode: int) -> None:
    """Zweck: Zeigt Preisvergleich. Parameter: Prognose und PLZ. Rückgabe: Keine."""
    # Der Tachometer vergleicht Schätzung und lokalen Durchschnitt.
    average = datenbank.get_average_price_for_postcode(postcode)
    maximum = max(float(prediction["price"]), average) * 1.5
    gauge = {"axis": {"range": [0, maximum]}, "bar": {"color": "#1A6B3C"}}
    indicator = go.Indicator(mode="gauge+number+delta", value=prediction["price"], delta={"reference": average}, title={"text": "Vergleich mit PLZ-Durchschnitt"}, gauge=gauge)
    figure = go.Figure(indicator)
    figure.update_layout(height=320, margin={"l": 20, "r": 20, "t": 60, "b": 10})
    figure = ml_modell.plotly_design_anwenden(figure)
    st.plotly_chart(figure, use_container_width=True)
def market_page(properties: pd.DataFrame, stats: pd.DataFrame) -> None:
    """Zweck: Zeigt Marktübersicht. Parameter: Immobilien und Statistiken. Rückgabe: Keine."""
    # Filter wirken auf alle Marktgrafiken.
    st.subheader("Marktübersicht")
    selected = st.sidebar.multiselect("PLZ filtern", POSTCODE_LIST, default=POSTCODE_LIST)
    filtered = properties[properties["postcode"].isin(selected)].copy()
    karte_anzeigen("Öffentliche Daten", opendata_box)
    summary = market_summary(filtered, stats)
    map_chart(summary)
    col_a, col_b = st.columns(2)
    with col_a:
        bar_chart(summary)
    with col_b:
        scatter_chart(filtered)
    heatmap_chart(filtered)
def opendata_box() -> None:
    """Zweck: Zeigt opendata.swiss-Treffer. Parameter: Keine. Rückgabe: Keine."""
    # Die Box zeigt, dass echte öffentliche Datensätze gesucht werden.
    result = api_daten.search_housing_datasets()
    if result["ok"] is True:
        st.info("opendata.swiss meldet " + str(result["count"]) + " Datensätze zu Immobilien und Wohnen.")
    else:
        st.warning(str(result["message"]))
def market_summary(properties: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Zweck: Berechnet Werte je PLZ. Parameter: Daten. Rückgabe: DataFrame."""
    # Einfache Schleifen ersetzen komplexe groupby-agg-Ausdrücke.
    rows = []
    for postcode in POSTCODE_LIST:
        part = properties[properties["postcode"] == postcode]
        if len(part) > 0:
            rows.append(summary_row(postcode, part, stats))
    return pd.DataFrame(rows)
def summary_row(postcode: int, part: pd.DataFrame, stats: pd.DataFrame) -> dict[str, float | int | str]:
    """Zweck: Baut eine PLZ-Zeile. Parameter: PLZ, Teilmenge, Statistik. Rückgabe: Zeile."""
    # Koordinaten kommen aus der PLZ-Statistik.
    stat = stats[stats["postcode"] == postcode].iloc[0]
    row: dict[str, float | int | str] = {}
    row["postcode"] = postcode
    row["place"] = str(stat["place"])
    row["lat"] = float(stat["lat"])
    row["lon"] = float(stat["lon"])
    row["average_price"] = float(part["estimated_price"].mean())
    row["standard_deviation"] = float(part["estimated_price"].std())
    row["count"] = int(len(part))
    return row
def map_chart(summary: pd.DataFrame) -> None:
    """Zweck: Zeigt Karte. Parameter: Summary. Rückgabe: Keine."""
    # Blasengrösse und Farbe zeigen das Preisniveau je PLZ.
    figure = px.scatter_mapbox(summary, lat="lat", lon="lon", size="average_price", color="average_price", hover_name="place", hover_data={"postcode": True, "count": True}, color_continuous_scale="Greens", zoom=6, height=420, labels={"average_price": "Durchschnittspreis", "postcode": "PLZ", "count": "Anzahl", "place": "Ort"})
    figure.update_layout(mapbox_style="open-street-map", margin={"l": 0, "r": 0, "t": 10, "b": 0})
    figure = ml_modell.plotly_design_anwenden(figure)
    st.plotly_chart(figure, use_container_width=True)
def bar_chart(summary: pd.DataFrame) -> None:
    """Zweck: Zeigt Balkendiagramm. Parameter: Summary. Rückgabe: Keine."""
    # Fehlerbalken zeigen die Streuung innerhalb eines Orts.
    figure = px.bar(summary, x="place", y="average_price", error_y="standard_deviation", color="average_price", color_continuous_scale="Greens", labels={"place": "Ort", "average_price": "Durchschnittspreis", "standard_deviation": "Streuung"}, title="Durchschnittspreis je Ort")
    figure = ml_modell.plotly_design_anwenden(figure)
    st.plotly_chart(figure, use_container_width=True)
def scatter_chart(properties: pd.DataFrame) -> None:
    """Zweck: Zeigt Fläche gegen Preis. Parameter: Immobilien. Rückgabe: Keine."""
    # Die Wohnfläche ist für Immobilienpreise besonders gut erklärbar.
    figure = px.scatter(properties, x="area_m2", y="estimated_price", color="place", labels={"area_m2": "Wohnfläche in Quadratmetern", "estimated_price": "Preis in CHF", "place": "Ort"}, title="Wohnfläche und Preis")
    add_line(figure, properties)
    figure = ml_modell.plotly_design_anwenden(figure)
    st.plotly_chart(figure, use_container_width=True)
def add_line(figure: go.Figure, properties: pd.DataFrame) -> None:
    """Zweck: Ergänzt Trendlinie. Parameter: Figur und Daten. Rückgabe: Keine."""
    # Die Linie zeigt den linearen Grundzusammenhang.
    if len(properties) < 2:
        return
    slope, intercept = np.polyfit(properties["area_m2"], properties["estimated_price"], 1)
    x_values = np.array([properties["area_m2"].min(), properties["area_m2"].max()])
    y_values = slope * x_values + intercept
    figure.add_trace(go.Scatter(x=x_values, y=y_values, mode="lines", name="Trendlinie", line={"color": "#1A6B3C"}))
def heatmap_chart(properties: pd.DataFrame) -> None:
    """Zweck: Zeigt Heatmap. Parameter: Immobilien. Rückgabe: Keine."""
    # Die Heatmap verbindet Ort und Zimmerzahl.
    table = heatmap_table(properties)
    figure = px.imshow(table, aspect="auto", color_continuous_scale="Greens", labels={"x": "Zimmer", "y": "Ort", "color": "Preis in CHF"}, title="Preis nach Ort und Zimmerzahl")
    figure = ml_modell.plotly_design_anwenden(figure)
    st.plotly_chart(figure, use_container_width=True)
def heatmap_table(properties: pd.DataFrame) -> pd.DataFrame:
    """Zweck: Baut Heatmap-Tabelle. Parameter: Immobilien. Rückgabe: Matrix."""
    # Die Schleife bleibt bewusst einfach statt einer komplexen Pivot-Aggregation.
    places = sorted(properties["place"].unique())
    rooms = sorted(properties["rooms"].unique())
    table = pd.DataFrame(index=places, columns=rooms)
    for place in places:
        for room in rooms:
            part = properties[properties["place"] == place]
            part = part[part["rooms"] == room]
            table.loc[place, room] = float(part["estimated_price"].mean()) if len(part) > 0 else 0.0
    return table.astype(float)
def explorer_page(properties: pd.DataFrame) -> None:
    """Zweck: Zeigt Datenexplorer. Parameter: Immobilien. Rückgabe: Keine."""
    # Der Explorer macht die SQLite-Daten sichtbar und exportierbar.
    st.subheader("Datenexplorer")
    filtered = explorer_filter(properties)
    summary_cards(filtered)
    st.dataframe(rename_columns(filtered), use_container_width=True, hide_index=True)
    download(filtered)
    manual_entry()
def explorer_filter(properties: pd.DataFrame) -> pd.DataFrame:
    """Zweck: Filtert Explorer. Parameter: Immobilien. Rückgabe: Gefilterte Daten."""
    # Sidebar-Filter halten die Hauptfläche frei.
    selected = st.sidebar.multiselect("PLZ im Explorer", POSTCODE_LIST, default=POSTCODE_LIST)
    price_min = int(properties["estimated_price"].min())
    price_max = int(properties["estimated_price"].max())
    area_min = float(properties["area_m2"].min())
    area_max = float(properties["area_m2"].max())
    price_range = st.sidebar.slider("Preisbereich", price_min, price_max, (price_min, price_max), step=10000)
    area_range = st.sidebar.slider("Flächenbereich", area_min, area_max, (area_min, area_max), step=1.0)
    return datenbank.filter_properties(selected, price_range, area_range)
def summary_cards(filtered: pd.DataFrame) -> None:
    """Zweck: Zeigt Kennzahlen. Parameter: Gefilterte Daten. Rückgabe: Keine."""
    # Mittelwert, Median und Streuung fassen die Tabelle zusammen.
    columns = st.columns(3)
    columns[0].markdown(card("Mittelwert", chf(float(filtered["estimated_price"].mean())), "Gefilterte Daten"), unsafe_allow_html=True)
    columns[1].markdown(card("Median", chf(float(filtered["estimated_price"].median())), "Robuster Lagewert"), unsafe_allow_html=True)
    columns[2].markdown(card("Standardabweichung", chf(float(filtered["estimated_price"].std())), "Preisstreuung"), unsafe_allow_html=True)
def rename_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Zweck: Übersetzt Spaltennamen. Parameter: DataFrame. Rückgabe: DataFrame."""
    # Intern bleiben englische Variablennamen, extern sieht man deutsche Namen.
    names = {"id": "ID", "postcode": "PLZ", "place": "Ort", "canton": "Kanton", "area_m2": "Wohnfläche", "rooms": "Zimmer", "floor": "Etage", "has_parking": "Parkplatz", "has_garden": "Garten", "year_built": "Baujahr", "estimated_price": "Preis", "source": "Quelle", "created_at": "Zeitpunkt"}
    return data.rename(columns=names)
def download(filtered: pd.DataFrame) -> None:
    """Zweck: Bietet CSV an. Parameter: Gefilterte Daten. Rückgabe: Keine."""
    # CSV ist einfach mit Tabellenprogrammen weiterverwendbar.
    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    filename = "immoinsight_export_" + datetime.now().strftime("%Y%m%d_%H%M") + ".csv"
    st.download_button("Gefilterte Daten als CSV herunterladen", csv_bytes, file_name=filename, mime="text/csv")
def manual_entry() -> None:
    """Zweck: Speichert manuelle Immobilie. Parameter: Keine. Rückgabe: Keine."""
    # Manuelle Eingaben zeigen die Persistenz der Datenbank.
    st.subheader("Neue Immobilie erfassen")
    with st.form("manual_form"):
        inputs = make_inputs("manual")
        price = st.number_input("Bekannter Preis in CHF", min_value=100000, max_value=9000000, value=850000, step=10000)
        submitted = st.form_submit_button("Immobilie speichern")
    if submitted:
        datenbank.insert_property(inputs, float(price), "Manuell")
        clear_cache()
        st.success("Die Immobilie wurde gespeichert.")
def training_page(properties: pd.DataFrame, runs: pd.DataFrame) -> None:
    """Zweck: Zeigt Modellseite. Parameter: Daten und Läufe. Rückgabe: Keine."""
    # Die Seite erklärt und bewertet die drei Modelle.
    st.subheader("Modelltraining und Leistung")
    model_explainers()
    bundle = ml_modell.load_or_train(properties)
    if st.button("Modelle neu trainieren"):
        bundle = train_again(properties)
    st.dataframe(ml_modell.metrics_to_dataframe(bundle), use_container_width=True, hide_index=True)
    importance_chart(bundle)
    history_chart(runs)
def model_explainers() -> None:
    """Zweck: Erklärt Modelle. Parameter: Keine. Rückgabe: Keine."""
    # Expander halten die Seite kurz und bieten trotzdem Erklärung.
    with st.expander("Lineare Regression"):
        st.markdown("Dieses Modell sucht eine gerade Linie durch die Daten. Es ist gut, wenn Zusammenhänge ungefähr linear sind.")
    with st.expander("Random Forest"):
        st.markdown("Dieses Modell kombiniert viele einfache Entscheidungsbäume. Es ist gut, wenn Merkmale unterschiedlich zusammenwirken.")
    with st.expander("Gradient Boosting"):
        st.markdown("Dieses Modell verbessert mehrere Bäume Schritt für Schritt. Es ist gut, wenn Fehler nacheinander reduziert werden sollen.")
def train_again(properties: pd.DataFrame) -> dict[str, object]:
    """Zweck: Trainiert neu. Parameter: Immobilien. Rückgabe: Bundle."""
    # Der Spinner zeigt, dass gerade gerechnet wird.
    with st.spinner("Modelle werden neu trainiert."):
        bundle = ml_modell.train_and_save(properties, True)
    st.success("Die Modelle wurden neu trainiert.")
    return bundle
def importance_chart(bundle: dict[str, object]) -> None:
    """Zweck: Zeigt Wichtigkeiten. Parameter: Bundle. Rückgabe: Keine."""
    # Nutzer wählen, welches Modell erklärt werden soll.
    selected = st.radio("Modell für Wichtigkeit", ml_modell.MODEL_NAMES, horizontal=True, key="importance")
    data = ml_modell.get_feature_importance(bundle, selected)
    figure = px.bar(data, x="Wichtigkeit", y="Merkmal", orientation="h", color="Wichtigkeit", color_continuous_scale="Greens", title="Wichtigkeit der Merkmale")
    figure = ml_modell.plotly_design_anwenden(figure)
    st.plotly_chart(figure, use_container_width=True)
def history_chart(runs: pd.DataFrame) -> None:
    """Zweck: Zeigt Trainingshistorie. Parameter: Läufe. Rückgabe: Keine."""
    # R-Quadrat über Zeit zeigt die Modellentwicklung.
    if len(runs) == 0:
        st.info("Es gibt noch keine Trainingshistorie.")
        return
    figure = px.line(runs, x="trained_at", y="r2", color="model_name", markers=True, labels={"trained_at": "Trainingszeit", "r2": "R-Quadrat", "model_name": "Modell"}, title="Trainingshistorie")
    figure = ml_modell.plotly_design_anwenden(figure)
    st.plotly_chart(figure, use_container_width=True)
def about_page() -> None:
    """Zweck: Zeigt Projektinfos. Parameter: Keine. Rückgabe: Keine."""
    # Diese Seite ersetzt die entfernte Contribution Matrix.
    st.subheader("Über die App")
    karte_anzeigen("Projektbeschreibung", about_project_content)
    karte_anzeigen("Technologien", about_tech_content)
    karte_anzeigen("Datenquellen", about_sources_content)
    karte_anzeigen("KI-Nutzung", about_ai_content)


def about_project_content() -> None:
    """Zweck: Zeigt Projekttext. Parameter: Keine. Rückgabe: Keine."""
    # Der Text erklärt den Zweck der App kurz und verständlich.
    st.markdown("ImmoInsight ist eine Lern-App zur Schätzung von Schweizer Immobilienpreisen. Die App verbindet SQLite, Visualisierung, öffentliche APIs und maschinelles Lernen. Die Seed-Daten sind synthetisch, aber an realistischen Schweizer Preisniveaus orientiert. Das Projekt ist so aufgebaut, dass Studierende die einzelnen Schritte erklären können.")


def about_tech_content() -> None:
    """Zweck: Zeigt Technologien. Parameter: Keine. Rückgabe: Keine."""
    # Die Liste hält die technischen Bausteine übersichtlich.
    st.markdown("- Streamlit\n- SQLite\n- Pandas und NumPy\n- scikit-learn\n- Plotly")


def about_sources_content() -> None:
    """Zweck: Zeigt Datenquellen. Parameter: Keine. Rückgabe: Keine."""
    # Links machen die öffentlichen Quellen direkt überprüfbar.
    st.markdown("[OpenStreetMap Nominatim](https://nominatim.openstreetmap.org/) und [opendata.swiss](https://opendata.swiss/) werden als Datenquellen verwendet.")


def about_ai_content() -> None:
    """Zweck: Zeigt KI-Hinweis. Parameter: Keine. Rückgabe: Keine."""
    # Der Hinweis unterstützt transparente Projektdokumentation.
    st.info("Hinweis zur KI-Nutzung: KI-Unterstützung kann gemäss HSG-Richtlinien transparent in der Projektdokumentation deklariert werden.")
def route(page: str, properties: pd.DataFrame, stats: pd.DataFrame, runs: pd.DataFrame, bundle: dict[str, object]) -> None:
    """Zweck: Wählt Seite. Parameter: Seite und Daten. Rückgabe: Keine."""
    # Einfache if-Blöcke sind für Anfänger gut nachvollziehbar.
    if page == "Preisschätzung":
        estimator_page(bundle, stats)
    if page == "Marktübersicht":
        market_page(properties, stats)
    if page == "Daten-Explorer":
        explorer_page(properties)
    if page == "ML-Modell":
        training_page(properties, runs)
    if page == "Über die App":
        about_page()
def main() -> None:
    """Zweck: Startet die App. Parameter: Keine. Rückgabe: Keine."""
    # Login schützt die Projekt-App vor sofortigem Zugriff.
    configure_page()
    if check_login() is False:
        return
    properties, stats, runs, bundle, page = show_shell()
    route(page, properties, stats, runs, bundle)
# Der Einstiegspunkt bleibt sichtbar und einfach.
if __name__ == "__main__":
    main()
