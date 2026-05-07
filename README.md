# ImmoInsight - Schweizer Immobilienpreisschätzer
ImmoInsight ist eine deutschsprachige Streamlit-App zur Schätzung von Immobilienpreisen in der Schweiz. Die App nutzt eine lokale SQLite-Datenbank, synthetische Seed-Daten, öffentliche API-Abfragen und drei einfache Machine-Learning-Modelle.
## So startest du die App
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
Danach öffnest du im Browser:
```text
http://localhost:8501
```
Das Projektpasswort lautet:
```text
immo2026
```
## Dateien
- `app.py`: Haupt-App mit Login, Navigation und allen Seiten.
- `datenbank.py`: SQLite-Tabellen, Seed-Daten und Datenbankabfragen.
- `ml_modell.py`: Training, Vorhersage und Modellmetriken.
- `api_daten.py`: API-Aufrufe für Nominatim und opendata.swiss.
- `konfiguration.py`: Konstanten, PLZ-Lookup und Passwort.
- `requirements.txt`: Benötigte Python-Pakete.
## Funktionen
- Preisschätzung mit PLZ, Ort, Fläche, Zimmern, Etage, Baujahr und Ausstattung.
- Automatische Anzeige von Ort und Kanton nach PLZ-Auswahl.
- Marktübersicht mit Karte, Balkendiagramm, Streudiagramm und Heatmap.
- Datenexplorer mit Filtern, Tabelle, CSV-Export und manueller Eingabe.
- Modellvergleich mit Linearer Regression, Random Forest und Gradient Boosting.
- Einfache Passwortabfrage für den Projektzugang.
## Datenquellen
- Nominatim von OpenStreetMap für Geocoding: https://nominatim.openstreetmap.org/
- opendata.swiss für die Suche nach öffentlichen Wohn- und Immobiliendatensätzen: https://opendata.swiss/
- Lokale synthetische Seed-Daten für die vollständige Offline-Nutzung.
## Hinweis zur KI-Nutzung
KI-Unterstützung kann gemäss HSG-Richtlinien transparent in der Projektdokumentation deklariert werden. Die Verantwortung für Verständnis, Prüfung und Abgabe bleibt beim Projektteam.
