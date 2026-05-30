# LIST Preisvergleich Review-App

Standalone Kunden-Review-App für Railway.

Diese App enthält nur das Dashboard zum Prüfen vorhandener Ergebnisdaten. Scraping, LLM-Extraktion, Produktmatching und Pipeline-Start sind im Kundenmodus deaktiviert.

## Lokal starten

```bash
DASHBOARD_MODE=review DASHBOARD_PORT=8768 python3 dashboard_app.py
```

Danach öffnen:

```text
http://127.0.0.1:8768
```

Standard-Passwort:

```text
List2026
```

## Erwartete Datenstruktur

Für lokale Tests können die Runtime-Daten direkt neben der App liegen:

```text
data/
parsed/
images/
```

Auf Railway sollten diese Ordner in einem Volume liegen:

```text
/app/runtime/data
/app/runtime/parsed
/app/runtime/images
```

## Railway Variablen

```bash
DASHBOARD_MODE=review
DASHBOARD_PASSWORD=List2026
DASHBOARD_AUTH_SECRET=<lange-zufaellige-zeichenfolge>
DASHBOARD_RUNTIME_ROOT=/app/runtime
DASHBOARD_DATA_ROOT=/app/runtime/data
DASHBOARD_PARSED_ROOT=/app/runtime/parsed
DASHBOARD_IMAGE_ROOT=/app/runtime/images
DASHBOARD_DATA_ZIP_URL=<r2-oder-s3-zip-url>
```

Railway setzt `PORT` automatisch.

## Railway Volume

Mount Path:

```text
/app/runtime
```

## Neue Woche exportieren

Im lokalen Pipeline-Projekt:

```bash
python3 export_dashboard_bundle.py --week 20 --year 2026
```

Alle vorhandenen Review-Daten als ein ZIP für R2 exportieren:

```bash
python3 export_dashboard_bundle.py --all-review-data
```

Das ZIP enthält:

```text
data/
parsed/
images/_relevance/
```

Für Railway + R2 das ZIP in R2 hochladen und `DASHBOARD_DATA_ZIP_URL` auf die öffentliche oder signierte ZIP-URL setzen. Beim Start lädt die App das ZIP in `/app/runtime`, wenn dort noch keine Daten liegen.
