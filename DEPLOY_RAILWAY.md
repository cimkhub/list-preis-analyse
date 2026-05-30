# Railway Deployment: Kunden-Review-App

Diese Deployment-Variante ist nur für Kundenprüfung gedacht. Die Pipeline kann online nicht gestartet werden.

Diese App ist ein eigenständiges Railway-Repository. In Railway bleibt das Root Directory leer bzw. auf dem Repository-Root.

## Railway Service

Railway erkennt die App über:

- `railway.json`
- `nixpacks.toml`
- `requirements-dashboard.txt`

Beim Erstellen des Services kein separates Root Directory setzen.

Startbefehl:

```bash
python dashboard_app.py
```

## Environment Variables

In Railway setzen:

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

`PORT` setzt Railway automatisch. Der Server bindet dann automatisch auf `0.0.0.0:$PORT`.

## Railway Volume

Ein Railway Volume auf `/app/runtime` mounten.

Die Ergebnisdaten liegen danach in:

```text
/app/runtime/data
/app/runtime/parsed
/app/runtime/images
```

Die App schreibt Kundenfeedback in:

```text
/app/runtime/data/qa_marks_KWXX_YYYY.json
```

Wenn `DASHBOARD_DATA_ZIP_URL` gesetzt ist und im Volume noch keine Daten liegen, lädt die App beim Start das ZIP herunter und entpackt es nach `/app/runtime`.

## Datenbundle erstellen

Nach lokalem Pipeline-Lauf:

```bash
python3 export_dashboard_bundle.py --week 20 --year 2026
```

Alle vorhandenen Review-Daten:

```bash
python3 export_dashboard_bundle.py --all-review-data
```

Das erzeugt:

```text
dashboard_review_data.zip
```

Dieses ZIP kann in R2/S3 hochgeladen werden. Es enthält nur die benötigten Review-Daten:

- `parsed/KW*_YYYY`
- passende `data/<anbieter>/<jahr>/<kw>` Ordner mit PDFs und `relevance_decisions.json`
- `data/qa_marks_KW*.json`, falls vorhanden
- PDF-Relevanz-Vorschauen aus `images/_relevance`

## Sicherheitslogik

Im Review-Modus:

- `/api/runs` mit `POST` ist serverseitig deaktiviert.
- Der Menüpunkt `Pipeline starten` wird im Frontend ausgeblendet.
- Scraping, LLM-Aufrufe und Matching können aus der Kunden-App nicht gestartet werden.
- Login ist per Passwort geschützt.
