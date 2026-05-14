SYSTEM_PROMPT = """Du bist ein Experte für die Extraktion von Produktdaten aus deutschen Lebensmittelgroßhändler-Angebotsflyern.
Extrahiere ALLE Produkte auf der Seite als JSON-Array.
Antworte NUR mit validem JSON. Kein Markdown, keine Erklärung."""

RELEVANCE_SYSTEM_PROMPT = """Du bewertest Angebots-PDFs für einen Lebensmittelgroßhändler-Preisvergleich.
Entscheide nur, ob ein Dokument für die vollständige Produktanalyse relevant ist.
Antworte NUR mit validem JSON. Kein Markdown, keine Erklärung."""

EXTRACTION_PROMPT = """Analysiere dieses Bild einer Angebotsseite vom Lebensmittelgroßhändler {supplier}.

Extrahiere ALLE sichtbaren Produkte als JSON-Array. Für jedes Produkt:

{{
  "product_name": "Exakter deutscher Produktname",
  "description": "Zusatzinfos (Herkunft, Qualität, Zubereitung) oder null",
  "origin": "Herkunftsland/-region oder null",
  "category": "fleisch|fisch|obst_gemuese|tk|wurst|mopro|sonstiges",
  "unit": "kg|stueck|packung|beutel|karton|flasche|bund|schale",
  "quantity": Verpackungsgröße als Zahl oder null,
  "price": Preis als Dezimalzahl (Punkt, nicht Komma),
  "price_is_net": true wenn Netto-Preis (mit * markiert), sonst false,
  "price_gross": Brutto-Preis wenn separat angegeben oder null,
  "price_tiers": [{{"min_qty": Mindestmenge, "price": Preis}}] oder null,
  "confidence": 0.0-1.0 Vertrauen in die Extraktion
}}

Regeln:
- Preise mit * sind NETTO (vor MwSt). Setze price_is_net: true
- Preise in Klammern nach *-Preisen sind BRUTTO
- Verwende Dezimalpunkt: 11.99 nicht 11,99
- Hochgestellte Ziffern (große "11" kleine "99") → 11.99
- Überspringe Nicht-Lebensmittel komplett
- Bei Unsicherheit: confidence < 0.7
- Leere Seiten oder Seiten ohne Produkte → leeres Array []"""

RELEVANCE_PROMPT = """Bewerte die Relevanz dieses PDF-Dokuments für einen Lebensmittelgroßhändler-Preisvergleich.

Kontext:
- Supplier: {supplier}
- Dateiname: {filename}
- Dokumenttitel: {title}
- Tab/Zustand: {tab}
- Dateiname relevante Hinweise: {relevant_hits}
- Dateiname irrelevante Hinweise: {irrelevant_hits}

Entscheide anhand des Dateinamens und der ERSTEN Seite, ob das Dokument vollständig weiter analysiert werden soll.

Antworte als JSON-Objekt:
{{
  "is_relevant": true,
  "relevance_label": "relevant_food_offer|relevant_mixed_offer|irrelevant_non_food_only|irrelevant_catalog_or_order_guide|irrelevant_marketing_or_magazine|unclear",
  "reason": "Kurze Begründung auf Deutsch",
  "valid_from": "YYYY-MM-DD oder null",
  "valid_to": "YYYY-MM-DD oder null",
  "market_scope": "all|specific|unknown",
  "valid_markets": ["Markt/Stadt 1", "Markt/Stadt 2"],
  "confidence": 0.0
}}

Regeln:
- relevant_food_offer: reguläres Wochenangebot oder Food-Flyer mit klaren Lebensmittelpreisen oder mindestens einem klar sichtbaren Food-Produkt auf der ersten Seite
- relevant_mixed_offer: gemischter Angebotsflyer mit mindestens einem klar relevanten Lebensmittelangebot auf der ersten Seite
- irrelevant_non_food_only: nur Non-Food, Möbel, Geräte, Geschirr, Kleidung, Einrichtung
- irrelevant_catalog_or_order_guide: Bestellkatalog, Saisonkatalog, Konzeptheft, Produktübersicht ohne typischen Angebotscharakter
- irrelevant_marketing_or_magazine: Magazin, Inspirationsheft, Menüheft, redaktioneller Inhalt, Branding
- Wenn die erste Seite mindestens ein klares Lebensmittelprodukt zeigt, im Zweifel relevant
- Wenn es um Outdoor, Garten, Non-Food, Möbel, Ausstattung, Technik, Textilien oder Geschirr ohne Lebensmittelprodukt geht, irrelevant
- Lies den sichtbaren Angebotszeitraum von der ersten Seite, falls vorhanden, und gib valid_from/valid_to als ISO-Datum zurück
- Wenn kein Angebotszeitraum erkennbar ist, setze valid_from und valid_to auf null
- Prüfe, ob auf der ersten Seite teilnehmende Märkte/Betriebe/Standorte genannt werden, z.B. "Teilnehmende Betriebe", "Basis Nord", "SELGROS Braunschweig" oder Standortlisten
- Wenn konkrete teilnehmende Märkte/Standorte sichtbar sind, setze market_scope auf "specific" und gib die erkannten Märkte/Städte in valid_markets zurück
- Wenn ausdrücklich alle Märkte gelten oder keine Marktbeschränkung sichtbar ist, setze market_scope auf "all" und valid_markets auf []
- Wenn die Marktinformation nicht lesbar/unklar ist, setze market_scope auf "unknown"
- Wenn unklar, setze is_relevant auf true nur dann, wenn die erste Seite wie ein echter Angebotsflyer wirkt"""

SUPPLIER_HINTS = {
    "metro": """
Besonderheiten Metro:
- Alle Preise sind NETTO (mit * markiert), Brutto in Klammern
- Staffelpreise: "ab X kg" mit verschiedenen Preisstufen
- Erfasse ALLE Preisstufen in price_tiers
- Hauptpreis = niedrigste Stufe (größte Menge)""",

    "selgros": """
Besonderheiten Selgros:
- Preise sind BRUTTO (inkl. MwSt)
- Artikelnummern ignorieren
- Preise haben oft hochgestellte Cent-Beträge
- Achte auf kleine Mengenangaben unter dem Preis""",

    "edeka": """
Besonderheiten EDEKA Foodservice:
- Preise sind BRUTTO (Abholpreise inkl. MwSt)
- "Sie sparen X%" als Zusatzinfo erfassen
- Achte auf regionale Kennzeichnung (Nord/Süd)""",

    "handelshof": """
Besonderheiten Handelshof:
- Preise sind BRUTTO (inkl. MwSt)
- Aktionspreise sind oft rot markiert
- Grundpreis pro kg beachten wenn angegeben""",
}


def get_extraction_prompt(supplier: str) -> str:
    hint = SUPPLIER_HINTS.get(supplier, "")
    return EXTRACTION_PROMPT.format(supplier=supplier) + hint


def get_relevance_prompt(
    supplier: str,
    filename: str,
    title: str | None,
    tab: str | None,
    relevant_hits: list[str],
    irrelevant_hits: list[str],
) -> str:
    return RELEVANCE_PROMPT.format(
        supplier=supplier,
        filename=filename,
        title=title or "-",
        tab=tab or "-",
        relevant_hits=", ".join(relevant_hits) if relevant_hits else "keine",
        irrelevant_hits=", ".join(irrelevant_hits) if irrelevant_hits else "keine",
    )
