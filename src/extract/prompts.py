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
  "product_family": "fleisch|fisch|obst_gemuese|mopro|wurst|sonstiges|unknown",
  "temperature_state": "fresh|chilled|frozen|thawed|ambient|unknown",
  "processing_state": "raw_plain|raw_cut|raw_minced|raw_formed|raw_skewered|raw_seasoned|marinated|sauced|cooked|fried|smoked|cured|pickled|preserved|ready_to_eat|unknown",
  "calibre": "Sichtbares Kaliber bzw. sichtbare Größen-/Gewichtsspanne als Text oder null",
  "source_brand": "Direkt am Produkt sichtbare Marke, exakt geschrieben, oder null",
  "brand_evidence": "Exakter sichtbarer Text/Logo-Text, der die Produktmarke belegt, oder null",
  "brand_evidence_source": "product_name|description|image|unknown",
  "certifications": ["Explizit sichtbare Zertifizierung wie ASC, MSC, BIO, HALAL oder QS"] oder [],
  "unit": "kg|g|l|ml|stueck|packung|beutel|karton|flasche|bund|schale",
  "quantity": Legacy-Mengenwert als Zahl oder null,
  "price": Preis als Dezimalzahl (Punkt, nicht Komma),
  "price_basis": "per_kg|per_100g|per_liter|per_100ml|per_piece|per_package|unknown",
  "price_is_net": true wenn Netto-Preis (mit * markiert), sonst false,
  "price_gross": Brutto-Preis wenn separat angegeben oder null,
  "price_tiers": [{{"min_qty": Mindestmenge, "price": Preis}}] oder null,
  "package_count": Anzahl gleichartiger Inhaltseinheiten im angebotenen Gebinde als ganze Zahl oder null,
  "package_size_value": Inhalt je Einheit als Zahl oder null,
  "package_size_unit": "g|kg|ml|l|piece|unknown",
  "total_content_value": Gesamtinhalt des angebotenen Gebindes als Zahl oder null,
  "total_content_unit": "g|kg|ml|l|piece|unknown",
  "packaging_type": "bag|pack|box|crate|basket|tray|bucket|bottle|can|bundle|piece|unknown",
  "packaging_raw": "Sichtbare Packungs-/Mengenangabe wortgetreu oder null",
  "confidence": 0.0-1.0 Vertrauen in die Extraktion
}}

Regeln:
- Preise mit * sind NETTO (vor MwSt). Setze price_is_net: true
- Preise in Klammern nach *-Preisen sind BRUTTO
- Verwende Dezimalpunkt: 11.99 nicht 11,99
- Hochgestellte Ziffern (große "11" kleine "99") → 11.99
- product_family beschreibt die fachliche Produktfamilie unabhängig von Temperatur und category. Tiefgekühlter Fisch bleibt product_family "fisch", tiefgekühltes Fleisch bleibt "fleisch"
- temperature_state nur aus sichtbaren oder eindeutig produktbezogenen Angaben ableiten. "erntefrisch" allein ist KEIN Hinweis auf "frozen"
- processing_state beschreibt die Produktform bzw. Verarbeitung. Bei fehlender oder widersprüchlicher Evidenz "unknown" verwenden
- calibre separat und wortgetreu erfassen; nicht in Produktname oder Verpackungsmenge umdeuten
- source_brand nur setzen, wenn die Marke direkt dem Produkt zugeordnet ist. Händler-/Supplier-Logo, Seitenkopf und Eigenwerbung sind keine Produktmarke
- source_brand nicht übersetzen oder als Synonym normalisieren. brand_evidence wortgetreu und brand_evidence_source passend zur sichtbaren Quelle setzen
- ASC, MSC, BIO, HALAL, QS und vergleichbare Siegel sind Zertifizierungen und keine Marken. Nur sichtbare Zertifizierungen in certifications aufnehmen
- Trenne Inhaltsmenge strikt von Verpackungsart. g, kg, ml und l sind Inhalts-/Maßeinheiten und NIEMALS eine Anzahl von Beuteln, Packungen, Kartons oder Flaschen
- Beispiel "2500 g Beutel": package_count 1, package_size_value 2500, package_size_unit "g", total_content_value 2500, total_content_unit "g", packaging_type "bag", packaging_raw "2500 g Beutel"
- Beispiel "1000 g Packung": package_count 1, package_size_value 1000, package_size_unit "g", total_content_value 1000, total_content_unit "g", packaging_type "pack"
- Beispiel "10 x 80 g, Gesamt 800 g pro Packung": package_count 10, package_size_value 80, package_size_unit "g", total_content_value 800, total_content_unit "g", packaging_type "pack"
- Angaben wie "100/200 Stück/lb" oder "8/12" sind Kaliber und NIEMALS package_count. Erfasse sie ausschließlich in calibre
- Legacy-Felder unit/quantity: Wenn ein Gesamtinhalt sichtbar ist, verwende dessen Maßeinheit und Menge (z.B. quantity 2500 und unit "g"), niemals quantity 2500 und unit "beutel". Nur wenn ausschließlich eine Packungsanzahl sichtbar ist, darf die Verpackungsart als unit verwendet werden
- price_basis beschreibt, wofür der große Angebotspreis gilt. Ein zusätzlicher Grundpreis "kg = ..." gehört in price_per_kg; der große Preis eines Beutels bleibt price_basis "per_package"
- Wenn count × Einzelinhalt und sichtbarer Gesamtinhalt angegeben sind, müssen sie nach g/kg- bzw. ml/l-Umrechnung zusammenpassen
- Mehrere echte Preis-/Packungsvarianten als mehrere getrennte Produktobjekte ausgeben. Preisstaffeln derselben Variante bleiben in price_tiers
- Wenn die Zuordnung von Anzahl, Einzelinhalt, Gesamtinhalt oder Verpackungsart unklar ist: packaging_raw wortgetreu erhalten, unklare strukturierte Felder auf null bzw. "unknown" setzen und confidence < 0.7 verwenden
- Wenn Produktfamilie, Temperatur, Verarbeitung oder Markenquelle nicht sicher bestimmbar sind, jeweils "unknown" bzw. null verwenden
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
