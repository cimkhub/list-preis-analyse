import os
import json
import requests
from pathlib import Path
from datetime import date, timedelta
from urllib.parse import urlparse

from google import genai


def load_env_file(path=".env"):
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


DAYS_BACK = 7
MAX_RESULTS_PER_QUERY = 10
MAX_ITEMS_FOR_GEMINI = 20

QUERIES = [
    ("Schweinefleisch Angebot Nachfrage"),
    ("Schweinefleisch Preise"),
    ("Schweinefleisch Tierseuche"),
    ("pork Europe market prices"),
]


def build_freshness_range(days_back=DAYS_BACK):
    end_date = date.today()
    start_date = end_date - timedelta(days=days_back)
    return f"{start_date.isoformat()}to{end_date.isoformat()}"


def brave_news_search(api_key, query, freshness, count=10, offset=0):
    url = "https://api.search.brave.com/res/v1/news/search"
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key,
    }
    params = {
        "q": query,
        "freshness": freshness,
        "country": "ALL",
        "ui_lang": "de-DE",
        "count": count,
        "offset": offset,
        "spellcheck": False,
        "extra_snippets": True,
        "safesearch": "moderate",
    }

    print("\n" + "=" * 80)
    print("BRAVE REQUEST")
    print("=" * 80)
    print("URL:", url)
    print("Params:", json.dumps(params, ensure_ascii=False, indent=2))

    response = requests.get(url, headers=headers, params=params, timeout=60)

    print("\nBRAVE RESPONSE STATUS:", response.status_code)
    print("BRAVE RESPONSE HEADERS:")
    for k, v in response.headers.items():
        if k.lower().startswith("x-") or k.lower() in {"content-type"}:
            print(f"  {k}: {v}")

    try:
        response.raise_for_status()
    except Exception:
        print("\nBRAVE ERROR BODY:")
        print(response.text[:3000])
        raise

    data = response.json()

    print("\nBRAVE RESPONSE TOP-LEVEL KEYS:", list(data.keys()))
    if "results" in data:
        print("NUMBER OF RESULTS:", len(data["results"]))
    else:
        print("NO 'results' KEY IN RESPONSE")
        print(json.dumps(data, ensure_ascii=False, indent=2)[:5000])

    return data


def hostname_from_item(item):
    meta = item.get("meta_url", {})
    host = (meta.get("hostname") or "").lower()
    if host:
        return host
    url = item.get("url", "")
    return urlparse(url).netloc.lower()


def normalize_url(url):
    if not url:
        return ""
    return url.split("?")[0].rstrip("/")


def dedupe_results(results):
    print("\n" + "=" * 80)
    print("DEDUPE START")
    print("=" * 80)
    print("Input results:", len(results))

    seen_urls = set()
    seen_titles = set()
    deduped = []

    for idx, item in enumerate(results, start=1):
        url = normalize_url(item.get("url", ""))
        title = (item.get("title") or "").strip().lower()

        duplicate_reason = None
        if url and url in seen_urls:
            duplicate_reason = "duplicate url"
        elif title and title in seen_titles:
            duplicate_reason = "duplicate title"

        if duplicate_reason:
            print(f"SKIP #{idx}: {duplicate_reason}")
            print("  Title:", item.get("title", ""))
            print("  URL:", item.get("url", ""))
            continue

        if url:
            seen_urls.add(url)
        if title:
            seen_titles.add(title)

        deduped.append(item)

    print("Output deduped results:", len(deduped))
    return deduped


def fetch_news(api_key):
    freshness = build_freshness_range(DAYS_BACK)
    raw_results = []
    per_query_debug = []

    print("\n" + "=" * 80)
    print("FETCH NEWS START")
    print("=" * 80)
    print("Freshness:", freshness)
    print("Queries:", len(QUERIES))

    for i, query in enumerate(QUERIES, start=1):
        print("\n" + "-" * 80)
        print(f"QUERY {i}/{len(QUERIES)}")
        print(query)

        data = brave_news_search(
            api_key=api_key,
            query=query,
            freshness=freshness,
            count=MAX_RESULTS_PER_QUERY,
            offset=0,
        )

        results = data.get("results", [])
        print(f"RESULTS FOR QUERY {i}: {len(results)}")

        if results:
            print("FIRST 3 TITLES:")
            for item in results[:3]:
                print(" -", item.get("title", "Ohne Titel"))
        else:
            print("NO RESULTS FOR THIS QUERY")

        per_query_debug.append({
            "query": query,
            "num_results": len(results),
            "raw_response": data,
        })

        raw_results.extend(results)

    print("\n" + "=" * 80)
    print("FETCH NEWS SUMMARY")
    print("=" * 80)
    print("Total raw results across all queries:", len(raw_results))

    deduped = dedupe_results(raw_results)

    print("Deduped results:", len(deduped))
    print("Taking first MAX_ITEMS_FOR_GEMINI =", MAX_ITEMS_FOR_GEMINI)

    return freshness, raw_results, deduped[:MAX_ITEMS_FOR_GEMINI], per_query_debug


def build_llm_input(items, freshness):
    lines = [f"Zeitraum: {freshness}", "", "News-Items:"]
    for i, item in enumerate(items, start=1):
        lines.append(f"{i}. Titel: {item.get('title', '')}")
        lines.append(f"   Quelle: {hostname_from_item(item)}")
        lines.append(f"   Alter: {item.get('age', '')}")
        lines.append(f"   URL: {item.get('url', '')}")
        lines.append(f"   Kurztext: {item.get('description', '')}")

        extras = item.get("extra_snippets", [])
        for snippet in extras[:2]:
            lines.append(f"   Snippet: {snippet}")

        lines.append("")
    return "\n".join(lines)


def summarize_with_gemini(gemini_key, items, freshness):
    print("\n" + "=" * 80)
    print("GEMINI START")
    print("=" * 80)
    print("Items sent to Gemini:", len(items))

    client = genai.Client(api_key=gemini_key)
    news_context = build_llm_input(items, freshness)

    print("\nGEMINI INPUT PREVIEW:")
    print(news_context[:4000])
    print("\n--- END OF PREVIEW ---")

    prompt = f"""
Du bist Marktanalyst für Schweinefleisch in Deutschland und Europa.

Nutze AUSSCHLIESSLICH die unten bereitgestellten News-Items.
Erfinde nichts hinzu. Nutze kein Weltwissen außerhalb dieser Liste.

Deine Aufgabe:
1. Prüfe, welche News-Items wirklich relevant für die Preisentwicklung von Schweinefleisch sind.
2. Ignoriere irrelevante, schwache oder doppelte Meldungen.
3. Fasse nur die relevanten Entwicklungen zusammen.
4. Gib nur Bullet Points aus.
5. Jeder Bullet Point beschreibt genau eine Entwicklung.
6. Jeder Bullet Point endet mit:
   - (+), wenn die News eher auf HÖHERE Schweinefleischpreise hindeutet
   - (-), wenn die News eher auf NIEDRIGERE Schweinefleischpreise hindeutet
7. Wenn mehrere Artikel dasselbe Signal zeigen, fasse sie zu einem Bullet Point zusammen.
8. Nenne pro Bullet Point 1 bis 2 Quellen in Klammern.
9. Wenn etwas zu indirekt, zu unsicher oder nicht preisrelevant ist, lasse es weg.
10. Schreibe auf Deutsch.
11. Keine Einleitung.
12. Kein Fazit.
13. Keine Nummerierung.

Format pro Zeile:
- <Aussage>. (Quelle: domain1, domain2) (+/-)

Hier sind die News-Items:

{news_context}
""".strip()

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    print("\nGEMINI RAW RESPONSE PREVIEW:")
    print((response.text or "")[:3000])

    return response.text.strip()


def write_summary_file(summary_text, freshness, items):
    output_path = Path("schweinefleisch_summary.txt")

    with output_path.open("w", encoding="utf-8") as f:
        f.write("Schweinefleisch – relevante News-Signale\n")
        f.write(f"Zeitraum: {freshness}\n")
        f.write("\n")
        f.write(summary_text)
        f.write("\n\n")
        f.write("Verwendete Artikel:\n")
        for item in items:
            f.write(f"- {item.get('title', '')}\n")
            f.write(f"  Quelle: {hostname_from_item(item)}\n")
            f.write(f"  URL: {item.get('url', '')}\n")

    return output_path


def main():
    print("\n" + "=" * 80)
    print("SCRIPT START")
    print("=" * 80)

    load_env_file(".env")

    brave_api_key = os.getenv("BRAVE_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    print("BRAVE_API_KEY found:", bool(brave_api_key))
    print("GEMINI_API_KEY found:", bool(gemini_api_key))

    if not brave_api_key:
        raise RuntimeError("BRAVE_API_KEY fehlt in .env")
    if not gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY fehlt in .env")

    freshness, raw_results, items, per_query_debug = fetch_news(brave_api_key)

    with open("brave_news_pork_raw.json", "w", encoding="utf-8") as f:
        json.dump(raw_results, f, ensure_ascii=False, indent=2)

    with open("brave_news_pork_deduped.json", "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    with open("brave_news_pork_per_query_debug.json", "w", encoding="utf-8") as f:
        json.dump(per_query_debug, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print(f"AKTUELLE NEWS ZU SCHWEINEFLEISCH ({freshness})")
    print("=" * 80)
    print("Raw results:", len(raw_results))
    print("Deduped items:", len(items))

    if not items:
        print("\nKeine News-Treffer gefunden.")
        print("Check these files:")
        print("- brave_news_pork_raw.json")
        print("- brave_news_pork_deduped.json")
        print("- brave_news_pork_per_query_debug.json")
        return

    for i, item in enumerate(items, start=1):
        print(f"\n{i}. {item.get('title', 'Ohne Titel')}")
        print(f"   Quelle: {hostname_from_item(item)}")
        print(f"   Alter: {item.get('age', '')}")
        print(f"   URL: {item.get('url', '')}")
        if item.get("description"):
            print(f"   Kurztext: {item.get('description')}")

    summary = summarize_with_gemini(gemini_api_key, items, freshness)

    print("\n" + "=" * 80)
    print("ZUSAMMENFASSUNG")
    print("=" * 80)
    print(summary)

    summary_path = write_summary_file(summary, freshness, items)
    print(f"\nZusammenfassung gespeichert in: {summary_path}")


if __name__ == "__main__":
    main()