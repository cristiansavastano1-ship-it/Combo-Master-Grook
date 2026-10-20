#!/usr/bin/env python3
"""
Scarica rating ClubElo dalle pagine paese (clubelo.com/ITA ecc.) → data/clubelo.csv
L'API api.clubelo.com è spesso giù; lo scraping HTML delle ranking page è più stabile.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "clubelo.csv"

PAGES = ["ITA", "ENG", "ESP", "GER", "FRA", "NED"]
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
}

# normalizza nomi ClubElo → più vicini a football-data.co.uk
RENAME = {
    "Internazionale": "Inter",
    "Bayern München": "Bayern",
    "Bayern Munchen": "Bayern",
    "Atlético": "Atletico",
    "Atletico Madrid": "Atletico",
    "Man City": "Man City",
    "Man United": "Man United",
    "Paris SG": "Paris SG",
    "Forest": "Nott'm Forest",
    "Milan": "Milan",
}


def fetch_country(code: str) -> list[dict]:
    r = requests.get(f"https://clubelo.com/{code}", timeout=35, headers=HEADERS)
    r.raise_for_status()
    text = r.text
    names = [(m.start(), m.group(1)) for m in re.finditer(r'"Name":\s*"([^"]+)"', text)]
    elos = [(m.start(), float(m.group(1))) for m in re.finditer(r'"Elo":\s*([0-9]+\.[0-9]+)', text)]
    rows = []
    for pos, name in names:
        prev = [e for p, e in elos if p < pos and pos - p < 800]
        if not prev:
            continue
        # unicode escapes
        try:
            name = bytes(name, "utf-8").decode("unicode_escape")
        except Exception:
            pass
        name = RENAME.get(name, name)
        if name.startswith("Bayern"):
            name = "Bayern"
        if "tl" in name.lower() and "tico" in name.lower():
            name = "Atletico"
        rows.append({"Club": name, "Elo": round(prev[-1], 1)})
    return rows


def main() -> int:
    all_rows: list[dict] = []
    for code in PAGES:
        try:
            rows = fetch_country(code)
            print(f"{code}: {len(rows)} squadre", file=sys.stderr)
            all_rows.extend(rows)
        except Exception as e:
            print(f"ERRORE {code}: {e}", file=sys.stderr)

    if not all_rows:
        print("Nessun dato ClubElo. CSV esistente non cancellato.", file=sys.stderr)
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows).drop_duplicates(subset=["Club"], keep="first")
    df = df.sort_values("Elo", ascending=False)
    df.to_csv(OUT, index=False)
    print(f"Scritto {OUT} ({len(df)} righe)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
