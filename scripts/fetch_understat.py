#!/usr/bin/env python3
"""
Scarica xG medi stagione da Understat (top-5 europee) e salva data/understat_xg.csv.

Uso locale:
  pip install requests pandas lxml
  python scripts/fetch_understat.py

GitHub Actions: eseguito dallo workflow .github/workflows/update-understat.yml

Understat espone i dati in variabili JS nelle pagine lega; non è un'API ufficiale.
Usare con moderazione (1 run/giorno).
"""
from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "understat_xg.csv"

# slug Understat → etichetta
LEAGUES = {
    "Serie_A": "Serie A",
    "EPL": "Premier League",
    "La_liga": "La Liga",
    "Bundesliga": "Bundesliga",
    "Ligue_1": "Ligue 1",
}

UA = {"User-Agent": "ComboMasterGrook-UnderstatBot/1.0 (+github actions daily xG)"}


def stagione_corrente() -> int:
    oggi = date.today()
    return oggi.year if oggi.month >= 7 else oggi.year - 1


def _extract_json(html: str, var_name: str):
    """Estrae JSON da `var teamsData = JSON.parse('...')`."""
    m = re.search(rf"var {var_name}\s*=\s*JSON\.parse\('(.+?)'\)", html, re.S)
    if not m:
        return None
    raw = m.group(1).encode("utf-8").decode("unicode_escape")
    return json.loads(raw)


def fetch_league(slug: str, season: int) -> list[dict]:
    url = f"https://understat.com/league/{slug}/{season}"
    r = requests.get(url, headers=UA, timeout=30)
    r.raise_for_status()
    data = _extract_json(r.text, "teamsData")
    if not data:
        return []
    rows = []
    for _, t in data.items():
        hist = t.get("history") or []
        if not hist:
            continue
        n = len(hist)
        xgf = sum(float(h.get("xG", 0)) for h in hist) / n
        xga = sum(float(h.get("xGA", 0)) for h in hist) / n
        npx = sum(float(h.get("npxG", h.get("xG", 0))) for h in hist) / n
        rows.append({
            "team": t.get("title") or t.get("team_name") or "",
            "league": LEAGUES.get(slug, slug),
            "season": season,
            "matches": n,
            "xg_for": round(xgf, 3),
            "xg_against": round(xga, 3),
            "npx_g_for": round(npx, 3),
        })
    return rows


def main() -> int:
    season = stagione_corrente()
    all_rows: list[dict] = []
    for slug in LEAGUES:
        try:
            rows = fetch_league(slug, season)
            print(f"{slug}: {len(rows)} squadre", file=sys.stderr)
            all_rows.extend(rows)
        except Exception as e:
            print(f"ERRORE {slug}: {e}", file=sys.stderr)
    if not all_rows:
        print("Nessun dato scaricato", file=sys.stderr)
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows).sort_values(["league", "team"])
    df.to_csv(OUT, index=False)
    print(f"Scritto {OUT} ({len(df)} righe)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
