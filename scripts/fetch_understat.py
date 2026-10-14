#!/usr/bin/env python3
"""
Scarica xG medi stagione da Understat (top-5) → data/understat_xg.csv

Nota 2026: Understat spesso non espone più teamsData nel HTML iniziale.
Se non ottiene dati, esce 0 senza cancellare un CSV già valido.
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

LEAGUES = {
    "Serie_A": "Serie A",
    "EPL": "Premier League",
    "La_liga": "La Liga",
    "Bundesliga": "Bundesliga",
    "Ligue_1": "Ligue 1",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://understat.com/",
}


def stagione_corrente() -> int:
    oggi = date.today()
    return oggi.year if oggi.month >= 7 else oggi.year - 1


def _extract_teams_data(html: str):
    patterns = [
        r"var teamsData\s*=\s*JSON\.parse\('(.+?)'\)",
        r'var teamsData\s*=\s*JSON\.parse\("(.+?)"\)',
        r"teamsData\s*=\s*JSON\.parse\('(.+?)'\)",
        r"teamsData\s*=\s*(\{.+?\})\s*;",
    ]
    for pat in patterns:
        m = re.search(pat, html, re.S)
        if not m:
            continue
        raw = m.group(1)
        try:
            if raw.startswith("{"):
                return json.loads(raw)
            decoded = raw.encode("utf-8").decode("unicode_escape")
            return json.loads(decoded)
        except Exception:
            continue
    return None


def fetch_league(slug: str, season: int) -> list[dict]:
    url = f"https://understat.com/league/{slug}/{season}"
    r = requests.get(url, headers=HEADERS, timeout=45)
    r.raise_for_status()
    data = _extract_teams_data(r.text)
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
    seasons = [stagione_corrente(), stagione_corrente() - 1]
    all_rows: list[dict] = []
    for season in seasons:
        for slug in LEAGUES:
            try:
                rows = fetch_league(slug, season)
                print(f"{slug} {season}: {len(rows)} squadre", file=sys.stderr)
                all_rows.extend(rows)
            except Exception as e:
                print(f"ERRORE {slug} {season}: {e}", file=sys.stderr)
        if all_rows:
            break

    if not all_rows:
        print(
            "Nessun dato Understat (sito bloccato o struttura cambiata). "
            "L'app funziona comunque con Dixon-Coles + ClubElo + forma. "
            "CSV esistente non cancellato.",
            file=sys.stderr,
        )
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows).drop_duplicates(subset=["team", "league"], keep="first")
    df = df.sort_values(["league", "team"])
    df.to_csv(OUT, index=False)
    print(f"Scritto {OUT} ({len(df)} righe)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
