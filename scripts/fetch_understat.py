#!/usr/bin/env python3
"""
Scarica xG medi stagione corrente da Understat (top-5) → data/understat_xg.csv
Usa endpoint AJAX /getLeagueData (il parsing HTML non funziona più).
Se matches < 8, blend con stagione precedente.
"""
from __future__ import annotations

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
    "La_Liga": "La Liga",
    "Bundesliga": "Bundesliga",
    "Ligue_1": "Ligue 1",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "X-Requested-With": "XMLHttpRequest",
}

MIN_MATCHES_FULL = 8


def stagione_corrente() -> int:
    oggi = date.today()
    return oggi.year if oggi.month >= 7 else oggi.year - 1


def fetch_league(slug: str, season: int) -> list[dict]:
    session = requests.Session()
    session.headers.update(HEADERS)
    session.headers["Referer"] = f"https://understat.com/league/{slug}/{season}"
    try:
        session.get(f"https://understat.com/league/{slug}/{season}", timeout=30)
        r = session.get(
            f"https://understat.com/getLeagueData/{slug}/{season}", timeout=30
        )
        if r.status_code != 200:
            return []
        data = r.json()
        teams = data.get("teams") or {}
        rows = []
        for _, t in teams.items():
            name = t.get("title") or ""
            hist = t.get("history") or []
            if not hist or not name:
                continue
            n = len(hist)
            xgf = sum(float(h.get("xG") or 0) for h in hist) / n
            xga = sum(float(h.get("xGA") or 0) for h in hist) / n
            npx = sum(float(h.get("npxG") or h.get("xG") or 0) for h in hist) / n
            rows.append({
                "team": name,
                "league": slug,
                "season": season,
                "matches": n,
                "xg_for": round(xgf, 3),
                "xg_against": round(xga, 3),
                "npx_g_for": round(npx, 3),
            })
        return rows
    except Exception as e:
        print(f"ERRORE {slug} {season}: {e}", file=sys.stderr)
        return []


def blend(now_rows: list[dict], prev_rows: list[dict]) -> list[dict]:
    prev = {r["team"]: r for r in prev_rows}
    out = []
    for r in now_rows:
        n = r["matches"]
        if n < MIN_MATCHES_FULL and r["team"] in prev:
            w = min(0.7, n / MIN_MATCHES_FULL)
            p = prev[r["team"]]
            r = dict(r)
            r["xg_for"] = round(w * r["xg_for"] + (1 - w) * p["xg_for"], 3)
            r["xg_against"] = round(w * r["xg_against"] + (1 - w) * p["xg_against"], 3)
            if "npx_g_for" in r and "npx_g_for" in p:
                r["npx_g_for"] = round(w * r["npx_g_for"] + (1 - w) * p["npx_g_for"], 3)
        out.append(r)
    return out


def main() -> int:
    season = stagione_corrente()
    all_rows: list[dict] = []
    for slug in LEAGUES:
        now = fetch_league(slug, season)
        prev = fetch_league(slug, season - 1)
        print(f"{slug} {season}: {len(now)} | prev: {len(prev)}", file=sys.stderr)
        if now:
            all_rows.extend(blend(now, prev))
        elif prev:
            all_rows.extend(prev)

    if not all_rows:
        print(
            "Nessun dato Understat. CSV esistente non cancellato.",
            file=sys.stderr,
        )
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows).drop_duplicates(subset=["team", "league"], keep="first")
    df = df.sort_values(["league", "xg_for"], ascending=[True, False])
    df.to_csv(OUT, index=False)
    print(f"Scritto {OUT} ({len(df)} righe)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
