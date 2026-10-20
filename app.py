# Combo Master Grook — v4.12
# Dixon-Coles + walk-forward + calibrazione + confronto quote
# ClubElo offline (data/clubelo.csv) + Understat xG offline

import difflib
import io
import math
import re
import unicodedata
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, gammaln

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(
    page_title="Combo Master Grook",
    layout="wide",
    page_icon="⚽",
)

# --- DESIGN ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    #MainMenu, footer, header { visibility: hidden; }
    .stApp { background-color: #0b0f19; color: #f3f4f6; }
    .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1280px; }
    [data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid #1f2937; }
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white; border: none; border-radius: 10px; font-weight: 600;
        padding: 0.5rem 1rem; transition: all 0.25s ease;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        box-shadow: 0 6px 16px rgba(59, 130, 246, 0.5);
        transform: translateY(-1px);
    }
    [data-testid="stMetric"] {
        background-color: #111827; border: 1px solid #1f2937;
        padding: 15px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    [data-testid="stMetricValue"] { color: #38bdf8 !important; font-weight: 700 !important; }
    div[data-testid="stTabs"] button { font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
                padding: 22px 25px; border-radius: 18px; border: 1px solid #312e81;
                text-align: center; margin-bottom: 18px;
                box-shadow: 0 20px 25px -5px rgba(0,0,0,0.4);">
        <h1 style="color:#f8fafc; font-weight:800; font-size:1.95rem; margin:0; letter-spacing:-0.5px;">
            ⚽ COMBO MASTER GROOK <span style="font-size:0.85rem; color:#818cf8; font-weight:600;">v4.12</span>
        </h1>
        <p style="color:#94a3b8; font-size:0.92rem; margin-top:6px;">
            Dixon-Coles · walk-forward · calibrazione · confronto quote reali
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ======================= COSTANTI =======================
LEAGUES = {
    "Serie A (Italia)": "I1",
    "Serie B (Italia)": "I2",
    "Premier League (Inghilterra)": "E0",
    "Championship (Inghilterra)": "E1",
    "La Liga (Spagna)": "SP1",
    "Bundesliga (Germania)": "D1",
    "Ligue 1 (Francia)": "F1",
    "Eredivisie (Olanda)": "N1",
    "Champions League (UEFA)": "CL",
    "Europa League (UEFA)": "EL",
}
# football-data.org competition codes
API_CODICI = {
    "I1": "SA", "I2": "SB", "E0": "PL", "E1": "ELC",
    "SP1": "PD", "D1": "BL1", "F1": "FL1", "N1": "DED",
    "CL": "CL", "EL": "EL",
}
# Divisione europee: niente CSV su football-data.co.uk → storico da football-data.org
EURO_DIVS = {"CL", "EL"}
# ID leghe API-Football (api-sports.io) — free tier: 100 req/giorno
API_FOOTBALL_LEAGUE_ID = {
    "I1": 135,   # Serie A
    "I2": 136,   # Serie B
    "E0": 39,    # Premier League
    "E1": 40,    # Championship
    "SP1": 140,  # La Liga
    "D1": 78,    # Bundesliga
    "F1": 61,    # Ligue 1
    "N1": 88,    # Eredivisie
    "CL": 2,     # UEFA Champions League
    "EL": 3,     # UEFA Europa League
}
BASE_URL = "https://football-data.co.uk"
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"
CLUBELO_URL = "http://api.clubelo.com"  # CSV giornaliero, nessuna API key
# quanto pesare Elo rispetto al solo Dixon-Coles (0 = ignora, 1 = molto)
ELO_BLEND = 0.35
ELO_SCALE = 400.0
FORM_N = 6              # ultime N partite per la forma
FORM_BLEND = 0.25       # peso forma recente sui λ (0=off)
XG_BLEND = 0.20         # peso Understat xG, fisso (niente slider)
UNDERSTAT_PATHS = ("data/understat_xg.csv", "understat_xg.csv")
UA = {"User-Agent": "Mozilla/5.0 (compatible; ComboMasterGrook/4.9)"}
N_STAGIONI = 3
N_STAGIONI_TEST = 5
VERSIONE_TEST = 13  # invalida cache test quando cambia la struttura
W_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
FINESTRA_GIORNI = 1000
PASSO_BACKTEST = 14
PASSO_PARAM = 28

COL_PRESA = [("AvgH", "AvgD", "AvgA"), ("B365H", "B365D", "B365A"), ("PSH", "PSD", "PSA"), ("MaxH", "MaxD", "MaxA")]
COL_CHIUSURA = [("PSCH", "PSCD", "PSCA"), ("AvgCH", "AvgCD", "AvgCA"), ("B365CH", "B365CD", "B365CA")]
COL_OU_PRESA = [("Avg>2.5", "Avg<2.5"), ("B365>2.5", "B365<2.5"), ("P>2.5", "P<2.5"), ("Max>2.5", "Max<2.5")]
COL_OU_CHIUSURA = [("AvgC>2.5", "AvgC<2.5"), ("PC>2.5", "PC<2.5"), ("B365C>2.5", "B365C<2.5")]
STAT_COLONNE = {"Tiri": ("HS", "AS"), "Tiri in Porta": ("HST", "AST"), "Corner": ("HC", "AC"), "Falli": ("HF", "AF")}

_STOP_NOMI = {
    "fc", "ac", "as", "cf", "afc", "acf", "ssc", "us", "ud", "rc", "rcd", "sv", "vfl", "vfb",
    "fsv", "tsg", "fk", "cd", "ca", "sc", "bv", "de", "calcio", "ogc", "sco",
    "cfc", "sporting", "club", "milano",
}
_ALIAS_NOMI = {
    # Italia
    "inter": "inter", "internazionale": "inter",
    "internazionale milano": "inter", "inter milan": "inter", "inter milano": "inter",
    "fc internazionale milano": "inter",
    "verona": "hellas verona", "hellas verona": "hellas verona",
    "milan": "milan", "ac milan": "milan",
    "roma": "roma", "as roma": "roma",
    "lazio": "lazio", "napoli": "napoli", "juventus": "juventus",
    "atalanta": "atalanta", "fiorentina": "fiorentina",
    "bologna": "bologna", "torino": "torino", "genoa": "genoa",
    "udinese": "udinese", "cagliari": "cagliari", "lecce": "lecce",
    "parma": "parma", "como": "como", "monza": "monza",
    "venezia": "venezia", "empoli": "empoli", "sassuolo": "sassuolo",
    "pisa": "pisa", "cremonese": "cremonese", "frosinone": "frosinone",
    "spezia": "spezia", "palermo": "palermo", "bari": "bari",
    "catanzaro": "catanzaro", "cesena": "cesena", "modena": "modena",
    "cittadella": "cittadella", "sudtirol": "sudtirol", "sud tirol": "sudtirol",
    # Inghilterra
    "man united": "manchester united", "man utd": "manchester united",
    "manchester utd": "manchester united", "man city": "manchester city",
    "nott m forest": "nottingham forest", "nottingham": "nottingham forest",
    "wolves": "wolverhampton", "wolverhampton wanderers": "wolverhampton",
    "newcastle": "newcastle", "newcastle united": "newcastle",
    "west ham": "west ham", "west ham united": "west ham",
    "tottenham": "tottenham", "spurs": "tottenham",
    "brighton": "brighton", "brighton and hove albion": "brighton",
    "leicester": "leicester", "leicester city": "leicester",
    "southampton": "southampton", "ipswich": "ipswich",
    "crystal palace": "crystal palace", "fulham": "fulham",
    "brentford": "brentford", "bournemouth": "bournemouth",
    "everton": "everton", "aston villa": "aston villa",
    "chelsea": "chelsea", "arsenal": "arsenal", "liverpool": "liverpool",
    # Spagna
    "ath madrid": "atletico", "atletico madrid": "atletico",
    "ath bilbao": "athletic", "athletic bilbao": "athletic", "athletic club": "athletic",
    "espanol": "espanyol", "espanyol barcelona": "espanyol",
    "real madrid": "real madrid", "barcelona": "barcelona", "barca": "barcelona",
    "sevilla": "sevilla", "real sociedad": "sociedad", "sociedad": "sociedad",
    "villarreal": "villarreal", "betis": "betis", "real betis": "betis",
    "valencia": "valencia", "osasuna": "osasuna", "getafe": "getafe",
    "celta": "celta", "celta vigo": "celta", "mallorca": "mallorca",
    "girona": "girona", "las palmas": "las palmas", "alaves": "alaves",
    "rayo vallecano": "rayo vallecano", "leganes": "leganes",
    # Germania
    "bayern munich": "bayern", "bayern munchen": "bayern", "bayern": "bayern",
    "m gladbach": "gladbach", "borussia monchengladbach": "gladbach",
    "ein frankfurt": "frankfurt", "eintracht frankfurt": "frankfurt",
    "dortmund": "dortmund", "borussia dortmund": "dortmund",
    "leverkusen": "leverkusen", "bayer leverkusen": "leverkusen",
    "rb leipzig": "leipzig", "leipzig": "leipzig",
    "wolfsburg": "wolfsburg", "stuttgart": "stuttgart",
    "hoffenheim": "hoffenheim", "freiburg": "freiburg",
    "werder bremen": "werder", "bremen": "werder",
    "mainz": "mainz", "augsburg": "augsburg", "union berlin": "union berlin",
    "heidenheim": "heidenheim", "st pauli": "st pauli", "koln": "koln", "koeln": "koln",
    # Francia
    "paris sg": "paris sg", "paris saint germain": "paris sg", "psg": "paris sg",
    "lyon": "lyon", "olympique lyonnais": "lyon",
    "marseille": "marseille", "olympique marseille": "marseille",
    "monaco": "monaco", "lille": "lille", "nice": "nice",
    "rennes": "rennes", "lens": "lens", "nantes": "nantes",
    "strasbourg": "strasbourg", "toulouse": "toulouse", "brest": "brest",
    "reims": "reims", "montpellier": "montpellier", "auxerre": "auxerre",
    # Olanda
    "ajax": "ajax", "psv": "psv", "psv eindhoven": "psv",
    "feyenoord": "feyenoord", "az": "az", "az alkmaar": "az",
    "twente": "twente", "utrecht": "utrecht", "vitesse": "vitesse",
    "parma calcio 1913": "parma",
    "ac milan": "milan",
    "rasenballsport leipzig": "leipzig",
    "rb leipzig": "leipzig",
    "borussia m.gladbach": "gladbach",
    "borussia monchengladbach": "gladbach",
    "mainz 05": "mainz",
    "fc cologne": "koln",
    "1 fc koln": "koln",
    "nott'm forest": "nottingham forest",
    "forest": "nottingham forest",
    "hamburger sv": "hamburg",
    "elversberg": "elversberg",

}


# ======================= UTILITY NOMI / QUOTE =======================
def _norm_sq(nome: str) -> str:
    n = unicodedata.normalize("NFKD", str(nome or "")).encode("ascii", "ignore").decode().lower()
    n = re.sub(r"[^a-z0-9 ]", " ", n)
    n = " ".join(t for t in n.split() if t not in _STOP_NOMI and not t.isdigit())
    return _ALIAS_NOMI.get(n, n)


def _similarita(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    ta, tb = set(a.split()), set(b.split())
    if ta <= tb or tb <= ta:
        return 0.92
    return difflib.SequenceMatcher(None, a, b).ratio()


def mappa_nomi(nomi_api, nomi_target):
    """Associa nomi football-data.org → football-data.co.uk."""
    norm_t = {t: _norm_sq(t) for t in nomi_target}
    out = {}
    for n in nomi_api:
        na = _norm_sq(n)
        best, best_s = None, 0.0
        for t, nt in norm_t.items():
            s_ = _similarita(na, nt)
            if s_ > best_s:
                best, best_s = t, s_
        out[n] = best if best is not None and best_s >= 0.75 else n
    return out


def prima_tripla(riga, colonne):
    for cols in colonne:
        try:
            v = [float(riga[c]) for c in cols]
        except (KeyError, TypeError, ValueError):
            continue
        if all(np.isfinite(v)) and all(x > 1.0 for x in v):
            return v
    return None


def prima_coppia(riga, colonne):
    for a, b in colonne:
        try:
            v = [float(riga[a]), float(riga[b])]
        except (KeyError, TypeError, ValueError):
            continue
        if all(np.isfinite(v)) and all(x > 1.0 for x in v):
            return v
    return None


def senza_margine(quote):
    p = 1.0 / np.asarray(quote, dtype=float)
    return p / p.sum()


# ======================= CARICAMENTO DATI =======================
def _leggi_csv(url: str):
    try:
        res = requests.get(url, timeout=20, headers=UA)
    except Exception as e:
        return None, f"connessione fallita ({type(e).__name__})"
    if res.status_code != 200:
        return None, f"risposta HTTP {res.status_code}"
    for enc in ("utf-8-sig", "latin-1"):
        try:
            d = pd.read_csv(io.BytesIO(res.content), encoding=enc, on_bad_lines="skip")
            d.columns = [str(c).strip().lstrip("\ufeff") for c in d.columns]
            return d, None
        except UnicodeDecodeError:
            continue
        except Exception as e:
            return None, f"file non leggibile ({type(e).__name__})"
    return None, "codifica non riconosciuta"


def _pulisci_nome_openfootball(nome: str) -> str:
    """Toglie nazione tra parentesi e suffissi FC/CF comuni."""
    n = re.sub(r"\s*\([A-Z]{3}\)\s*", " ", nome or "").strip()
    n = re.sub(r"\s+", " ", n)
    return n


def _parse_openfootball_txt(text: str, stag: int) -> pd.DataFrame:
    """
    Parser del formato Football.TXT (openfootball):
      21:00  Team A (ITA)  v Team B (ENG)  1-3 (0-1)
    """
    rows = []
    # data corrente mentre scorre il file
    data_cur = None
    # linee tipo: Tue Sep 17 2024  oppure  Wed Oct 2
    re_data = re.compile(
        r"^(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+"
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+"
        r"(\d{1,2})(?:\s+(\d{4}))?",
        re.I,
    )
    mesi = {m: i for i, m in enumerate(
        ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], 1)}
    # match: opzionale orario, home v away, score
    re_match = re.compile(
        r"(?:\d{1,2}:\d{2}\s+)?"
        r"(.+?)\s+v\s+(.+?)\s+"
        r"(\d+)\s*[-–]\s*(\d+)"
        r"(?:\s*\([^)]*\))?",
        re.I,
    )
    anno = stag
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("=") or line.startswith("▪"):
            continue
        md = re_data.match(line)
        if md:
            mese, giorno, y = md.group(1), int(md.group(2)), md.group(3)
            if y:
                anno = int(y)
            try:
                data_cur = date(anno, mesi[mese[:1].upper() + mese[1:3].lower()], giorno)
                # fix month key: Jan -> Jan
                mi = mesi.get(mese.capitalize()) or mesi.get(mese.title())
                if mi is None:
                    for k, v in mesi.items():
                        if k.lower() == mese.lower():
                            mi = v
                            break
                if mi:
                    data_cur = date(anno, mi, giorno)
            except Exception:
                pass
            continue
        mm = re_match.search(line)
        if not mm or data_cur is None:
            continue
        home = _pulisci_nome_openfootball(mm.group(1))
        away = _pulisci_nome_openfootball(mm.group(2))
        try:
            gh, ga = int(mm.group(3)), int(mm.group(4))
        except ValueError:
            continue
        if not home or not away:
            continue
        rows.append({
            "Date": pd.Timestamp(data_cur),
            "HomeTeam": home,
            "AwayTeam": away,
            "FTHG": gh,
            "FTAG": ga,
            "Stag": stag,
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


@st.cache_data(ttl=12 * 3600, show_spinner="Scarico storico europeo (openfootball, gratis)...")
def carica_storico_europeo(div: str, n_stagioni: int, giorno: str, chiave: str):
    """
    Champions / Europa League — GRATIS via openfootball (GitHub).
    Fallback opzionale: football-data.org se hai la chiave.
    Non serve piano a pagamento.
    """
    oggi = date.today()
    anno = oggi.year if oggi.month >= 7 else oggi.year - 1
    file_name = "cl.txt" if div == "CL" else "el.txt"
    frames = []

    # 1) openfootball (nessuna chiave)
    for k in range(max(n_stagioni, 2) + 1):
        y = anno - k
        # stagioni tipo 2024-25
        slug = f"{y}-{str(y+1)[2:]}"
        url = f"https://raw.githubusercontent.com/openfootball/champions-league/master/{slug}/{file_name}"
        try:
            res = requests.get(url, timeout=25, headers=UA)
            if res.status_code != 200 or len(res.text) < 200:
                continue
            d = _parse_openfootball_txt(res.text, y)
            if d is not None and len(d) >= 10:
                frames.append(d)
        except Exception:
            continue

    # 2) fallback football-data.org (se chiave e openfootball insufficiente)
    if len(frames) < 1 and chiave:
        codice = API_CODICI.get(div, div)
        for k in range(n_stagioni + 1):
            y = anno - k
            try:
                res = requests.get(
                    f"https://api.football-data.org/v4/competitions/{codice}/matches",
                    headers={"X-Auth-Token": chiave},
                    params={"season": y, "status": "FINISHED"},
                    timeout=25,
                )
                if res.status_code != 200:
                    continue
                matches = (res.json() or {}).get("matches") or []
                rows = []
                for m in matches:
                    h = (m.get("homeTeam") or {}).get("name")
                    a = (m.get("awayTeam") or {}).get("name")
                    ft = ((m.get("score") or {}).get("fullTime") or {})
                    gh, ga = ft.get("home"), ft.get("away")
                    if not h or not a or gh is None or ga is None:
                        continue
                    rows.append({
                        "Date": m.get("utcDate"),
                        "HomeTeam": h,
                        "AwayTeam": a,
                        "FTHG": int(gh),
                        "FTAG": int(ga),
                        "Stag": y,
                    })
                if rows:
                    frames.append(pd.DataFrame(rows))
            except Exception:
                continue

    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    # dedup
    df = df.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"], keep="last")
    return df.sort_values("Date").reset_index(drop=True)


@st.cache_data(ttl=6 * 3600)
def carica_storico(div: str, n_stagioni: int, giorno: str, chiave: str = ""):
    """Storico nazionale (CSV football-data.co.uk) o europeo (API football-data.org)."""
    if div in EURO_DIVS:
        return carica_storico_europeo(div, n_stagioni, giorno, chiave or "")

    oggi = date.today()
    anno = oggi.year if oggi.month >= 7 else oggi.year - 1
    frames = []
    for k in range(n_stagioni):
        y = anno - k
        codice = f"{y % 100:02d}{(y + 1) % 100:02d}"
        d, _ = _leggi_csv(f"{BASE_URL}/mmz4281/{codice}/{div}.csv")
        if d is None or not {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}.issubset(d.columns):
            continue
        d["Stag"] = y
        frames.append(d)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce", format="mixed")
    df = df.dropna(subset=["Date"])
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    return df.sort_values("Date").reset_index(drop=True)


@st.cache_data(ttl=3600)
def carica_fixtures(div: str):
    d, err = _leggi_csv(f"{BASE_URL}/fixtures.csv")
    if d is None:
        return None, err
    if not {"Div", "HomeTeam", "AwayTeam"}.issubset(d.columns):
        return None, "colonne attese non trovate nel file"
    d = d[d["Div"] == div].dropna(subset=["HomeTeam", "AwayTeam"]).copy()
    if "Date" in d.columns:
        d["Date"] = pd.to_datetime(d["Date"], dayfirst=True, errors="coerce", format="mixed")
    return d.reset_index(drop=True), None




# ======================= CLUBELO (gratis, no key) =======================
@st.cache_data(ttl=12 * 3600, show_spinner="Carico rating ClubElo...")
def carica_clubelo(giorno: str):
    """
    Prima prova CSV offline (data/clubelo.csv), poi API ClubElo.
    """
    # 1) Offline
    for rel in ("data/clubelo.csv", "clubelo.csv"):
        for fp in (Path(rel), Path.cwd() / rel):
            try:
                if not fp.is_file() or fp.stat().st_size < 20:
                    continue
                df = pd.read_csv(fp)
                cols = {c.lower().strip(): c for c in df.columns}
                club_c = cols.get("club") or cols.get("team") or cols.get("squadra")
                elo_c = cols.get("elo") or cols.get("rating")
                if not club_c or not elo_c:
                    continue
                df = df.dropna(subset=[club_c]).copy()
                df["Elo"] = pd.to_numeric(df[elo_c], errors="coerce")
                df = df.dropna(subset=["Elo"])
                df["Club"] = df[club_c].astype(str)
                df["norm"] = df["Club"].map(_norm_sq)
                if len(df) >= 10:
                    return df[["Club", "Elo", "norm"]].reset_index(drop=True)
            except Exception:
                continue

    # 2) API (se offline non c'è)
    base = date.fromisoformat(giorno) if isinstance(giorno, str) else date.today()
    bases = [CLUBELO_URL, CLUBELO_URL.replace("http://", "https://")]
    for delta in range(0, 7):
        d = (base - pd.Timedelta(days=delta)).strftime("%Y-%m-%d")
        for base_url in bases:
            try:
                res = requests.get(f"{base_url}/{d}", timeout=35, headers=UA)
                if res.status_code != 200 or not res.text or "Club" not in res.text[:300]:
                    continue
                df = pd.read_csv(io.StringIO(res.text))
                if "Club" not in df.columns or "Elo" not in df.columns:
                    continue
                df = df.dropna(subset=["Club", "Elo"]).copy()
                df["Elo"] = pd.to_numeric(df["Elo"], errors="coerce")
                df = df.dropna(subset=["Elo"])
                df["norm"] = df["Club"].map(_norm_sq)
                return df.reset_index(drop=True)
            except Exception:
                continue
    return pd.DataFrame()


def mappa_elo(squadre, df_elo):
    """Associa ogni squadra del modello al rating Elo più simile (fuzzy)."""
    if df_elo is None or df_elo.empty:
        return {}
    out = {}
    norms = list(zip(df_elo["norm"].tolist(), df_elo["Elo"].tolist(), df_elo["Club"].tolist()))
    for sq in squadre:
        ns = _norm_sq(sq)
        best, best_s, best_e = None, 0.0, None
        for n, e, raw in norms:
            s = _similarita(ns, n)
            if s > best_s:
                best, best_s, best_e = raw, s, float(e)
        if best is not None and best_s >= 0.65:
            out[sq] = {"elo": best_e, "club_elo_name": best, "sim": round(best_s, 2)}
    return out


def fattore_elo(elo_casa, elo_trasf, scale=ELO_SCALE):
    """
    Fattori moltiplicativi su λ casa / λ trasferta basati sulla differenza Elo.
    delta = +100 → casa segna un po' di più, trasferta un po' di meno.
    """
    if elo_casa is None or elo_trasf is None:
        return 1.0, 1.0
    # vantaggio tipico ~ home già nel modello; qui solo forza relativa
    delta = (elo_casa - elo_trasf) / scale
    # soft: 10^(delta/2) su casa, 10^(-delta/2) su trasferta in scala gol
    f_h = 10 ** (0.5 * delta * 0.25)  # molto soft
    f_a = 10 ** (-0.5 * delta * 0.25)
    return float(f_h), float(f_a)


def lambda_con_elo(m, casa, trasf, elo_map, blend=ELO_BLEND):
    """λ Dixon-Coles eventualmente corrette con ClubElo (blend verso Elo)."""
    lh, la = lambda_partita(m, casa, trasf)
    if not elo_map or blend <= 0:
        return lh, la, None, None
    ec = (elo_map.get(casa) or {}).get("elo")
    et = (elo_map.get(trasf) or {}).get("elo")
    if ec is None and et is None:
        return lh, la, None, None
    # se manca un Elo, usa mediana lega ~1500
    med = 1500.0
    fh, fa = fattore_elo(ec if ec is not None else med, et if et is not None else med)
    # blend: λ' = λ^(1-b) * (λ * f)^b  ≈ λ * f^b
    lh2 = lh * (fh ** blend)
    la2 = la * (fa ** blend)
    return lh2, la2, ec, et


def forma_recente(df, n=FORM_N):
    """
    Media gol fatti/subiti nelle ultime n partite (casa+trasferta mescolate).
    Ritorna dict squadra -> {gf, gs, n, form_score}.
    form_score ≈ (gf - gs) medio, shrinkage verso 0.
    """
    if df is None or df.empty:
        return {}
    d = df.sort_values("Date")
    out = {}
    for sq in set(d["HomeTeam"]) | set(d["AwayTeam"]):
        mask_h = d["HomeTeam"] == sq
        mask_a = d["AwayTeam"] == sq
        rows = []
        for _, r in d[mask_h | mask_a].iterrows():
            if r["HomeTeam"] == sq:
                rows.append((r["Date"], int(r["FTHG"]), int(r["FTAG"])))
            else:
                rows.append((r["Date"], int(r["FTAG"]), int(r["FTHG"])))
        rows.sort(key=lambda x: x[0])
        last = rows[-n:]
        if not last:
            continue
        gf = np.mean([x[1] for x in last])
        gs = np.mean([x[2] for x in last])
        k = 2.0  # shrinkage
        gf_s = (gf * len(last) + k * 1.2) / (len(last) + k)
        gs_s = (gs * len(last) + k * 1.2) / (len(last) + k)
        out[sq] = {
            "gf": round(float(gf_s), 3),
            "gs": round(float(gs_s), 3),
            "n": len(last),
            "form_score": round(float(gf_s - gs_s), 3),
        }
    return out


def fattore_forma(form_map, casa, trasf):
    """Fattori soft su λ da forma recente (gol fatti / subiti)."""
    if not form_map:
        return 1.0, 1.0, None, None
    fc, ft = form_map.get(casa), form_map.get(trasf)
    # baseline lega ~1.2 gol
    base = 1.2
    # casa: fatti propri * subiti avversario
    if fc and ft:
        exp_h = (fc["gf"] / base) * (ft["gs"] / base)
        exp_a = (ft["gf"] / base) * (fc["gs"] / base)
        # clip per non esplodere
        exp_h = float(np.clip(exp_h, 0.6, 1.6))
        exp_a = float(np.clip(exp_a, 0.6, 1.6))
        return exp_h, exp_a, fc, ft
    return 1.0, 1.0, fc, ft


def lambda_full(m, casa, trasf, elo_map, form_map, elo_blend=ELO_BLEND, form_blend=FORM_BLEND):
    """
    λ finali = Dixon-Coles × (fattore Elo)^elo_blend × (fattore forma)^form_blend
    """
    lh, la = lambda_partita(m, casa, trasf)
    elo_c = elo_t = None
    if elo_map and elo_blend > 0:
        lh, la, elo_c, elo_t = lambda_con_elo(m, casa, trasf, elo_map, elo_blend)
    ff_h, ff_a, fc, ft = fattore_forma(form_map, casa, trasf)
    if form_blend > 0 and (fc or ft):
        lh = lh * (ff_h ** form_blend)
        la = la * (ff_a ** form_blend)
    return lh, la, elo_c, elo_t, fc, ft


@st.cache_data(ttl=6 * 3600)
def carica_understat_csv(giorno: str):
    """
    Carica xG medi da CSV offline (prodotto da GitHub Actions / script locale).
    Colonne attese: team, xg_for, xg_against, npx_g_for, matches, league (opzionale).
    """
    candidates = []
    for rel in UNDERSTAT_PATHS:
        candidates.append(Path(rel))
        candidates.append(Path.cwd() / rel)
    try:
        candidates.append(Path(__file__).resolve().parent / "data" / "understat_xg.csv")
    except Exception:
        pass
    for fp in candidates:
        try:
            if not fp.is_file() or fp.stat().st_size < 20:
                continue
            df = pd.read_csv(fp)
        except Exception:
            continue
        cols = {c.lower().strip(): c for c in df.columns}
        team_c = cols.get("team") or cols.get("squadra") or cols.get("name")
        xgf_c = cols.get("xg_for") or cols.get("xg") or cols.get("xgf")
        xga_c = cols.get("xg_against") or cols.get("xga")
        if not team_c or not xgf_c:
            continue
        df = df.dropna(subset=[team_c]).copy()
        if df.empty:
            continue
        df["_norm"] = df[team_c].map(_norm_sq)
        df["_xgf"] = pd.to_numeric(df[xgf_c], errors="coerce")
        df["_xga"] = pd.to_numeric(df[xga_c], errors="coerce") if xga_c else np.nan
        out = df.dropna(subset=["_xgf"]).reset_index(drop=True)
        if not out.empty:
            return out
    return pd.DataFrame()


def mappa_understat(squadre, df_u):
    if df_u is None or df_u.empty:
        return {}
    norms = list(zip(df_u["_norm"].tolist(), df_u["_xgf"].tolist(),
                     df_u["_xga"].tolist() if "_xga" in df_u.columns else [np.nan] * len(df_u),
                     df_u.iloc[:, 0].tolist()))
    out = {}
    for sq in squadre:
        ns = _norm_sq(sq)
        best_s, best = 0.0, None
        for n, xgf, xga, raw in norms:
            s = _similarita(ns, n)
            if s > best_s:
                best_s = s
                best = {"xg_for": float(xgf), "xg_against": float(xga) if np.isfinite(xga) else None,
                        "name": str(raw), "sim": round(s, 2)}
        if best and best_s >= 0.78:
            out[sq] = best
    return out


def fattore_xg(xg_map, casa, trasf):
    if not xg_map:
        return 1.0, 1.0
    hc, ht = xg_map.get(casa), xg_map.get(trasf)
    base = 1.3
    if hc and ht and hc.get("xg_for") and ht.get("xg_against"):
        exp_h = (hc["xg_for"] / base) * ((ht.get("xg_against") or base) / base)
        exp_a = (ht["xg_for"] / base) * ((hc.get("xg_against") or base) / base)
        return float(np.clip(exp_h, 0.7, 1.5)), float(np.clip(exp_a, 0.7, 1.5))
    return 1.0, 1.0


def log_matching(squadre, elo_map, xg_map, quote_keys, api_odds):
    """Tabella debug: copertura Elo / xG / quote per squadra."""
    rows = []
    q_teams = set()
    for a, b in (quote_keys or []):
        q_teams.add(a)
        q_teams.add(b)
    api_teams = set()
    for (ah, aa), v in (api_odds or {}).items():
        api_teams.add(v[2] if len(v) > 2 else ah)
        api_teams.add(v[3] if len(v) > 3 else aa)
    for sq in sorted(squadre):
        rows.append({
            "Squadra": sq,
            "Elo": round(elo_map[sq]["elo"], 1) if sq in elo_map else "—",
            "Elo match": elo_map[sq]["sim"] if sq in elo_map else "—",
            "xG for": round(xg_map[sq]["xg_for"], 2) if sq in xg_map else "—",
            "In fixtures/quote": "✓" if sq in q_teams else "",
            "In API odds": "✓" if any(_norm_sq(sq) == _norm_sq(x) for x in api_teams) else "",
        })
    return pd.DataFrame(rows)



def _stagione_api_football() -> int:
    """Stagione calcistica corrente (es. ott 2026 → 2026)."""
    oggi = date.today()
    return oggi.year if oggi.month >= 7 else oggi.year - 1


@st.cache_data(ttl=6 * 3600, show_spinner="Scarico quote da API-Football (cache 6h)...")
def scarica_odds_api_football(div: str, api_key: str, giorno: str):
    """
    Quote 1X2 e O/U 2.5 per il prossimo turno.
    Strategia a basso consumo (free = 100 req/giorno):
      1) /fixtures?league&season&next=12  → 1 req
      2) /odds?fixture=id per al massimo 8 partite  → fino a 8 req
    Totale ≤ 9 req per lega, cache 6 ore. 8 leghe ≈ 72 req/giorno nel worst case.
    Ritorna dict {(home_norm, away_norm): (p1x2, p_over25|None, home, away)}.
    """
    if not api_key or div not in API_FOOTBALL_LEAGUE_ID:
        return {}
    league_id = API_FOOTBALL_LEAGUE_ID[div]
    season = _stagione_api_football()
    headers = {"x-apisports-key": api_key}
    out = {}
    try:
        r = requests.get(
            f"{API_FOOTBALL_BASE}/fixtures",
            headers=headers,
            params={"league": league_id, "season": season, "next": 12},
            timeout=20,
        )
        if r.status_code != 200:
            return {}
        fixtures = (r.json() or {}).get("response") or []
        for fx in fixtures[:8]:
            fid = (fx.get("fixture") or {}).get("id")
            teams = fx.get("teams") or {}
            home = (teams.get("home") or {}).get("name") or ""
            away = (teams.get("away") or {}).get("name") or ""
            if not fid or not home or not away:
                continue
            try:
                ro = requests.get(
                    f"{API_FOOTBALL_BASE}/odds",
                    headers=headers,
                    params={"fixture": fid},
                    timeout=15,
                )
            except Exception:
                continue
            if ro.status_code != 200:
                continue
            payload = (ro.json() or {}).get("response") or []
            if not payload:
                continue
            vals_1, vals_x, vals_2 = [], [], []
            vals_over, vals_under = [], []
            for item in payload:
                for b in (item.get("bookmakers") or []):
                    for bet in (b.get("bets") or []):
                        name = (bet.get("name") or "").lower()
                        for v in (bet.get("values") or []):
                            lab = (v.get("value") or "").strip().lower()
                            try:
                                odd = float(v.get("odd"))
                            except (TypeError, ValueError):
                                continue
                            if odd <= 1.0:
                                continue
                            if name in ("match winner", "home/away", "1x2"):
                                if lab in ("home", "1"):
                                    vals_1.append(odd)
                                elif lab in ("draw", "x"):
                                    vals_x.append(odd)
                                elif lab in ("away", "2"):
                                    vals_2.append(odd)
                            elif "over/under" in name:
                                lab2 = lab.replace(" ", "")
                                if lab2 in ("over2.5",):
                                    vals_over.append(odd)
                                elif lab2 in ("under2.5",):
                                    vals_under.append(odd)
            if vals_1 and vals_x and vals_2:
                q = [float(np.mean(vals_1)), float(np.mean(vals_x)), float(np.mean(vals_2))]
                p1x2 = senza_margine(q)
                p_ou = None
                if vals_over and vals_under:
                    p_ou = float(senza_margine([float(np.mean(vals_over)), float(np.mean(vals_under))])[0])
                out[(_norm_sq(home), _norm_sq(away))] = (p1x2, p_ou, home, away)
    except Exception:
        pass
    return out


def _match_quote_api(api_odds: dict, casa: str, trasf: str):
    """Trova quote API per una partita confrontando nomi normalizzati."""
    if not api_odds:
        return None
    nc, nt = _norm_sq(casa), _norm_sq(trasf)
    # match esatto
    if (nc, nt) in api_odds:
        p1x2, p_ou, _, _ = api_odds[(nc, nt)]
        return p1x2, p_ou
    # fuzzy
    best, best_s = None, 0.0
    for (ah, aa), (p1x2, p_ou, _, _) in api_odds.items():
        s = 0.5 * (_similarita(nc, ah) + _similarita(nt, aa))
        if s > best_s:
            best, best_s = (p1x2, p_ou), s
    return best if best is not None and best_s >= 0.78 else None


@st.cache_data(ttl=3600)
def scarica_calendario_api(chiave: str, codice: str):
    if not chiave:
        return None
    try:
        res = requests.get(
            f"https://api.football-data.org/v4/competitions/{codice}/matches",
            headers={"X-Auth-Token": chiave},
            timeout=15,
        )
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return None


def assegna_giornate(righe):
    conta = {}
    for r in righe:
        g = max(conta.get(r["Casa"], 0), conta.get(r["Trasferta"], 0)) + 1
        r["Giornata"] = g
        conta[r["Casa"]] = conta.get(r["Casa"], 0) + 1
        conta[r["Trasferta"]] = conta.get(r["Trasferta"], 0) + 1
    return righe


def costruisci_calendario(div, df_storico, fx, chiave, squadre_modello):
    dati = scarica_calendario_api(chiave, API_CODICI.get(div, "")) if chiave else None
    if dati and dati.get("matches"):
        nomi = {m["homeTeam"]["name"] for m in dati["matches"]} | {m["awayTeam"]["name"] for m in dati["matches"]}
        nomi = {n for n in nomi if n}
        mp = mappa_nomi(nomi, squadre_modello)
        righe = []
        for m in dati["matches"]:
            h, a = m["homeTeam"].get("name"), m["awayTeam"].get("name")
            if not h or not a:
                continue
            ft = (m.get("score") or {}).get("fullTime") or {}
            ts = pd.to_datetime(m["utcDate"], utc=True).tz_convert("Europe/Rome")
            righe.append({
                "Giornata": m.get("matchday") or 0,
                "Data": ts.strftime("%Y-%m-%d"),
                "Ora": ts.strftime("%H:%M"),
                "Casa": mp.get(h, h),
                "Trasferta": mp.get(a, a),
                "GolC": ft.get("home"),
                "GolT": ft.get("away"),
                "Stato": m.get("status", ""),
            })
        return pd.DataFrame(righe), "football-data.org (calendario completo)"

    righe = []
    stag = df_storico[df_storico["Stag"] == df_storico["Stag"].max()]
    for _, r in stag.iterrows():
        righe.append({
            "Data": r["Date"].strftime("%Y-%m-%d"),
            "Ora": str(r["Time"])[:5] if "Time" in r and pd.notna(r.get("Time")) else "",
            "Casa": r["HomeTeam"],
            "Trasferta": r["AwayTeam"],
            "GolC": int(r["FTHG"]),
            "GolT": int(r["FTAG"]),
            "Stato": "FINISHED",
            "_d": r["Date"],
        })
    giocate = {(r["Data"], r["Casa"], r["Trasferta"]) for r in righe}
    if fx is not None:
        for _, r in fx.iterrows():
            d = r["Date"] if "Date" in r and pd.notna(r["Date"]) else pd.Timestamp.max
            key = (d.strftime("%Y-%m-%d") if d != pd.Timestamp.max else "", r["HomeTeam"], r["AwayTeam"])
            if key in giocate:
                continue
            righe.append({
                "Data": d.strftime("%Y-%m-%d") if d != pd.Timestamp.max else "",
                "Ora": str(r["Time"])[:5] if "Time" in r and pd.notna(r.get("Time")) else "",
                "Casa": r["HomeTeam"],
                "Trasferta": r["AwayTeam"],
                "GolC": None,
                "GolT": None,
                "Stato": "SCHEDULED",
                "_d": d,
            })
    righe.sort(key=lambda r: r["_d"])
    righe = assegna_giornate(righe)
    return pd.DataFrame(righe).drop(columns="_d"), "football-data.co.uk (giocate + prossimo turno)"


# ======================= MODELLO DIXON-COLES =======================
def adatta_modello(df, data_rif, ridge: float, halflife: float, usa_dc: bool = True):
    squadre = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
    n = len(squadre)
    idx = {t: i for i, t in enumerate(squadre)}
    hi = df["HomeTeam"].map(idx).to_numpy()
    ai = df["AwayTeam"].map(idx).to_numpy()
    x = df["FTHG"].to_numpy(float)
    y = df["FTAG"].to_numpy(float)
    eta_gg = (data_rif - df["Date"]).dt.days.clip(lower=0).to_numpy(float)
    w = np.exp(-np.log(2) * eta_gg / halflife) if halflife and halflife > 0 else np.ones(len(df))

    def f(th):
        mu, hm = th[0], th[1]
        att, dfn = th[2:2 + n], th[2 + n:]
        eh = mu + hm + att[hi] + dfn[ai]
        ea = mu + att[ai] + dfn[hi]
        lh, la = np.exp(eh), np.exp(ea)
        nll = -np.sum(w * (x * eh - lh + y * ea - la)) + ridge * (att @ att + dfn @ dfn)
        rh, ra = w * (x - lh), w * (y - la)
        g = np.zeros_like(th)
        g[0] = -(rh.sum() + ra.sum())
        g[1] = -rh.sum()
        g[2:2 + n] = -(np.bincount(hi, rh, n) + np.bincount(ai, ra, n)) + 2 * ridge * att
        g[2 + n:] = -(np.bincount(ai, rh, n) + np.bincount(hi, ra, n)) + 2 * ridge * dfn
        return nll, g

    th0 = np.zeros(2 + 2 * n)
    th0[0] = np.log(max((x.mean() + y.mean()) / 2, 0.3))
    th = minimize(f, th0, jac=True, method="L-BFGS-B").x
    mu, hm, att, dfn = th[0], th[1], th[2:2 + n], th[2 + n:]

    rho = 0.0
    if usa_dc:
        lh = np.exp(mu + hm + att[hi] + dfn[ai])
        la = np.exp(mu + att[ai] + dfn[hi])
        m00 = (x == 0) & (y == 0)
        m01 = (x == 0) & (y == 1)
        m10 = (x == 1) & (y == 0)
        m11 = (x == 1) & (y == 1)

        def neg(r):
            t = np.ones(len(x))
            t[m00] = 1 - lh[m00] * la[m00] * r
            t[m01] = 1 + lh[m01] * r
            t[m10] = 1 + la[m10] * r
            t[m11] = 1 - r
            return -np.sum(w * np.log(np.maximum(t, 1e-6)))

        rho = float(minimize_scalar(neg, bounds=(-0.3, 0.3), method="bounded").x)

    return {"squadre": squadre, "idx": idx, "mu": float(mu), "hm": float(hm), "att": att, "dfn": dfn, "rho": rho}


def lambda_partita(m, casa, trasf):
    ic, it = m["idx"].get(casa), m["idx"].get(trasf)
    ac, dc = (m["att"][ic], m["dfn"][ic]) if ic is not None else (0.0, 0.0)
    at, dt = (m["att"][it], m["dfn"][it]) if it is not None else (0.0, 0.0)
    return math.exp(m["mu"] + m["hm"] + ac + dt), math.exp(m["mu"] + at + dc)


_K = np.arange(11)
_LOGFACT = gammaln(_K + 1)


def _pmf(lam):
    return np.exp(-lam + _K * np.log(np.maximum(lam, 1e-12)) - _LOGFACT)


def matrice_punteggi(lh, la, rho, mg=10):
    M = np.outer(_pmf(lh), _pmf(la))
    if rho != 0:
        M[0, 0] *= max(1 - lh * la * rho, 1e-6)
        M[0, 1] *= max(1 + lh * rho, 1e-6)
        M[1, 0] *= max(1 + la * rho, 1e-6)
        M[1, 1] *= max(1 - rho, 1e-6)
    s = M.sum()
    return M / s if s > 0 else M


def mercati(M):
    tot = np.add.outer(np.arange(M.shape[0]), np.arange(M.shape[1]))
    d = {
        "p1": float(np.tril(M, -1).sum()),
        "px": float(np.trace(M)),
        "p2": float(np.triu(M, 1).sum()),
        "btts": float(M[1:, 1:].sum()),
    }
    for s in (1.5, 2.5, 3.5):
        d[f"over{s}"] = float(M[tot > s].sum())
    return d


def sintesi_pronostico(M):
    """
    Pronostico 1X2, doppia chance, combo (congiunte sulla matrice) e multigol.
    La X pura perde quasi sempre l'argmax (pareggio ~25% vs favorito ~45%).
    Se i due lati sono vicini e il pareggio è competitivo, il segno diventa X.
    """
    ii, jj = np.indices(M.shape)
    totg = ii + jj
    p1 = float(M[ii > jj].sum())
    px = float(M[ii == jj].sum())
    p2 = float(M[ii < jj].sum())
    btts = (ii > 0) & (jj > 0)
    home, away, draw = ii > jj, ii < jj, ii == jj

    secco = max((("1", p1), ("X", px), ("2", p2)), key=lambda t: t[1])[0]
    equilibrata = abs(p1 - p2) <= 0.10 and px >= 0.22 and px >= max(p1, p2) - 0.10
    segno = "X" if equilibrata else secco

    dc = {"1X": p1 + px, "X2": px + p2, "12": p1 + p2}
    dc_nome = max(dc, key=dc.get)

    combo_core = {
        "1 + Gol": home & btts,
        "1 + No Gol": home & ~btts,
        "X + Gol": draw & btts,
        "X + No Gol": draw & ~btts,
        "2 + Gol": away & btts,
        "2 + No Gol": away & ~btts,
    }
    combo_extra = {
        "1 + Over 2.5": home & (totg >= 3),
        "2 + Over 2.5": away & (totg >= 3),
        "X + Under 2.5": draw & (totg <= 2),
        "1 + Over 1.5": home & (totg >= 2),
        "2 + Over 1.5": away & (totg >= 2),
    }
    core = sorted(
        ((nome, float(M[mask].sum())) for nome, mask in combo_core.items()),
        key=lambda t: t[1], reverse=True,
    )
    # Combo principale: solo quelle del segno scelto (se è X, non 1+No Gol / 2+No Gol)
    del_segno = [c for c in core if c[0].startswith(segno + " ")]
    scelta = del_segno[0] if del_segno else core[0]
    extra = sorted(
        ((nome, float(M[mask].sum())) for nome, mask in combo_extra.items()),
        key=lambda t: t[1], reverse=True,
    )
    # Over/Under solo del segno (1 + Over, 2 + Under, X + Under, …)
    ou_by = {}
    for segno_k, lato_k in (("1", home), ("X", draw), ("2", away)):
        coppie = [
            (f"{segno_k} + Over 2.5", lato_k & (totg >= 3)),
            (f"{segno_k} + Under 2.5", lato_k & (totg <= 2)),
            (f"{segno_k} + Over 1.5", lato_k & (totg >= 2)),
            (f"{segno_k} + Under 1.5", lato_k & (totg <= 1)),
        ]
        tutte = sorted(
            ((nome, float(M[mask].sum())) for nome, mask in coppie),
            key=lambda t: t[1], reverse=True,
        )
        ou_25 = [c for c in tutte if "2.5" in c[0]]
        ou_by[segno_k] = {"scelta": max(ou_25, key=lambda t: t[1]), "tutte": tutte}
    ou_scelta = ou_by[segno]["scelta"]
    ou_all = ou_by[segno]["tutte"]

    bande = (("1-2", 1, 2), ("2-3", 2, 3), ("0-2", 0, 2), ("1-3", 1, 3), ("2-4", 2, 4), ("3-5", 3, 5))
    mg = sorted(
        ((nome, float(M[(totg >= a) & (totg <= b)].sum())) for nome, a, b in bande),
        key=lambda t: t[1],
        reverse=True,
    )
    bande_sq = (("0-1", 0, 1), ("1-2", 1, 2), ("0-2", 0, 2), ("1-3", 1, 3), ("2-3", 2, 3))
    marg_h = M.sum(axis=1)
    marg_a = M.sum(axis=0)

    def _fasce(marg, bande_):
        out = []
        for nome, a, b in bande_:
            if a >= len(marg):
                out.append((nome, 0.0))
            else:
                out.append((nome, float(marg[a:min(b + 1, len(marg))].sum())))
        out.sort(key=lambda t: t[1], reverse=True)
        return out

    def _stretta(lista):
        """In evidenza 0-1 / 1-2 / 2-3: 0-2 e 1-3 coprono troppo e vincono sempre."""
        strette = [c for c in lista if c[0] in ("0-1", "1-2", "2-3")]
        return max(strette, key=lambda t: t[1]) if strette else lista[0]

    mg_casa = _fasce(marg_h, bande_sq)
    mg_osp = _fasce(marg_a, bande_sq)
    # 1°/2° tempo: split dei gol attesi (niente storico tempi). 1T ~45%, 2T ~55%.
    lh_hat = float((marg_h * np.arange(len(marg_h))).sum())
    la_hat = float((marg_a * np.arange(len(marg_a))).sum())
    bande_ht = (("0-1", 0, 1), ("1-2", 1, 2), ("0-2", 0, 2), ("1-3", 1, 3))

    def _mg_tempo(share):
        Mt = matrice_punteggi(max(lh_hat * share, 0.05), max(la_hat * share, 0.05), 0.0)
        ti, tj = np.indices(Mt.shape)
        tt = ti + tj
        p = np.zeros(int(tt.max()) + 1)
        for k in range(len(p)):
            p[k] = float(Mt[tt == k].sum())
        return _fasce(p, bande_ht)

    mg_1t = _mg_tempo(0.45)
    mg_2t = _mg_tempo(0.55)
    casa_s, osp_s = _stretta(mg_casa), _stretta(mg_osp)
    t1_s, t2_s = _stretta(mg_1t), _stretta(mg_2t)
    return {
        "p1": p1, "px": px, "p2": p2,
        "segno": segno,
        "segno_max": secco,
        "equilibrata": equilibrata,
        "dc": dc_nome, "dc_p": dc[dc_nome], "dc_all": dc,
        "combo": scelta[0], "combo_p": scelta[1], "combo_top": (del_segno or core)[:3],
        "combo_all": core,
        "combo_extra": extra[0],
        "ou": ou_scelta[0], "ou_p": ou_scelta[1], "ou_top": ou_all, "ou_by": ou_by,
        "mg": mg[0][0], "mg_p": mg[0][1], "mg_top": mg[:3],
        "mg_casa": casa_s[0], "mg_casa_p": casa_s[1], "mg_casa_top": mg_casa[:3],
        "mg_osp": osp_s[0], "mg_osp_p": osp_s[1], "mg_osp_top": mg_osp[:3],
        "mg_1t": t1_s[0], "mg_1t_p": t1_s[1], "mg_1t_top": mg_1t[:3],
        "mg_2t": t2_s[0], "mg_2t_p": t2_s[1], "mg_2t_top": mg_2t[:3],
    }


def matrice_frequenze(df):
    cm = np.zeros((11, 11))
    np.add.at(cm, (np.minimum(df["FTHG"].to_numpy(), 10), np.minimum(df["FTAG"].to_numpy(), 10)), 1)
    return (cm + 0.05) / (cm.sum() + 0.05 * 121)


def matrice_ancorata(p1x2, p_over25, rho, start):
    def err(th):
        mk = mercati(matrice_punteggi(math.exp(th[0]), math.exp(th[1]), rho))
        e = (mk["p1"] - p1x2[0]) ** 2 + (mk["px"] - p1x2[1]) ** 2 + (mk["p2"] - p1x2[2]) ** 2
        if p_over25 is not None:
            e += (mk["over2.5"] - p_over25) ** 2
        return e

    res = minimize(
        err, np.log(np.asarray(start, float)),
        method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-10, "maxiter": 250},
    )
    return matrice_punteggi(math.exp(res.x[0]), math.exp(res.x[1]), rho)


# ======================= CALIBRAZIONE PLATT =======================
def platt_fit(p_raw, y_bin):
    """Regressione logistica 1D: P_cal = sigmoid(a + b * logit(p))."""
    p = np.clip(np.asarray(p_raw, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y_bin, float)
    if len(y) < 30 or y.mean() in (0.0, 1.0):
        return 0.0, 1.0  # identità

    z = np.log(p / (1 - p))

    def nll(th):
        a, b = th
        q = expit(a + b * z)
        q = np.clip(q, 1e-9, 1 - 1e-9)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))

    res = minimize(nll, [0.0, 1.0], method="L-BFGS-B")
    return float(res.x[0]), float(res.x[1])


def platt_apply(p_raw, a, b):
    p = np.clip(np.asarray(p_raw, float), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p))
    return expit(a + b * z)


def _param_platt(p, y):
    """Platt stabile: se il fit è storto, lascia la probabilità invariata."""
    a, b = platt_fit(p, y)
    if not np.isfinite(a) or not np.isfinite(b) or b < 0.25:
        return 0.0, 1.0
    return float(a), float(b)


def _fasce_hit(p_cal, y, width=0.10):
    p_cal = np.asarray(p_cal, float)
    y = np.asarray(y, float)
    out = []
    lo = 0.15
    while lo < 0.90:
        hi = lo + width
        sel = (p_cal >= lo) & (p_cal < hi)
        n = int(sel.sum())
        if n >= 25:
            out.append({"lo": round(lo, 2), "hi": round(hi, 2), "hit": float(y[sel].mean()), "n": n})
        lo += width
    return out


def fit_calibrazione(bt):
    """Calibrazione 1X2, Over 2.5 e Goal sul walk-forward, più il tasso storico per fascia."""
    if bt is None or len(bt) < 80 or "over25" not in bt.columns:
        return None
    y = bt["y"].to_numpy(int)
    out = {"n": int(len(bt)), "fasce": {}}
    for col, yy in (("p1", y == 0), ("px", y == 1), ("p2", y == 2)):
        raw = bt[col].to_numpy(float)
        a, b = _param_platt(raw, yy.astype(float))
        out[col] = (a, b)
        out["fasce"][col] = _fasce_hit(np.asarray(platt_apply(raw, a, b), float), yy.astype(float))
    for col, ycol in (("over25", "y_o25"), ("btts", "y_btts")):
        raw = bt[col].to_numpy(float)
        yy = bt[ycol].to_numpy(float)
        a, b = _param_platt(raw, yy)
        out[col] = (a, b)
        out["fasce"][col] = _fasce_hit(np.asarray(platt_apply(raw, a, b), float), yy)
    return out


def storico_fascia(cal, chiave, p):
    if not cal:
        return None
    for f in cal.get("fasce", {}).get(chiave, []):
        if f["lo"] <= p < f["hi"] or (p >= 0.85 and f["hi"] >= 0.85 and p >= f["lo"]):
            return f
    return None


def _testo_storico(cal, chiave, p, ancorata=False):
    if ancorata and chiave in ("p1", "px", "p2", "over25"):
        return "allineato alle quote (fonte più affidabile del modello puro)"
    f = storico_fascia(cal, chiave, p)
    if not f:
        return "storico in questa fascia ancora corto"
    return f"quando il modello diceva {f['lo']*100:.0f}–{f['hi']*100:.0f}%, è uscito il {f['hit']*100:.0f}% (n={f['n']})"


def _segna_da_p(p1, px, p2):
    secco = max((("1", p1), ("X", px), ("2", p2)), key=lambda t: t[1])[0]
    equilibrata = abs(p1 - p2) <= 0.10 and px >= 0.22 and px >= max(p1, p2) - 0.10
    segno = "X" if equilibrata else secco
    dc = {"1X": p1 + px, "X2": px + p2, "12": p1 + p2}
    dc_nome = max(dc, key=dc.get)
    return segno, secco, equilibrata, dc_nome, dc[dc_nome], dc


def applica_affidabilita(syn, cal, ancorata=False):
    """
    Senza quote, 1X2 e doppia chance usano le probabilità calibrate.
    Combo e multigol restano sulla matrice: sono indicativi.
    Con le quote il 1X2 non si tocca.
    """
    syn = dict(syn)
    syn["calib"] = False
    if ancorata or not cal:
        return syn
    p1 = float(platt_apply(syn["p1"], *cal["p1"]))
    px = float(platt_apply(syn["px"], *cal["px"]))
    p2 = float(platt_apply(syn["p2"], *cal["p2"]))
    s = p1 + px + p2
    if s <= 0:
        return syn
    p1, px, p2 = p1 / s, px / s, p2 / s
    segno, secco, eq, dc_nome, dc_p, dc = _segna_da_p(p1, px, p2)
    syn.update(p1=p1, px=px, p2=p2, segno=segno, segno_max=secco, equilibrata=eq,
               dc=dc_nome, dc_p=dc_p, dc_all=dc, calib=True)
    del_segno = [c for c in syn.get("combo_all", []) if c[0].startswith(segno + " ")]
    if del_segno:
        syn["combo"], syn["combo_p"] = del_segno[0][0], del_segno[0][1]
        syn["combo_top"] = del_segno[:3]
    ou = (syn.get("ou_by") or {}).get(segno)
    if ou:
        syn["ou"], syn["ou_p"], syn["ou_top"] = ou["scelta"][0], ou["scelta"][1], ou["tutte"]
    return syn


def prob_calibrata(p, cal, chiave):
    if not cal or chiave not in cal:
        return float(p)
    return float(np.clip(platt_apply(p, *cal[chiave]), 0.01, 0.99))


# ======================= BACKTEST 1X2 =======================
def esegui_backtest_df(df, ridge, halflife, usa_dc, passo):
    stagioni = sorted(df["Stag"].unique())
    if len(stagioni) < 2:
        return pd.DataFrame()
    t = df.loc[df["Stag"] == stagioni[1], "Date"].min()
    fine = df["Date"].max()
    righe = []
    while t <= fine:
        blocco = df[(df["Date"] >= t) & (df["Date"] < t + pd.Timedelta(days=passo))]
        train = df[df["Date"] < t]
        if len(blocco) and len(train) >= 150:
            m = adatta_modello(train, t, ridge, halflife, usa_dc)
            esiti_tr = np.where(train["FTHG"] > train["FTAG"], 0, np.where(train["FTHG"] == train["FTAG"], 1, 2))
            freq = np.bincount(esiti_tr, minlength=3) / len(esiti_tr)
            for _, r in blocco.iterrows():
                lh, la = lambda_partita(m, r["HomeTeam"], r["AwayTeam"])
                mk = mercati(matrice_punteggi(lh, la, m["rho"]))
                esito = 0 if r["FTHG"] > r["FTAG"] else (1 if r["FTHG"] == r["FTAG"] else 2)
                qp, qc = prima_tripla(r, COL_PRESA), prima_tripla(r, COL_CHIUSURA)
                pm = senza_margine(qc) if qc else [np.nan] * 3
                righe.append({
                    "Date": r["Date"], "Partita": f"{r['HomeTeam']} - {r['AwayTeam']}", "y": esito,
                    "p1": mk["p1"], "px": mk["px"], "p2": mk["p2"],
                    "over25": mk["over2.5"], "btts": mk["btts"],
                    "y_o25": int(r["FTHG"] + r["FTAG"] >= 3),
                    "y_btts": int(r["FTHG"] > 0 and r["FTAG"] > 0),
                    "f1": freq[0], "fx": freq[1], "f2": freq[2],
                    "q1": qp[0] if qp else np.nan, "qx": qp[1] if qp else np.nan, "q2": qp[2] if qp else np.nan,
                    "m1": pm[0], "mx": pm[1], "m2": pm[2],
                })
        t += pd.Timedelta(days=passo)
    return pd.DataFrame(righe)


def _ll_brier(P, y):
    P = np.asarray(P, float)
    ll = -np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-9, 1)))
    br = np.mean(np.sum((P - np.eye(3)[y]) ** 2, axis=1))
    return ll, br


def metriche_backtest(bt):
    y = bt["y"].to_numpy(int)
    out = {"n": len(bt)}
    out["modello"] = _ll_brier(bt[["p1", "px", "p2"]].to_numpy(), y)
    out["frequenze"] = _ll_brier(bt[["f1", "fx", "f2"]].to_numpy(), y)
    ok = bt[["m1", "mx", "m2"]].notna().all(axis=1).to_numpy()
    if ok.sum() > 30:
        out["mercato"] = _ll_brier(bt.loc[ok, ["m1", "mx", "m2"]].to_numpy(), y[ok])
        out["modello_su_mercato"] = _ll_brier(bt.loc[ok, ["p1", "px", "p2"]].to_numpy(), y[ok])
        out["n_mercato"] = int(ok.sum())
    return out


def tabella_scommesse(bt, soglie=(0, 2, 5, 8, 10)):
    ok = bt[["q1", "qx", "q2"]].notna().all(axis=1)
    b = bt[ok]
    if b.empty:
        return pd.DataFrame()
    P = b[["p1", "px", "p2"]].to_numpy()
    Q = b[["q1", "qx", "q2"]].to_numpy()
    M = b[["m1", "mx", "m2"]].to_numpy()
    y = b["y"].to_numpy(int)
    edge = P * Q - 1
    k = edge.argmax(axis=1)
    e_best = edge[np.arange(len(b)), k]
    q_k = Q[np.arange(len(b)), k]
    profitto = np.where(y == k, q_k - 1, -1.0)
    clv = q_k * M[np.arange(len(b)), k] - 1
    righe = []
    for s in soglie:
        sel = e_best >= s / 100
        n = int(sel.sum())
        if n == 0:
            righe.append({"Edge min %": s, "Scommesse": 0, "Vinte %": None, "ROI %": None, "± ROI %": None, "CLV medio %": None})
            continue
        pr, c = profitto[sel], clv[sel]
        righe.append({
            "Edge min %": s, "Scommesse": n,
            "Vinte %": round((y[sel] == k[sel]).mean() * 100, 1),
            "ROI %": round(pr.mean() * 100, 1),
            "± ROI %": round(pr.std(ddof=1) / math.sqrt(n) * 100, 1) if n > 1 else None,
            "CLV medio %": round(float(np.nanmean(c)) * 100, 1) if np.isfinite(c).any() else None,
        })
    return pd.DataFrame(righe)


def tabella_calibrazione(bt, bins=10):
    P = bt[["p1", "px", "p2"]].to_numpy().ravel()
    O = np.eye(3)[bt["y"].to_numpy(int)].ravel()
    cut = np.minimum((P * bins).astype(int), bins - 1)
    righe = []
    for b in range(bins):
        sel = cut == b
        if sel.sum() >= 10:
            righe.append({
                "Prob. prevista %": round(P[sel].mean() * 100, 1),
                "Frequenza reale %": round(O[sel].mean() * 100, 1),
                "Casi": int(sel.sum()),
            })
    return pd.DataFrame(righe)


# ======================= PARAMETRI AUTOMATICI (veloce + hold-out) =======================
def _logloss_holdout_veloce(df_tune, df_hold, ridge, halflife):
    """Un solo fit sul tune, valutazione sull'hold-out (molto più veloce del walk-forward)."""
    if df_hold is None or len(df_hold) < 20 or df_tune is None or len(df_tune) < 100:
        return None
    t0 = df_hold["Date"].min()
    m = adatta_modello(df_tune, t0, ridge, halflife, True)
    if m is None:
        return None
    ps, ys = [], []
    for _, r in df_hold.iterrows():
        try:
            lh, la = lambda_partita(m, r["HomeTeam"], r["AwayTeam"])
            mk = mercati(matrice_punteggi(lh, la, m["rho"]))
            ps.append([mk["p1"], mk["px"], mk["p2"]])
            ys.append(0 if r["FTHG"] > r["FTAG"] else (1 if r["FTHG"] == r["FTAG"] else 2))
        except Exception:
            continue
    if len(ys) < 20:
        return None
    return _ll_brier(np.asarray(ps, float), np.asarray(ys, int))[0]


@st.cache_data(ttl=6 * 3600, show_spinner="Ottimizzo i parametri (griglia leggera)...")
def parametri_auto(div: str, giorno: str, chiave: str = ""):
    """
    Griglia RIDOTTA (3×3) + hold-out veloce.
    Prima versione faceva 25–50 walk-forward completi → su Streamlit Cloud sembrava bloccata.
    """
    df = carica_storico(div, N_STAGIONI, giorno, chiave)
    default = {"ridge": 2.0, "halflife": 365, "log_loss": None, "log_loss_holdout": None, "ricerca": [], "auto": False}
    if df is None or df["Stag"].nunique() < 2:
        return default

    stagioni = sorted(df["Stag"].unique())
    if len(stagioni) >= 3:
        df_tune = df[df["Stag"] < stagioni[-1]]
        df_hold = df[df["Stag"] == stagioni[-1]]
    else:
        df_tune = df
        df_hold = None

    # griglia leggera: 9 punti, passo più largo
    passo_veloce = max(PASSO_PARAM, 42)
    ricerca = []
    for r_ in (1.5, 2.0, 4.0):
        for h_ in (180, 365, 0):
            bt = esegui_backtest_df(df_tune, r_, h_, True, passo_veloce)
            if bt.empty or len(bt) < 40:
                continue
            ll = metriche_backtest(bt)["modello"][0]
            ll_h = _logloss_holdout_veloce(df_tune, df_hold, r_, h_)
            ricerca.append({
                "Shrinkage λ": r_, "Emivita (gg)": str(h_) if h_ else "nessuna",
                "Log loss (tune)": round(ll, 4),
                "Log loss (hold-out)": round(ll_h, 4) if ll_h is not None else "—",
                "_r": r_, "_h": h_, "_ll": ll, "_ll_h": ll_h if ll_h is not None else ll,
            })

    if not ricerca:
        return default

    best = min(ricerca, key=lambda z: z["_ll_h"])
    return {
        "ridge": best["_r"],
        "halflife": best["_h"],
        "log_loss": best["_ll"],
        "log_loss_holdout": best["_ll_h"] if best.get("_ll_h") != best["_ll"] else None,
        "auto": True,
        "ricerca": [{k: v for k, v in z.items() if not k.startswith("_")} for z in ricerca],
    }


@st.cache_data(ttl=6 * 3600)
def modello_corrente(div, ridge, halflife, giorno, chiave: str = ""):
    df = carica_storico(div, N_STAGIONI, giorno, chiave)
    if df is None or len(df) < 100:
        return None
    return adatta_modello(df, pd.Timestamp.now().normalize(), ridge, halflife, True)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def backtest_cached(div, ridge, halflife, passo, giorno, chiave: str = ""):
    df = carica_storico(div, N_STAGIONI, giorno, chiave)
    return None if df is None else esegui_backtest_df(df, ridge, halflife, True, passo)


@st.cache_data(ttl=6 * 3600, show_spinner="Calibro le percentuali sul passato...")
def calibrazione_live(div, ridge, halflife, giorno, chiave: str = ""):
    """Un walk-forward (passo 28 giorni) per riallineare le % a quello che è successo davvero."""
    df = carica_storico(div, N_STAGIONI, giorno, chiave)
    if df is None or len(df) < 150:
        return None
    bt = esegui_backtest_df(df, ridge, halflife, True, 28)
    return fit_calibrazione(bt)


# ======================= TEST COMPLETO =======================
_I, _J = np.meshgrid(np.arange(11), np.arange(11), indexing="ij")
_MASK = {
    "p1": _I > _J, "px": _I == _J, "p2": _I < _J,
    "over1.5": (_I + _J) > 1.5, "over2.5": (_I + _J) > 2.5,
    "over3.5": (_I + _J) > 3.5, "btts": (_I > 0) & (_J > 0),
}
VARIANTI_TIRI = ["Gol + Tiri 50/50", "Tiri in porta"]
VARIANTI_MODELLO = ["Modello", "Senza Dixon-Coles", "Ancorato al mercato", "Misto 50/50"] + VARIANTI_TIRI
VARIANTI = VARIANTI_MODELLO + ["Frequenze storiche"]


def esegui_test_completo(df, ridge, halflife, passo, solo_base=False):
    stagioni = sorted(df["Stag"].unique())
    if len(stagioni) < 2:
        return None
    t = df.loc[df["Stag"] == stagioni[1], "Date"].min()
    fine = df["Date"].max()
    nan_m = np.full((11, 11), np.nan)
    rec = {
        "Partita": [], "xs": [], "ys": [], "ok": [], "ou_ok": [],
        "m1x2": [], "mou": [], "min_storia": [], "tiri_ok": [],
    }
    Ms = {v: [] for v in VARIANTI}
    ha_tiri = {"HST", "AST"}.issubset(df.columns)

    while t <= fine:
        blocco = df[(df["Date"] >= t) & (df["Date"] < t + pd.Timedelta(days=passo))]
        train = df[(df["Date"] < t) & (df["Date"] >= t - pd.Timedelta(days=FINESTRA_GIORNI))]
        if len(blocco) and len(train) >= 150:
            m = adatta_modello(train, t, ridge, halflife, True)
            cnt = pd.concat([train["HomeTeam"], train["AwayTeam"]]).value_counts()
            m_freq = matrice_frequenze(train)
            m_t, conv = None, None
            if ha_tiri and not solo_base:
                tr_t = train.dropna(subset=["HST", "AST"])
                if len(tr_t) >= 150 and tr_t["HST"].sum() > 0 and tr_t["AST"].sum() > 0:
                    d_t = tr_t.copy()
                    d_t["FTHG"] = tr_t["HST"].astype(int)
                    d_t["FTAG"] = tr_t["AST"].astype(int)
                    scala = (tr_t["HST"] + tr_t["AST"]).mean() / max((tr_t["FTHG"] + tr_t["FTAG"]).mean(), 1e-9)
                    m_t = adatta_modello(d_t, t, ridge * scala, halflife, False)
                    conv = (
                        tr_t["FTHG"].sum() / max(tr_t["HST"].sum(), 1),
                        tr_t["FTAG"].sum() / max(tr_t["AST"].sum(), 1),
                    )

            for _, r in blocco.iterrows():
                h, a = r["HomeTeam"], r["AwayTeam"]
                lh, la = lambda_partita(m, h, a)
                Mb = matrice_punteggi(lh, la, m["rho"])
                Mgt = Mt = nan_m
                tiri_ok = False
                if m_t is not None:
                    lth, lta = lambda_partita(m_t, h, a)
                    ls_h, ls_a = lth * conv[0], lta * conv[1]
                    Mt = matrice_punteggi(ls_h, ls_a, m["rho"])
                    Mgt = matrice_punteggi(math.sqrt(lh * ls_h), math.sqrt(la * ls_a), m["rho"])
                    tiri_ok = True

                if solo_base:
                    Mn = Ma = Mx = nan_m
                    qp = qc = qoc = None
                else:
                    Mn = matrice_punteggi(lh, la, 0.0)
                    qp = prima_tripla(r, COL_PRESA)
                    qo = prima_coppia(r, COL_OU_PRESA)
                    if qp:
                        p_o = float(senza_margine(qo)[0]) if qo else None
                        Ma = matrice_ancorata(senza_margine(qp), p_o, m["rho"], (lh, la))
                        Mx = 0.5 * (Mb + Ma)
                    else:
                        Ma = Mx = nan_m
                    qc = prima_tripla(r, COL_CHIUSURA)
                    qoc = prima_coppia(r, COL_OU_CHIUSURA)

                rec["Partita"].append(f"{h} - {a}")
                rec["xs"].append(min(int(r["FTHG"]), 10))
                rec["ys"].append(min(int(r["FTAG"]), 10))
                rec["ok"].append(bool(qp))
                rec["ou_ok"].append(bool(qp and qoc))
                rec["m1x2"].append(senza_margine(qc) if qc else np.full(3, np.nan))
                rec["mou"].append(float(senza_margine(qoc)[0]) if qoc else np.nan)
                rec["min_storia"].append(int(min(cnt.get(h, 0), cnt.get(a, 0))))
                rec["tiri_ok"].append(tiri_ok)
                for nome, Mv in {
                    "Modello": Mb, "Senza Dixon-Coles": Mn, "Ancorato al mercato": Ma,
                    "Misto 50/50": Mx, "Gol + Tiri 50/50": Mgt, "Tiri in porta": Mt,
                    "Frequenze storiche": m_freq,
                }.items():
                    Ms[nome].append(Mv)
        t += pd.Timedelta(days=passo)

    if not rec["xs"]:
        return None
    out = {k: np.array(v) for k, v in rec.items() if k != "Partita"}
    out["Partita"] = rec["Partita"]
    out["Ms"] = {k: np.array(v) for k, v in Ms.items()}
    return out


def _p_mercato(Ms, chiave):
    return (Ms * _MASK[chiave]).sum(axis=(1, 2))


def _ll_bin(p, o):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(o * np.log(p) + (1 - o) * np.log(1 - p)))


def _vettore_perdita(Mv, chiave, xs, ys, outc, esito):
    if chiave == "1x2":
        P = np.stack([_p_mercato(Mv, "p1"), _p_mercato(Mv, "px"), _p_mercato(Mv, "p2")], axis=1)
        return -np.log(np.clip(P[np.arange(len(esito)), esito], 1e-9, 1))
    if chiave == "exact":
        return -np.log(np.clip(Mv[np.arange(len(Mv)), xs, ys], 1e-12, 1))
    p = np.clip(_p_mercato(Mv, chiave), 1e-9, 1 - 1e-9)
    o = outc[chiave]
    return -(o * np.log(p) + (1 - o) * np.log(1 - p))


def _diff_se(a, b):
    d = a - b
    se = d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
    return f"{d.mean():+.4f} ± {se:.4f}"


def pesi_ottimi(res):
    xs, ys = res["xs"], res["ys"]
    esito = np.where(xs > ys, 0, np.where(xs == ys, 1, 2))
    outc = {
        "over1.5": (xs + ys > 1.5).astype(float),
        "over2.5": (xs + ys > 2.5).astype(float),
        "over3.5": (xs + ys > 3.5).astype(float),
        "btts": ((xs > 0) & (ys > 0)).astype(float),
    }
    Mb, Mf = res["Ms"]["Modello"], res["Ms"]["Frequenze storiche"]
    gol, esatto = {}, {}
    for w in W_GRID:
        Mw = w * Mb + (1 - w) * Mf
        gol[w] = float(np.mean([
            _vettore_perdita(Mw, k_, xs, ys, outc, esito).mean()
            for k_ in ("over1.5", "over2.5", "over3.5", "btts")
        ]))
        esatto[w] = float(_vettore_perdita(Mw, "exact", xs, ys, outc, esito).mean())
    return {"gol": min(gol, key=gol.get), "esatto": min(esatto, key=esatto.get), "n": len(xs)}


def tabelle_test(res):
    usa_tiri = res.get("tiri_ok") is not None and bool(res["tiri_ok"].mean() >= 0.5)
    VARIANTI_A = VARIANTI if usa_tiri else [v for v in VARIANTI if v not in VARIANTI_TIRI]
    VARIANTI_MOD_A = [v for v in VARIANTI_MODELLO if v in VARIANTI_A]
    xs, ys = res["xs"], res["ys"]
    ok = res["ok"] & res["tiri_ok"] if usa_tiri else res["ok"]
    ou_ok = res["ou_ok"] & res["tiri_ok"] if usa_tiri else res["ou_ok"]
    esito = np.where(xs > ys, 0, np.where(xs == ys, 1, 2))
    outc = {
        "over1.5": (xs + ys > 1.5).astype(float),
        "over2.5": (xs + ys > 2.5).astype(float),
        "over3.5": (xs + ys > 3.5).astype(float),
        "btts": ((xs > 0) & (ys > 0)).astype(float),
    }
    mercati_def = [
        ("1X2", "1x2"), ("Over/Under 1.5", "over1.5"), ("Over/Under 2.5", "over2.5"),
        ("Over/Under 3.5", "over3.5"), ("Goal / No Goal", "btts"),
        ("Risultato esatto (log score)", "exact"),
    ]
    righe, migliore = [], {}
    for etichetta, mk in mercati_def:
        sel = ok.copy()
        if mk == "over2.5":
            sel = ou_ok.copy()
        if mk == "1x2":
            sel = ok & ~np.isnan(res["m1x2"]).any(axis=1)
        if sel.sum() < 10:
            continue
        xs_s, ys_s = xs[sel], ys[sel]
        outc_s = {k_: v[sel] for k_, v in outc.items()}
        es_s = esito[sel]
        L = {v: _vettore_perdita(res["Ms"][v][sel], mk, xs_s, ys_s, outc_s, es_s) for v in VARIANTI_A}
        r = {"Mercato": etichetta, "Partite": int(sel.sum())}
        for v in VARIANTI_A:
            r[v] = round(float(L[v].mean()), 4)
        Mb, Mf = res["Ms"]["Modello"][sel], res["Ms"]["Frequenze storiche"][sel]
        lw = {w: float(_vettore_perdita(w * Mb + (1 - w) * Mf, mk, xs_s, ys_s, outc_s, es_s).mean()) for w in W_GRID}
        w_best = min(lw, key=lw.get)
        r["Modello+Frequenze"] = round(lw[w_best], 4)
        r["Peso modello %"] = int(w_best * 100)
        if mk == "1x2":
            r["Mercato (chiusura)"] = round(float(
                -np.mean(np.log(np.clip(res["m1x2"][sel][np.arange(sel.sum()), es_s], 1e-9, 1)))
            ), 4)
        elif mk == "over2.5":
            r["Mercato (chiusura)"] = round(_ll_bin(res["mou"][sel], outc_s["over2.5"]), 4)
        else:
            r["Mercato (chiusura)"] = None
        r["Modello − Frequenze (± err.)"] = _diff_se(L["Modello"], L["Frequenze storiche"])
        r["Ancorato − Modello (± err.)"] = _diff_se(L["Ancorato al mercato"], L["Modello"])
        best = min(VARIANTI_MOD_A, key=lambda v: r[v])
        r["Migliore"] = best
        migliore[mk] = best
        righe.append(r)
    tab_a = pd.DataFrame(righe)

    # --- Presi / non presi + calibrazione Platt ---
    righe_b = []
    sel = ok
    for etichetta, mk, lato in [
        ("Goal", "btts", 1), ("No Goal", "btts", 0),
        ("Over 1.5", "over1.5", 1), ("Under 1.5", "over1.5", 0),
        ("Over 2.5", "over2.5", 1), ("Under 2.5", "over2.5", 0),
        ("Over 3.5", "over3.5", 1), ("Under 3.5", "over3.5", 0),
    ]:
        v = migliore.get(mk, "Modello")
        p_raw = _p_mercato(res["Ms"][v][sel], mk)
        p_raw = p_raw if lato == 1 else 1 - p_raw
        hit = (outc[mk][sel] == lato)
        # Platt su metà dati (in-sample walk-forward non perfetto, ma utile)
        n = len(p_raw)
        if n >= 60:
            mid = n // 2
            a_pl, b_pl = platt_fit(p_raw[:mid], hit[:mid].astype(float))
            p_cal = platt_apply(p_raw[mid:], a_pl, b_pl)
            hit_cal = hit[mid:]
            for thr in (55, 65, 75):
                s_ = p_cal >= thr / 100
                if s_.sum() >= 5:
                    righe_b.append({
                        "Mercato": etichetta, "Soglia %": thr, "Variante": v + " (Platt)",
                        "Selezioni": int(s_.sum()),
                        "Presi %": round(hit_cal[s_].mean() * 100, 1),
                        "Prob. dichiarata %": round(p_cal[s_].mean() * 100, 1),
                    })
        for thr in (55, 65, 75):
            s_ = p_raw >= thr / 100
            if s_.sum() >= 5:
                righe_b.append({
                    "Mercato": etichetta, "Soglia %": thr, "Variante": v,
                    "Selezioni": int(s_.sum()),
                    "Presi %": round(hit[s_].mean() * 100, 1),
                    "Prob. dichiarata %": round(p_raw[s_].mean() * 100, 1),
                })

    v1 = migliore.get("1x2", "Modello")
    P1 = np.stack([_p_mercato(res["Ms"][v1][sel], k_) for k_ in ("p1", "px", "p2")], axis=1)
    for i, nome in enumerate(["1 (Casa)", "X (Pareggio)", "2 (Trasferta)"]):
        for thr in (45, 55, 65):
            s_ = P1[:, i] >= thr / 100
            if s_.sum() >= 5:
                righe_b.append({
                    "Mercato": nome, "Soglia %": thr, "Variante": v1,
                    "Selezioni": int(s_.sum()),
                    "Presi %": round((esito[sel][s_] == i).mean() * 100, 1),
                    "Prob. dichiarata %": round(P1[s_, i].mean() * 100, 1),
                })
    tab_b = pd.DataFrame(righe_b)

    # Risultati esatti
    righe_c = []
    for v in VARIANTI_A:
        Mv = res["Ms"][v][ok].reshape(ok.sum(), -1)
        ordine = np.argsort(-Mv, axis=1)[:, :3]
        reale = xs[ok] * 11 + ys[ok]
        righe_c.append({
            "Variante": v,
            "Log score": round(float(-np.mean(np.log(np.clip(Mv[np.arange(len(Mv)), reale], 1e-12, 1)))), 4),
            "1° esatto preso %": round((ordine[:, 0] == reale).mean() * 100, 1),
            "Nei primi 3 %": round((ordine == reale[:, None]).any(axis=1).mean() * 100, 1),
            "Prob. media del 1° %": round(float(np.take_along_axis(Mv, ordine[:, :1], 1).mean() * 100), 1),
        })
    tab_c = pd.DataFrame(righe_c)
    best_ex = min(VARIANTI_MOD_A, key=lambda v: float(tab_c.loc[tab_c["Variante"] == v, "Log score"].iloc[0]))
    idx = np.where(ok)[0][-25:]
    Mv = res["Ms"][best_ex]
    righe_d = []
    for i in idx[::-1]:
        flat = Mv[i].ravel()
        top = np.argsort(-flat)[:3]
        reale = xs[i] * 11 + ys[i]
        stato = "✅ preso (1°)" if top[0] == reale else ("🟡 nei primi 3" if reale in top else "❌ non preso")
        righe_d.append({
            "Partita": res["Partita"][i],
            "1°": f"{top[0]//11}-{top[0]%11} ({flat[top[0]]*100:.0f}%)",
            "2°": f"{top[1]//11}-{top[1]%11} ({flat[top[1]]*100:.0f}%)",
            "3°": f"{top[2]//11}-{top[2]%11} ({flat[top[2]]*100:.0f}%)",
            "Risultato": f"{xs[i]}-{ys[i]}", "Esito": stato,
        })
    tab_d = pd.DataFrame(righe_d)

    # Neopromosse
    righe_e = []
    base_ok = ok & ~np.isnan(res["m1x2"]).any(axis=1)
    for nome, gruppo in [
        ("Squadra con meno di 15 partite di storico", res["min_storia"] < 15),
        ("Tutte le altre", res["min_storia"] >= 15),
    ]:
        sel_e = gruppo & base_ok
        if sel_e.sum() >= 5:
            es_e = esito[sel_e]
            riga = {"Gruppo": nome, "Partite": int(sel_e.sum())}
            for v in ("Modello", "Ancorato al mercato"):
                riga[f"Log loss 1X2 ({v})"] = round(float(
                    _vettore_perdita(res["Ms"][v][sel_e], "1x2", xs[sel_e], ys[sel_e], {}, es_e).mean()
                ), 4)
            riga["Log loss 1X2 (Mercato chiusura)"] = round(float(
                -np.mean(np.log(np.clip(res["m1x2"][sel_e][np.arange(sel_e.sum()), es_e], 1e-9, 1)))
            ), 4)
            righe_e.append(riga)
    return tab_a, tab_b, tab_c, tab_d, pd.DataFrame(righe_e), best_ex


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def test_completo_cached(div, ridge, halflife, passo, giorno, versione, chiave: str = ""):
    df = carica_storico(div, N_STAGIONI_TEST, giorno, chiave)
    return None if df is None else esegui_test_completo(df, ridge, halflife, passo)


@st.cache_data(ttl=6 * 3600, show_spinner="Calcolo pesi modello ↔ frequenze...")
def pesi_blend_auto(div, ridge, halflife, giorno, chiave: str = ""):
    """
    Pesi blend con test LEGGERO (3 stagioni, passo largo).
    Se troppo lento/vuoto → default sicuri 50% gol / 100% esatti sul modello.
    """
    default = {"gol": 0.5, "esatto": 1.0, "n": 0, "auto": False}
    df = carica_storico(div, N_STAGIONI, giorno, chiave)  # 3 stagioni, non 5
    if df is None or len(df) < 200:
        return default
    try:
        res = esegui_test_completo(df, ridge, halflife, max(PASSO_PARAM, 42), solo_base=True)
    except Exception:
        return default
    if res is None or len(res.get("xs", [])) < 80:
        return default
    w = pesi_ottimi(res)
    w["auto"] = True
    return w


# ======================= STATISTICHE / CLASSIFICA =======================
def stats_attese(df, casa, trasf, k=5):
    out = {}
    for nome, (h, a) in STAT_COLONNE.items():
        if h not in df.columns or a not in df.columns:
            continue
        d = df.dropna(subset=[h, a])
        if d.empty:
            continue
        lega_h, lega_a = d[h].mean(), d[a].mean()

        def sm(serie, lega):
            return (serie.sum() + k * lega) / (len(serie) + k)

        dc, dt = d[d["HomeTeam"] == casa], d[d["AwayTeam"] == trasf]
        out[nome] = (
            round((sm(dc[h], lega_h) + sm(dt[h], lega_h)) / 2, 1),
            round((sm(dt[a], lega_a) + sm(dc[a], lega_a)) / 2, 1),
        )
    return out


def tabella_classifica(df_stag):
    righe = {}
    for r in df_stag.itertuples():
        for sq, gf, gs in ((r.HomeTeam, r.FTHG, r.FTAG), (r.AwayTeam, r.FTAG, r.FTHG)):
            d = righe.setdefault(sq, {"G": 0, "Pt": 0, "GF": 0, "GS": 0})
            d["G"] += 1
            d["GF"] += gf
            d["GS"] += gs
            d["Pt"] += 3 if gf > gs else (1 if gf == gs else 0)
    out = pd.DataFrame([{
        "Squadra": sq, "Giocate": v["G"], "Punti": v["Pt"],
        "Media GF": round(v["GF"] / v["G"], 2), "Media GS": round(v["GS"] / v["G"], 2),
        "DR": round((v["GF"] - v["GS"]) / v["G"], 2),
    } for sq, v in righe.items()])
    return out.sort_values(["Punti", "DR"], ascending=False).reset_index(drop=True) if not out.empty else out


# ======================= INTERFACCIA =======================
st.markdown("### 🏆 Seleziona Campionato")
campionato = st.selectbox("Scegli il torneo", list(LEAGUES.keys()), label_visibility="collapsed")
div = LEAGUES[campionato]

try:
    _chiave_default = st.secrets.get("FOOTBALL_DATA_API_KEY", "")
except Exception:
    _chiave_default = ""
try:
    _af_default = st.secrets.get("API_FOOTBALL_KEY", "")
except Exception:
    _af_default = ""

st.sidebar.markdown("### ⚙ Configurazione")
chiave_api = st.sidebar.text_input(
    "🔑 football-data.org (calendario)", value=_chiave_default, type="password",
)
st.sidebar.caption("Calendario. Per CL/EL lo storico è gratis (openfootball); questa chiave è solo extra.")
chiave_af = st.sidebar.text_input(
    "🔑 API-Football / api-sports (quote)", value=_af_default, type="password",
)
st.sidebar.caption(
    "Quote 1X2 e O/U pre-match. Free = 100 req/giorno. "
    "Cache 6h → ~1 richiesta per lega ogni 6 ore."
)
st.sidebar.markdown("---")
st.sidebar.info(
    "💡 Parametri automatici: shrinkage, emivita, Elo, forma e xG. "
    "Non c'è niente da regolare a mano."
)
st.sidebar.caption(f"Versione test cache: {VERSIONE_TEST}")

giorno = date.today().isoformat()
df_storico = carica_storico(div, N_STAGIONI, giorno, chiave_api)
if df_storico is None or len(df_storico) < 80:
    if div in EURO_DIVS:
        st.error(
            "Impossibile caricare lo storico Champions/Europa (openfootball). "
            "Riprova tra poco o verifica la connessione. "
            "La chiave football-data.org è **opzionale** (fallback)."
        )
    else:
        st.warning("Impossibile scaricare lo storico da football-data.co.uk (o dati insufficienti). Riprova tra poco.")
    st.stop()

par = parametri_auto(div, giorno, chiave_api)
modello = modello_corrente(div, par["ridge"], par["halflife"], giorno, chiave_api)
if modello is None:
    st.warning("Modello non addestrabile (dati insufficienti).")
    st.stop()

df_elo = carica_clubelo(giorno)
elo_map = mappa_elo(modello["squadre"], df_elo)
form_map = forma_recente(df_storico, FORM_N)
df_xg = carica_understat_csv(giorno)
xg_map = mappa_understat(modello["squadre"], df_xg)

_fonti = ["football-data.co.uk", f"ClubElo {len(elo_map)}/{len(modello['squadre'])}"]
_fonti.append(f"Forma U{FORM_N} ({len(form_map)} sq)")
if xg_map:
    _fonti.append(f"Understat xG {len(xg_map)}/{len(modello['squadre'])}")
_fonti.append("API quote" if True else "")
st.caption("📡 Fonti: " + " · ".join(_fonti))

fx, fx_err = carica_fixtures(div)
oggi_ts = pd.Timestamp.now().normalize()
calendario, fonte_cal = costruisci_calendario(div, df_storico, fx, chiave_api, modello["squadre"])

pesi = pesi_blend_auto(div, par["ridge"], par["halflife"], giorno, chiave_api)
cal = calibrazione_live(div, par["ridge"], par["halflife"], giorno, chiave_api)
M_freq_live = matrice_frequenze(df_storico)

# Quote: 1) fixtures.csv  2) API-Football (sovrascrive se presenti)
quote_fixtures = {}
fonte_quote = "nessuna"
if fx is not None:
    for _, _r in fx.iterrows():
        _q = prima_tripla(_r, COL_PRESA)
        _qo = prima_coppia(_r, COL_OU_PRESA)
        if _q:
            p_ou = float(senza_margine(_qo)[0]) if _qo else None
            quote_fixtures[(_r["HomeTeam"], _r["AwayTeam"])] = (senza_margine(_q), p_ou)
            fonte_quote = "football-data.co.uk (fixtures.csv)"

api_odds = {}
if chiave_af:
    api_odds = scarica_odds_api_football(div, chiave_af, giorno)
    if api_odds:
        fonte_quote = "API-Football" if not quote_fixtures else "API-Football + fixtures.csv"
        # integra: se match già in quote_fixtures, aggiorna; altrimenti aggiungi via fuzzy dopo


def _risolvi_quote(casa, trasf):
    """Priorità: fixtures.csv esatto → API-Football (nome normalizzato) → None."""
    q = quote_fixtures.get((casa, trasf))
    if q is not None:
        return q[0], q[1], "fixtures.csv"
    aq = _match_quote_api(api_odds, casa, trasf)
    if aq is not None:
        return aq[0], aq[1], "API-Football"
    return None, None, None


def distribuzione(casa, trasf):
    lh, la, elo_c, elo_t, fc, ft = lambda_full(
        modello, casa, trasf, elo_map, form_map, ELO_BLEND, FORM_BLEND
    )
    fxg_h, fxg_a = fattore_xg(xg_map, casa, trasf)
    if xg_map and XG_BLEND > 0 and (casa in xg_map or trasf in xg_map):
        lh, la = lh * (fxg_h ** XG_BLEND), la * (fxg_a ** XG_BLEND)
    Mb = matrice_punteggi(lh, la, modello["rho"])
    p1x2, p_ou, src = _risolvi_quote(casa, trasf)
    bits = []
    if elo_c is not None and elo_t is not None:
        bits.append(f"ClubElo {elo_c:.0f}–{elo_t:.0f} ({int(ELO_BLEND*100)}%)")
    if fc and ft:
        bits.append(f"Forma U{FORM_N}: {fc['form_score']:+.2f}/{ft['form_score']:+.2f} ({int(FORM_BLEND*100)}%)")
    if casa in xg_map or trasf in xg_map:
        bits.append(f"xG ({int(XG_BLEND*100)}%)")
    extra_src = (" · " + " · ".join(bits)) if bits else ""
    if p1x2 is not None:
        Ma = matrice_ancorata(p1x2, p_ou, modello["rho"], (lh, la))
        extra = "1X2 + O/U 2.5" if p_ou is not None else "solo 1X2"
        return {
            "M_1x2": Ma, "M_gol": Ma, "M_es": Ma, "ancorata": True,
            "fonte": f"ancorata alle quote ({extra}, {src}){extra_src}",
            "elo_c": elo_c, "elo_t": elo_t, "form_c": fc, "form_t": ft, "lh": lh, "la": la,
        }
    Mg = pesi["gol"] * Mb + (1 - pesi["gol"]) * M_freq_live
    Me = pesi["esatto"] * Mb + (1 - pesi["esatto"]) * M_freq_live
    return {
        "M_1x2": Mb, "M_gol": Mg, "M_es": Me, "ancorata": False,
        "fonte": (
            f"modello + frequenze + prior (Elo/forma/xG); peso modello "
            f"{pesi['gol']*100:.0f}% Goal/Over, {pesi['esatto']*100:.0f}% esatti"
            f"{extra_src} – nessuna quota bookmaker"
        ),
        "elo_c": elo_c, "elo_t": elo_t, "form_c": fc, "form_t": ft, "lh": lh, "la": la,
    }



def ritardi_squadra(df, squadra: str, max_look=20):
    """
    Ritardi recenti: partite consecutive senza segnare, senza subire, senza BTTS, under 2.5, ecc.
    Conta all'indietro dall'ultima partita giocata.
    """
    if df is None or squadra is None:
        return {}
    d = df[(df["HomeTeam"] == squadra) | (df["AwayTeam"] == squadra)].sort_values("Date")
    if d.empty:
        return {}
    d = d.tail(max_look)
    out = {}

    def streak(pred):
        n = 0
        for _, r in d.iloc[::-1].iterrows():
            if pred(r):
                n += 1
            else:
                break
        return n

    out["non_segna"] = streak(
        lambda r: (r["HomeTeam"] == squadra and r["FTHG"] == 0)
        or (r["AwayTeam"] == squadra and r["FTAG"] == 0)
    )
    out["non_subisce"] = streak(
        lambda r: (r["HomeTeam"] == squadra and r["FTAG"] == 0)
        or (r["AwayTeam"] == squadra and r["FTHG"] == 0)
    )
    out["no_btts"] = streak(lambda r: r["FTHG"] == 0 or r["FTAG"] == 0)
    out["under25"] = streak(lambda r: (r["FTHG"] + r["FTAG"]) <= 2)
    out["over25"] = streak(lambda r: (r["FTHG"] + r["FTAG"]) >= 3)
    out["non_vince"] = streak(
        lambda r: not (
            (r["HomeTeam"] == squadra and r["FTHG"] > r["FTAG"])
            or (r["AwayTeam"] == squadra and r["FTAG"] > r["FTHG"])
        )
    )
    return out


def gol_attesi(M):
    return (
        float((M.sum(axis=1) * np.arange(M.shape[0])).sum()),
        float((M.sum(axis=0) * np.arange(M.shape[1])).sum()),
    )


# Value finder (fixtures.csv; se assenti quote CSV ma presenti API, le usiamo ricostruendo quote da p)
righe_val = []
viste = set()
if fx is not None:
    for _, r in fx.iterrows():
        if "Date" in r and pd.notna(r["Date"]) and r["Date"] < oggi_ts:
            continue
        casa, trasf = r["HomeTeam"], r["AwayTeam"]
        viste.add((casa, trasf))
        q = prima_tripla(r, COL_PRESA)
        p1x2_api, p_ou_api, src = None, None, None
        if not q:
            p1x2_api, p_ou_api, src = _risolvi_quote(casa, trasf)
            if p1x2_api is None:
                continue
            q = [1.0 / max(p, 1e-6) for p in p1x2_api]
        lh, la, _, _, _, _ = lambda_full(modello, casa, trasf, elo_map, form_map, ELO_BLEND, FORM_BLEND)
        mk = mercati(matrice_punteggi(lh, la, modello["rho"]))
        P = np.array([mk["p1"], mk["px"], mk["p2"]])
        pm = senza_margine(q)
        nomi = [f"1 ({casa})", "X (Pareggio)", f"2 ({trasf})"]
        k = int(np.argmax(P * np.array(q)))
        righe_val.append({
            "Data": r["Date"].strftime("%d/%m") if "Date" in r and pd.notna(r["Date"]) else "",
            "Partita": f"{casa} vs {trasf}",
            "Selezione": nomi[k],
            "P1 %": round(P[0] * 100, 1), "PX %": round(P[1] * 100, 1), "P2 %": round(P[2] * 100, 1),
            "Prob_Modello": round(P[k] * 100, 1), "Prob_Mercato": round(pm[k] * 100, 1),
            "Quota_Book": round(q[k], 2), "Edge": round((P[k] * q[k] - 1) * 100, 1),
            "Nota": "⚠ senza storico" if (casa not in modello["idx"] or trasf not in modello["idx"]) else "",
        })
# partite solo da API (non in fixtures.csv)
for (ah, aa), (p1x2, p_ou, home, away) in (api_odds or {}).items():
    # skip se già coperta da fixtures con stesso nome normalizzato
    if any(_norm_sq(c) == ah and _norm_sq(t) == aa for c, t in viste):
        continue
    q = [1.0 / max(p, 1e-6) for p in p1x2]
    lh, la, _, _, _, _ = lambda_full(modello, home, away, elo_map, form_map, ELO_BLEND, FORM_BLEND)
    mk = mercati(matrice_punteggi(lh, la, modello["rho"]))
    P = np.array([mk["p1"], mk["px"], mk["p2"]])
    pm = p1x2
    nomi = [f"1 ({home})", "X (Pareggio)", f"2 ({away})"]
    k = int(np.argmax(P * np.array(q)))
    righe_val.append({
        "Data": "",
        "Partita": f"{home} vs {away}",
        "Selezione": nomi[k],
        "P1 %": round(P[0] * 100, 1), "PX %": round(P[1] * 100, 1), "P2 %": round(P[2] * 100, 1),
        "Prob_Modello": round(P[k] * 100, 1), "Prob_Mercato": round(pm[k] * 100, 1),
        "Quota_Book": round(q[k], 2), "Edge": round((P[k] * q[k] - 1) * 100, 1),
        "Nota": "API-Football" if (home in modello["idx"] and away in modello["idx"]) else "⚠ senza storico",
    })
df_value = pd.DataFrame(righe_val).sort_values("Edge", ascending=False) if righe_val else pd.DataFrame()

tab_calendario, tab_classifica, tab_value_finder, tab_ai_schedine, tab_monte_carlo, tab_backtest, tab_kelly = st.tabs([
    "📅 Calendario", "🏆 Classifica", "⚡ Value Finder", "🤖 Schedine AI",
    "🎲 Monte Carlo", "🧪 Backtest", "🧮 Value Bet",
])


def mostra_analisi(casa, trasf, data, ora):
    d = distribuzione(casa, trasf)
    lh, la = gol_attesi(d["M_gol"])
    M = d["M_es"]
    mk = mercati(d["M_1x2"])
    mg = mercati(d["M_gol"])
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
                    border: 1px solid #312e81; border-radius: 16px; padding: 16px 22px;
                    margin-top: 18px; box-shadow: 0 16px 24px -5px rgba(0,0,0,0.45);
                    display:flex; justify-content:space-between; align-items:center;">
            <span style="font-weight:800; font-size:1.15rem; color:#f8fafc;">📋 SCHEDA PARTITA</span>
            <span style="color:#94a3b8; font-size:0.9rem;">📅 {data} {ora}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col_sq, col_st = st.columns([1.2, 1.8])
    with col_sq:
        st.markdown(
            f"""
            <div style="text-align:center; padding:28px 18px; background:rgba(15,23,42,0.6);
                        border-radius:14px; border:1px solid #1e293b;">
                <h2 style="color:#38bdf8; margin:0; font-size:1.35rem;">{casa}</h2>
                <p style="color:#64748b; margin:4px 0 16px 0; font-size:0.8rem;">CASA</p>
                <h3 style="color:#a855f7; margin:0; font-size:1.1rem;">VS</h3>
                <h2 style="color:#f43f5e; margin:16px 0 0 0; font-size:1.35rem;">{trasf}</h2>
                <p style="color:#64748b; margin:4px 0 0 0; font-size:0.8rem;">TRASFERTA</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_st:
        st.markdown(
            "<p style='text-align:center; font-weight:700; color:#94a3b8; margin-bottom:12px;'>STATISTICHE ATTESE</p>",
            unsafe_allow_html=True,
        )
        righe_st = {"Gol Attesi": (round(lh, 2), round(la, 2))}
        righe_st.update(stats_attese(df_storico, casa, trasf))
        for label, (vc, vt) in righe_st.items():
            st.markdown(
                f"""
                <div style="display:flex; justify-content:space-between; align-items:center;
                            background:#111827; padding:11px 16px; border-radius:10px;
                            border:1px solid #1f2937; margin-bottom:8px;">
                    <span style="font-weight:700; color:#38bdf8; font-size:1.05rem; width:55px; text-align:left;">{vc}</span>
                    <span style="color:#94a3b8; font-size:0.85rem; font-weight:600; text-align:center; flex-grow:1;">{label}</span>
                    <span style="font-weight:700; color:#f43f5e; font-size:1.05rem; width:55px; text-align:right;">{vt}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(f"<h3>🔬 Poisson / Dixon-Coles · {casa} vs {trasf}</h3>", unsafe_allow_html=True)
    st.caption(f"Fonte Goal / Over / Esatti: {d['fonte']}.")
    if not d["ancorata"]:
        st.error(
            "🚫 **Niente scommesse consigliate** su questa partita: mancano le quote bookmaker. "
            "Le % sotto sono solo indicative (modello + forma/Elo). Non usarle come value bet."
        )
    a1, a2, a3 = st.columns(3)
    a1.metric("1 (Casa)", f"{mk['p1']*100:.1f}%", f"Equa: {1/mk['p1']:.2f}")
    a2.metric("X (Pareggio)", f"{mk['px']*100:.1f}%", f"Equa: {1/mk['px']:.2f}")
    a3.metric("2 (Trasferta)", f"{mk['p2']*100:.1f}%", f"Equa: {1/mk['p2']:.2f}")

    syn = applica_affidabilita(sintesi_pronostico(d["M_1x2"]), cal, d["ancorata"])
    st.markdown("#### Pronostico")
    if syn.get("calib"):
        st.caption(
            f"1X2 calibrato su {cal['n']} partite passate: la percentuale è più vicina a quante volte è uscito davvero. "
            + _testo_storico(cal, {"1": "p1", "X": "px", "2": "p2"}[syn["segno"]],
                            {"1": syn["p1"], "X": syn["px"], "2": syn["p2"]}[syn["segno"]])
            + "."
        )
    elif d["ancorata"]:
        st.caption(
            "1X2 preso dalle quote, senza margine del book: è la stima più affidabile che abbiamo."
        )
    else:
        st.caption("Calibrazione non disponibile su questo campionato: percentuali del modello, da prendere con più cautela.")
    if syn["equilibrata"]:
        st.caption(
            "Partita equilibrata: i due esiti sono vicini e il pareggio è competitivo, quindi il segno è **X**."
        )
    else:
        st.caption(
            f"Segno sul massimo di probabilità (**{syn['segno_max']}**). "
            f"X = {syn['px']*100:.0f}%."
        )
    st.caption(
        "Da usare: segno, doppia chance, Over 2.5 e Goal/No Goal. "
        "Combo, multigol e risultato esatto sono **indicativi** (fasce larghe o probabilità basse)."
    )
    p1c, p2c, p3c, p4c, p5c = st.columns(5)
    p1c.metric("Segno 1X2", syn["segno"], f"{({'1': syn['p1'], 'X': syn['px'], '2': syn['p2']})[syn['segno']]*100:.0f}%")
    p2c.metric("Doppia chance", syn["dc"], f"{syn['dc_p']*100:.0f}%")
    p3c.metric("Combo gol", syn["combo"], f"{syn['combo_p']*100:.0f}%")
    p4c.metric("Combo O/U", syn["ou"].split("+ ", 1)[-1], f"{syn['ou_p']*100:.0f}%")
    p5c.metric("Multigol", syn["mg"], f"{syn['mg_p']*100:.0f}%")
    def _riga_card(titolo, voci, scelta=None):
        celle = []
        for nome, p in voci:
            attiva = scelta is not None and nome == scelta
            bordo = "#38bdf8" if attiva else "#1f2937"
            celle.append(
                f'<div style="flex:1; min-width:108px; background:#111827; border:1px solid {bordo};'
                f' border-radius:12px; padding:10px 12px;">'
                f'<div style="color:#94a3b8; font-size:0.75rem; line-height:1.2;">{nome}</div>'
                f'<div style="color:#38bdf8; font-weight:800; font-size:1.15rem; margin-top:4px;">{p*100:.0f}%</div>'
                f'</div>'
            )
        st.markdown(
            f'<p style="color:#cbd5e1; font-weight:700; font-size:0.92rem; margin:16px 0 8px 0;">{titolo}</p>'
            f'<div style="display:flex; gap:8px; flex-wrap:wrap;">{"".join(celle)}</div>',
            unsafe_allow_html=True,
        )

    _riga_card("Doppia chance", [(n, syn["dc_all"][n]) for n in ("1X", "X2", "12")], scelta=syn["dc"])
    _riga_card(f"Indicativo · combo gol · segno {syn['segno']}", syn["combo_top"], scelta=syn["combo"])
    _riga_card(f"Indicativo · combo Over/Under · segno {syn['segno']}", syn["ou_top"], scelta=syn["ou"])
    _riga_card("Indicativo · multigol partita (fascia larga)", syn["mg_top"], scelta=syn["mg"])
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Multigol Casa", syn["mg_casa"], f"{syn['mg_casa_p']*100:.0f}%")
    h2.metric("Multigol Ospite", syn["mg_osp"], f"{syn['mg_osp_p']*100:.0f}%")
    h3.metric("1° tempo · indicativo", syn["mg_1t"], f"{syn['mg_1t_p']*100:.0f}%")
    h4.metric("2° tempo · indicativo", syn["mg_2t"], f"{syn['mg_2t_p']*100:.0f}%")
    _riga_card("Fasce casa · indicativo", syn["mg_casa_top"], scelta=syn["mg_casa"])
    _riga_card("Fasce ospite · indicativo", syn["mg_osp_top"], scelta=syn["mg_osp"])
    _riga_card("1° tempo · stima, non è uno storico", syn["mg_1t_top"], scelta=syn["mg_1t"])
    _riga_card("2° tempo · stima, non è uno storico", syn["mg_2t_top"], scelta=syn["mg_2t"])
    if d.get("elo_c") is not None or d.get("elo_t") is not None:
        e1, e2, e3 = st.columns(3)
        e1.metric("ClubElo Casa", f"{d['elo_c']:.0f}" if d.get("elo_c") else "—")
        e2.metric("Δ Elo", f"{(d.get('elo_c') or 0) - (d.get('elo_t') or 0):+.0f}" if d.get("elo_c") and d.get("elo_t") else "—")
        e3.metric("ClubElo Trasferta", f"{d['elo_t']:.0f}" if d.get("elo_t") else "—")
    if d.get("form_c") or d.get("form_t"):
        f1, f2, f3 = st.columns(3)
        fc, ft = d.get("form_c") or {}, d.get("form_t") or {}
        f1.metric(f"Forma casa (U{FORM_N})", f"{fc.get('form_score', 0):+.2f}" if fc else "—",
                  f"GF {fc.get('gf', '—')} GS {fc.get('gs', '—')}" if fc else None)
        f2.metric("Δ Forma", f"{(fc.get('form_score') or 0) - (ft.get('form_score') or 0):+.2f}" if fc and ft else "—")
        f3.metric(f"Forma trasf. (U{FORM_N})", f"{ft.get('form_score', 0):+.2f}" if ft else "—",
                  f"GF {ft.get('gf', '—')} GS {ft.get('gs', '—')}" if ft else None)

    st.markdown("#### ⚽ Under / Over & Goal / No Goal")
    over25 = mg["over2.5"] if d["ancorata"] else prob_calibrata(mg["over2.5"], cal, "over25")
    btts = mg["btts"] if d["ancorata"] else prob_calibrata(mg["btts"], cal, "btts")
    u1, u2, u3, u4, u5 = st.columns(5)
    u1.metric("Over 2.5", f"{over25*100:.1f}%", f"Under: {(1-over25)*100:.1f}%")
    u2.metric("Over 1.5", f"{mg['over1.5']*100:.1f}%", "indicativo")
    u3.metric("Over 3.5", f"{mg['over3.5']*100:.1f}%", "indicativo")
    u4.metric("Goal (BTTS)", f"{btts*100:.1f}%", f"Equa: {1/max(btts, 1e-6):.2f}")
    u5.metric("No Goal", f"{(1-btts)*100:.1f}%", f"Equa: {1/max(1-btts, 1e-6):.2f}")
    st.caption(
        "Over 2.5: " + _testo_storico(cal, "over25", over25, d["ancorata"])
        + ". Goal: " + _testo_storico(cal, "btts", btts, False) + "."
    )

    # --- Ritardi squadre (stile "scheda giornata") ---
    st.markdown("#### ⏱️ Ritardi squadre")
    rc = ritardi_squadra(df_storico, casa)
    rt = ritardi_squadra(df_storico, trasf)
    if rc or rt:
        r1, r2 = st.columns(2)
        with r1:
            st.markdown(f"**{casa}**")
            if rc:
                st.write(
                    f"- Non segna da **{rc.get('non_segna', 0)}** partite\n"
                    f"- Non subisce da **{rc.get('non_subisce', 0)}**\n"
                    f"- Under 2.5 da **{rc.get('under25', 0)}** · Over 2.5 da **{rc.get('over25', 0)}**\n"
                    f"- No BTTS da **{rc.get('no_btts', 0)}** · Non vince da **{rc.get('non_vince', 0)}**"
                )
            else:
                st.caption("Nessuno storico recente")
        with r2:
            st.markdown(f"**{trasf}**")
            if rt:
                st.write(
                    f"- Non segna da **{rt.get('non_segna', 0)}** partite\n"
                    f"- Non subisce da **{rt.get('non_subisce', 0)}**\n"
                    f"- Under 2.5 da **{rt.get('under25', 0)}** · Over 2.5 da **{rt.get('over25', 0)}**\n"
                    f"- No BTTS da **{rt.get('no_btts', 0)}** · Non vince da **{rt.get('non_vince', 0)}**"
                )
            else:
                st.caption("Nessuno storico recente")
        st.caption("I ritardi sono descrittivi, non un segnale automatico di scommessa.")
    else:
        st.caption("Ritardi non disponibili per queste squadre.")

    st.markdown("#### 🎯 Top 3 Risultati Esatti · indicativo")
    st.caption(
        "Non è un pronostico affidabile: in backtest il 1° esatto esce circa il 10–16% delle volte. "
        "Guardalo solo come curiosità, non come selezione."
    )
    flat = sorted(((i, j, M[i, j]) for i in range(M.shape[0]) for j in range(M.shape[1])), key=lambda t: t[2], reverse=True)[:3]
    for col, medaglia, nome, (i, j, p) in zip(st.columns(3), ["🥇", "🥈", "🥉"], ["1°", "2°", "3°"], flat):
        col.info(f"{medaglia} {nome}: **{i} - {j}** ({p*100:.1f}%)")
    # Presentazione: risultato più probabile vs mercato largo (senza toccare la matrice)
    if flat:
        bi, bj, bp = flat[0]
        if bi > bj:
            lato_es = "1"
        elif bi < bj:
            lato_es = "2"
        else:
            lato_es = "X"
        tot_es = bi + bj
        if tot_es >= 3:
            stretto = f"{lato_es} + Over 2.5"
        elif tot_es <= 2:
            stretto = f"{lato_es} + Under 2.5"
        else:
            stretto = lato_es
        st.markdown(
            f"""
            <div style="margin-top:10px; padding:12px 14px; background:#0f172a; border:1px solid #334155;
                        border-radius:12px; line-height:1.55;">
              <div style="color:#e2e8f0; font-size:0.92rem;">
                <strong style="color:#38bdf8;">Risultato più probabile:</strong>
                {bi}-{bj} ({bp*100:.1f}%) → <span style="color:#a5b4fc;">{stretto}</span>
              </div>
              <div style="color:#e2e8f0; font-size:0.92rem; margin-top:6px;">
                <strong style="color:#f472b6;">Mercato largo:</strong>
                {syn.get("ou", "—")} ({syn.get("ou_p", 0)*100:.0f}%)
                <span style="color:#64748b; font-size:0.8rem;"> · somma più esiti, non solo il punteggio top</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    if d["ancorata"]:
        st.caption("Con quote reali le probabilità 1X2 coincidono col mercato (senza margine): non c'è edge automatico sul 1X2.")
    if casa not in modello["idx"] or trasf not in modello["idx"]:
        st.warning(
            "Una delle due squadre non è nel modello (nome non riconosciuto o senza storico). "
            "I gol attesi in quel caso sono la media di lega, non la media di quella squadra."
        )


# --- CALENDARIO / VISTA GIORNATA ---
with tab_calendario:
    st.subheader("📅 Vista giornata")
    st.caption(f"Fonte calendario: {fonte_cal} · % = vostro modello (Dixon-Coles + prior)")
    if calendario.empty:
        st.info("Nessuna partita disponibile.")
    else:
        giornate = sorted(g for g in calendario["Giornata"].unique() if g)
        non_giocate = calendario[calendario["Stato"] != "FINISHED"]
        g_def = int(non_giocate["Giornata"].min()) if not non_giocate.empty and pd.notna(non_giocate["Giornata"].min()) else int(giornate[-1])
        cc1, cc2 = st.columns([1, 1])
        cc1.selectbox("🏆 Torneo", [campionato], key="cal_torneo")
        giornata_sel = cc2.selectbox(
            "📅 Giornata", giornate,
            index=giornate.index(g_def) if g_def in giornate else 0,
            key="cal_giornata",
        )
        block = calendario[calendario["Giornata"] == giornata_sel].reset_index(drop=True)
        st.markdown(f"**{len(block)} partite** · clicca **Scheda** per dettagli, multigol e ritardi")

        for idx, row in block.iterrows():
            casa, trasf = row["Casa"], row["Trasferta"]
            gc = row["GolC"] if pd.notna(row["GolC"]) else None
            gt = row["GolT"] if pd.notna(row["GolT"]) else None
            finita = row.get("Stato") == "FINISHED" and gc is not None
            # % rapide modello
            try:
                d_r = distribuzione(casa, trasf)
                syn = applica_affidabilita(sintesi_pronostico(d_r["M_1x2"]), cal, d_r["ancorata"])
                gh, ga = gol_attesi(d_r["M_gol"])
            except Exception:
                syn, d_r, gh, ga = None, None, None, None

            c_left, c_mid, c_btn = st.columns([2.2, 3.6, 0.8])
            with c_left:
                if finita:
                    st.markdown(f"**{casa}** {int(gc)}-{int(gt)} **{trasf}**")
                    st.caption(f"{row['Data']} {row['Ora']} · conclusa")
                else:
                    st.markdown(f"**{casa}** vs **{trasf}**")
                    st.caption(f"{row['Data']} · {row['Ora']}")
            with c_mid:
                if syn:
                    fonte = "quote" if d_r and d_r["ancorata"] else ("calibrato" if syn.get("calib") else "modello")
                    st.caption(
                        f"Pron. **{syn['segno']}** ({fonte}) · DC **{syn['dc']}** {syn['dc_p']*100:.0f}% · "
                        f"{syn['combo']} {syn['combo_p']*100:.0f}% · {syn['ou']} {syn['ou_p']*100:.0f}% · MG **{syn['mg']}**"
                    )
                    st.caption(
                        f"1 {syn['p1']*100:.0f}% · X {syn['px']*100:.0f}% · 2 {syn['p2']*100:.0f}%"
                        f"  ·  gol attesi {gh:.2f}-{ga:.2f}"
                        f"  ·  MG casa {syn['mg_casa']} · osp {syn['mg_osp']}"
                    )
                else:
                    st.caption("Probabilità non calcolabili")
            with c_btn:
                if st.button("Scheda", key=f"btn_{div}_{giornata_sel}_{idx}"):
                    st.session_state["match_attivo"] = {
                        "div": div, "casa": casa, "trasf": trasf,
                        "data": row["Data"], "ora": row["Ora"],
                    }

    att_m = st.session_state.get("match_attivo")
    if att_m and att_m.get("div") == div:
        st.divider()
        mostra_analisi(att_m["casa"], att_m["trasf"], att_m["data"], att_m["ora"])


# --- CLASSIFICA ---
with tab_classifica:
    st.subheader("🏆 Classifica e Rendimento")
    df_cls = tabella_classifica(df_storico[df_storico["Stag"] == df_storico["Stag"].max()])
    if df_cls.empty:
        st.warning("Dati non disponibili.")
    else:
        st.caption("Stagione in corso (football-data.co.uk). Media GF/GS e DR per partita.")
        st.dataframe(df_cls, use_container_width=True, hide_index=True)
        st.download_button("📥 Scarica CSV", df_cls.to_csv(index=False).encode("utf-8"), "classifica.csv", "text/csv")
        try:
            df_plot = df_cls.copy()
            if "Media GF" in df_plot.columns:
                df_plot = df_plot.rename(columns={"Media GF": "Media_GF"})
                fig = px.bar(
                    df_plot, x="Squadra", y="Punti", color="Media_GF",
                    template="plotly_dark", title="Punti e potenziale offensivo",
                )
            else:
                fig = px.bar(
                    df_plot, x="Squadra", y="Punti",
                    template="plotly_dark", title="Punti",
                )
            fig.update_layout(xaxis_tickangle=-45, margin=dict(b=80))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.caption(f"Grafico non disponibile ({type(e).__name__}). La tabella sopra resta valida.")

    st.markdown("#### 📊 Forza squadre (modello)")
    st.caption("Scala log rispetto alla media (0). Attacco alto = segna di più; difesa alta = subisce meno.")
    df_f = pd.DataFrame({
        "Squadra": modello["squadre"],
        "Attacco": np.round(modello["att"], 3),
        "Difesa": np.round(-modello["dfn"], 3),
    })
    df_f["Elo"] = [round(elo_map[s]["elo"], 1) if s in elo_map else None for s in modello["squadre"]]
    df_f[f"Forma U{FORM_N}"] = [form_map[s]["form_score"] if s in form_map else None for s in modello["squadre"]]
    df_f["xG for"] = [round(xg_map[s]["xg_for"], 2) if s in xg_map else None for s in modello["squadre"]]
    df_f["Totale"] = (df_f["Attacco"] + df_f["Difesa"]).round(3)
    k1, k2, k3 = st.columns(3)
    k1.metric("Vantaggio casa (×)", f"{math.exp(modello['hm']):.2f}")
    k2.metric("Gol base (media, trasferta)", f"{math.exp(modello['mu']):.2f}")
    k3.metric("ρ Dixon-Coles", f"{modello['rho']:+.3f}")
    st.dataframe(df_f.sort_values("Totale", ascending=False), use_container_width=True, hide_index=True)
    with st.expander("🔍 Debug matching nomi (Elo / xG / quote)"):
        st.caption(
            "Se Elo o xG sono '—', il nome squadra non ha trovato corrispondenza nella fonte. "
            "Utile per capire perché manca un prior."
        )
        log_df = log_matching(
            modello["squadre"], elo_map, xg_map,
            list(quote_fixtures.keys()) if quote_fixtures else [],
            api_odds if "api_odds" in dir() else {},
        )
        st.dataframe(log_df, use_container_width=True, hide_index=True)


# --- VALUE FINDER ---
with tab_value_finder:
    st.subheader("⚡ Scanner Value Finder")
    st.warning(
        "Nei backtest multi-lega il **CLV medio sull'edge del modello è negativo** (~−5/−7%). "
        "Un Edge alto qui **non** è un tip automatico: confronta sempre col tuo book e col tab Backtest."
    )
    if fx is None:
        st.warning(f"Impossibile scaricare fixtures/quote: {fx_err}.")
    elif df_value.empty:
        st.info(
            "Nessuna partita futura con quote. Il file fixtures.csv viene aggiornato "
            "pochi giorni prima di ogni turno. **Senza quote non ci sono value bet.**"
        )
    else:
        st.caption(
            f"Fonte quote: {fonte_quote}. "
            "Edge = prob. modello × quota − 1 (modello puro vs quota book). "
            "Con probabilità ancorate l'edge 1X2 sarebbe ~0 (margine book)."
        )
        c_a, c_b = st.columns(2)
        min_edge = c_a.slider("Edge minimo (%)", 0.0, 25.0, 8.0, help="Default 8%: sotto è quasi sempre rumore")
        solo_storico = c_b.checkbox("Solo squadre con storico completo", value=True)
        vf = df_value.copy()
        if solo_storico:
            vf = vf[vf["Nota"].fillna("") == ""]
        vf = vf[vf["Edge"] >= min_edge]
        if vf.empty:
            st.info("Nessuna riga con questi filtri. Alza/abbassa l'edge o togli il filtro storico.")
        else:
            st.dataframe(vf, use_container_width=True, hide_index=True)
            st.caption(
                f"{len(vf)} righe · Ordinate per Edge. "
                "Prima di puntare: quota del **tuo** book ≥ Quota_Book e probabilità coerente col Backtest."
            )
        if (df_value["Nota"].fillna("") != "").any():
            st.caption("⚠ Righe con nota = squadre senza storico (spesso sopravvalutate).")


# --- SCHEDINE AI ---
with tab_ai_schedine:
    st.subheader("🤖 Schedine Smart & Combo")
    st.error(
        "⛔ **Le multiple non sono consigliate.** "
        "L'EV combinato è ottimistico: moltiplica errori del modello e il margine del book. "
        "In backtest l'edge singolo ha già CLV negativo — in multipla peggiora."
    )
    base = df_value[df_value["Nota"].fillna("") == ""] if not df_value.empty else df_value
    if base.empty or len(base) < 2:
        st.info("Servono almeno 2 partite del prossimo turno con quote e squadre con storico.")
    else:
        n_disp = len(base)
        # max 3 eventi (prima 6)
        num_ev = st.slider("Eventi in multipla (max 3)", 2, min(3, n_disp), 2) if n_disp > 2 else 2
        # solo edge più alti
        base2 = base[base["Edge"] >= 5].head(num_ev) if "Edge" in base.columns else base.head(num_ev)
        if len(base2) < 2:
            base2 = base.head(num_ev)
        subset = base2
        quota_tot = float(np.prod(subset["Quota_Book"].values))
        prob_comb = float(np.prod(subset["Prob_Modello"].values / 100))
        # haircut empirico: ~15% di probabilità per correlazione/overconfidence
        prob_adj = prob_comb * (0.85 ** max(0, len(subset) - 1))
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("📈 Quota accumulatore", f"{quota_tot:.2f}")
        s2.metric("Prob. grezza modello", f"{prob_comb*100:.1f}%")
        s3.metric("Prob. aggiustata", f"{prob_adj*100:.1f}%", help="Haircut per multipla")
        s4.metric("EV aggiustato", f"{(prob_adj*quota_tot-1)*100:+.1f}%")
        st.dataframe(subset, use_container_width=True, hide_index=True)
        st.caption(
            "Preferisci **singole** selezionate dal Value Finder filtrato + confronto col tuo book. "
            "Questa tab resta solo per sperimentare, non come metodo principale."
        )


# --- MONTE CARLO ---
with tab_monte_carlo:
    st.subheader("🎲 Simulatore Monte Carlo")
    nomi = modello["squadre"]
    c1, c2, c3 = st.columns(3)
    sq_c = c1.selectbox("Casa", nomi, index=0, key="mc_casa")
    sq_t = c2.selectbox("Trasferta", nomi, index=min(1, len(nomi) - 1), key="mc_trasf")
    iterazioni = c3.slider("Iterazioni", 1000, 20000, 5000, step=1000, key="mc_iter")
    if st.button("🚀 Esegui simulazione", key="btn_mc"):
        if sq_c == sq_t:
            st.info("Scegli due squadre diverse.")
        else:
            d_mc = distribuzione(sq_c, sq_t)
            M = d_mc["M_gol"]
            st.caption(f"Distribuzione: {d_mc['fonte']}.")
            rng = np.random.default_rng()
            scelta = rng.choice(M.size, size=iterazioni, p=M.ravel())
            gc_sim, gt_sim = scelta // M.shape[1], scelta % M.shape[1]
            pc = 100.0 / iterazioni
            v_c = np.sum(gc_sim > gt_sim) * pc
            v_p = np.sum(gc_sim == gt_sim) * pc
            v_t = np.sum(gc_sim < gt_sim) * pc
            tot_gol = gc_sim + gt_sim
            p_over15 = np.sum(tot_gol > 1.5) * pc
            p_over25 = np.sum(tot_gol > 2.5) * pc
            p_btts = np.sum((gc_sim > 0) & (gt_sim > 0)) * pc
            st.markdown("#### 📊 Esiti 1X2")
            m1, m2, m3 = st.columns(3)
            m1.metric("1 (Casa)", f"{v_c:.1f}%", f"Equa: {100/v_c:.2f}" if v_c > 0 else "N.D.")
            m2.metric("X", f"{v_p:.1f}%", f"Equa: {100/v_p:.2f}" if v_p > 0 else "N.D.")
            m3.metric("2 (Trasferta)", f"{v_t:.1f}%", f"Equa: {100/v_t:.2f}" if v_t > 0 else "N.D.")
            st.markdown("#### ⚽ Mercati gol")
            uo1, uo2, uo3 = st.columns(3)
            uo1.metric("Over 1.5", f"{p_over15:.1f}%")
            uo2.metric("Over 2.5", f"{p_over25:.1f}%")
            uo3.metric("Goal (BTTS)", f"{p_btts:.1f}%")
            st.markdown("#### 🎯 Top 3 esatti più frequenti")
            conteggio = pd.Series(list(zip(gc_sim.tolist(), gt_sim.tolist()))).value_counts().head(3)
            for col, medaglia, ((rc, rt), count) in zip(st.columns(3), ["🥇", "🥈", "🥉"], conteggio.items()):
                col.info(f"{medaglia} **{rc} - {rt}** ({count / iterazioni * 100:.1f}%)")
            fig_mc = px.histogram(
                pd.DataFrame({"Gol Totali": tot_gol}), x="Gol Totali",
                nbins=int(tot_gol.max()) + 1,
                title="Distribuzione gol totali (Monte Carlo)", template="plotly_dark",
            )
            st.plotly_chart(fig_mc, use_container_width=True)


# --- BACKTEST ---
with tab_backtest:
    st.subheader("🧪 Backtest walk-forward")
    st.caption(
        "Ogni blocco di partite è previsto da un modello addestrato solo sul passato. "
        "Confronto con frequenze storiche e quote di chiusura senza margine."
    )
    if par["auto"]:
        hl_txt = par["halflife"] if par["halflife"] else "nessun decadimento"
        extra = ""
        if par.get("log_loss_holdout") is not None:
            extra = f" · hold-out log-loss = {par['log_loss_holdout']:.4f}"
        st.caption(f"Parametri auto: λ = {par['ridge']}, emivita = {hl_txt}{extra}. Dixon-Coles attivo.")
        with st.expander("Come sono stati scelti"):
            st.dataframe(pd.DataFrame(par["ricerca"]).sort_values("Log loss (tune)"), use_container_width=True, hide_index=True)
            st.caption(
                "Selezione su walk-forward (tune). Se c'è hold-out, la scelta preferisce "
                "il log-loss sull'ultima stagione non usata per la griglia."
            )

    if st.button("🚀 Esegui Backtest"):
        with st.spinner("Calcolo in corso..."):
            st.session_state["bt"] = backtest_cached(div, par["ridge"], par["halflife"], PASSO_BACKTEST, giorno, chiave_api)
            st.session_state["bt_div"] = div

    bt = st.session_state.get("bt") if st.session_state.get("bt_div") == div else None
    if bt is not None and not bt.empty:
        mt = metriche_backtest(bt)
        st.markdown(f"#### 📏 Qualità probabilità 1X2 ({mt['n']} partite)")
        r1, r2, r3 = st.columns(3)
        r1.metric("Log loss modello", f"{mt['modello'][0]:.4f}", f"Brier {mt['modello'][1]:.4f}", delta_color="off")
        r2.metric("Log loss frequenze", f"{mt['frequenze'][0]:.4f}", f"Brier {mt['frequenze'][1]:.4f}", delta_color="off")
        if "mercato" in mt:
            r3.metric("Log loss mercato (chiusura)", f"{mt['mercato'][0]:.4f}", f"Brier {mt['mercato'][1]:.4f}", delta_color="off")
            gap = mt["modello_su_mercato"][0] - mt["mercato"][0]
            st.caption(
                f"Sulle {mt['n_mercato']} partite con quote di chiusura: modello {mt['modello_su_mercato'][0]:.4f} "
                f"vs mercato {mt['mercato'][0]:.4f} (scarto {gap:+.4f}; più basso = meglio)."
            )
        else:
            r3.metric("Log loss mercato", "N.D.")

        st.markdown("#### 💰 Scommesse simulate (1 u. sull'esito con edge più alto)")
        tb = tabella_scommesse(bt)
        if tb.empty:
            st.info("Quote non disponibili per simulare le scommesse.")
        else:
            st.dataframe(tb, use_container_width=True, hide_index=True)
            st.caption(
                "ROI su poche centinaia di scommesse è rumoroso: guarda ± ROI e CLV. "
                "CLV medio positivo e stabile è segnale più affidabile del ROI."
            )

        st.markdown("#### 🎯 Calibrazione: prevista vs frequenza reale")
        cal = tabella_calibrazione(bt)
        if not cal.empty:
            figc = go.Figure()
            figc.add_trace(go.Scatter(
                x=cal["Prob. prevista %"], y=cal["Frequenza reale %"],
                mode="lines+markers", name="Modello",
                text=cal["Casi"], hovertemplate="Prevista %{x}%<br>Reale %{y}%<br>Casi %{text}",
            ))
            figc.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Perfetta", line=dict(dash="dash")))
            figc.update_layout(template="plotly_dark", xaxis_title="Prob. prevista (%)", yaxis_title="Frequenza reale (%)")
            st.plotly_chart(figc, use_container_width=True)
    elif bt is not None:
        st.warning("Backtest vuoto: servono almeno 2 stagioni.")

    st.markdown("---")
    st.markdown("### 🔬 Test completo: Goal, Over/Under, Risultati esatti")
    st.caption(
        f"Test su {N_STAGIONI_TEST} stagioni (prima = storico iniziale), walk-forward. "
        "Varianti: Modello, Senza DC, Ancorato, Misto, Tiri, Gol+Tiri, Frequenze, "
        "Modello+Frequenze (peso ottimo per mercato). Valori più bassi = meglio."
    )
    st.caption(
        f"Pesi live (senza quote): modello {pesi['gol']*100:.0f}% Goal/Over, "
        f"{pesi['esatto']*100:.0f}% esatti"
        + (f" (su {pesi['n']} partite)." if pesi.get("auto") else " (riserva: dati insufficienti).")
    )

    if st.button("🧪 Esegui test completo"):
        with st.spinner("Calcolo in corso (può richiedere 1–2 minuti)..."):
            st.session_state["tc"] = test_completo_cached(div, par["ridge"], par["halflife"], PASSO_BACKTEST, giorno, VERSIONE_TEST, chiave_api)
            st.session_state["tc_div"] = div

    tc = st.session_state.get("tc") if st.session_state.get("tc_div") == div else None
    if tc is not None and "tiri_ok" not in tc:
        tc = None
    if tc is not None:
        tab_a, tab_b, tab_c, tab_d, tab_e, best_ex = tabelle_test(tc)
        st.markdown("#### 📏 Qualità per mercato (log loss)")
        st.dataframe(tab_a, use_container_width=True, hide_index=True)
        st.caption(
            "Risultato esatto = log score del punteggio reale. 'Mercato (chiusura)' solo per 1X2 e O/U 2.5. "
            "Colonne ± = differenza media per partita e SE: se |diff| < ~2×SE non è distinguibile dal caso."
        )
        st.markdown("#### ✅ Presi / non presi (anche con calibrazione Platt)")
        st.dataframe(tab_b, use_container_width=True, hide_index=True)
        st.caption(
            "Se 'Presi %' << 'Prob. dichiarata %' il modello è troppo ottimista. "
            "Le righe '(Platt)' usano calibrazione logistica fit sulla prima metà del campione."
        )
        st.markdown("#### 🎯 Risultati esatti")
        st.dataframe(tab_c, use_container_width=True, hide_index=True)
        st.markdown(f"Ultime 25 partite · variante migliore esatti ({best_ex}):")
        st.dataframe(tab_d, use_container_width=True, hide_index=True)
        if not tab_e.empty:
            st.markdown("#### 🆕 Squadre con poca storia")
            st.dataframe(tab_e, use_container_width=True, hide_index=True)
            st.caption("Stesse partite, tre fonti: se anche il mercato sbaglia di più, non è solo un difetto del modello.")


# --- VALUE BET (KELLY) ---
with tab_kelly:
    st.subheader("🧮 Calcolatore Value Bet & Kelly")
    c1, c2, c3 = st.columns(3)
    p_stim = c1.slider("Probabilità modello (%)", 1.0, 100.0, 45.0)
    q_book = c2.number_input("Quota bookmaker", 1.01, 50.0, 2.30)
    bankroll = c3.number_input("Bankroll (€)", 10.0, 100_000.0, 1000.0)
    q_equa = 100 / p_stim
    ev = (p_stim / 100) * q_book - 1
    b, p = q_book - 1, p_stim / 100
    kelly_full = max(0.0, (b * p - (1 - p)) / b)
    kelly_frac = kelly_full * 0.25
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Quota equa", f"{q_equa:.2f}")
    m2.metric("EV", f"{ev*100:+.2f}%")
    m3.metric("Kelly pieno", f"{kelly_full*100:.1f}%")
    m4.metric("Stake (¼ Kelly)", f"€{bankroll * kelly_frac:.2f}")
    if q_book > q_equa:
        st.success("Edge positivo secondo la probabilità inserita (dipende dalla calibrazione del modello).")
    else:
        st.error("❌ Nessun valore rilevato.")
    st.caption("¼ Kelly riduce la varianza; full Kelly è aggressivo e spesso non consigliabile con probabilità stimate.")
