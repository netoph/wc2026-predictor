"""
wc2026/scrapers/intl_results.py
Descarga el histórico completo de partidos internacionales
desde el repositorio martj42/international-football-results (GitHub).
~50,000 partidos desde 1872.
"""
import sys
from pathlib import Path
import requests
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT   = Path(__file__).resolve().parent.parent
OUT    = ROOT / "data" / "international"
OUT.mkdir(parents=True, exist_ok=True)

# ─── Fuentes de datos ────────────────────────────────────────────────────────
SOURCES = {
    "results": "https://raw.githubusercontent.com/martj42/international-football-results/master/results.csv",
    "goalscorers": "https://raw.githubusercontent.com/martj42/international-football-results/master/goalscorers.csv",
    "shootouts": "https://raw.githubusercontent.com/martj42/international-football-results/master/shootouts.csv",
}

# Torneos que nos interesan (filtramos amistosos de poca relevancia)
RELEVANT_TOURNAMENTS = {
    "FIFA World Cup",
    "FIFA World Cup qualification",
    "UEFA Euro",
    "UEFA European Championship",
    "Copa América",
    "CONCACAF Gold Cup",
    "Africa Cup of Nations",
    "AFC Asian Cup",
    "Friendly",  # Mantenemos amistosos pero con peso menor
    "UEFA Nations League",
    "CONCACAF Nations League",
    "FIFA Confederations Cup",
}

def download_and_clean():
    log.info("Descargando resultados históricos...")
    r = requests.get(SOURCES["results"], timeout=30)
    r.raise_for_status()
    
    df = pd.read_csv(pd.io.common.StringIO(r.text))
    log.info(f"  → {len(df):,} partidos totales descargados")
    
    # Filtrar desde año 2000 para mayor relevancia
    df["date"] = pd.to_datetime(df["date"])
    
    # Mantener todo pero marcar peso
    df["weight"] = 1.0
    df.loc[df["date"].dt.year < 2000, "weight"] = 0.3
    df.loc[df["date"].dt.year < 2010, "weight"] = 0.6
    df.loc[df["tournament"] == "Friendly", "weight"] *= 0.7
    df.loc[df["tournament"].str.contains("qualification", case=False, na=False), "weight"] *= 0.85
    df.loc[df["tournament"].str.contains("World Cup$", case=False, na=False), "weight"] = 1.5
    
    # Renombrar para compatibilidad con Dixon-Coles
    df = df.rename(columns={
        "home_score": "home_goals",
        "away_score": "away_goals",
    })
    
    # Quitar partidos sin resultado
    df = df.dropna(subset=["home_goals", "away_goals"])
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    
    # Añadir confederación de cada equipo
    df = add_confederation(df)
    
    out_path = OUT / "results.csv"
    df.to_csv(out_path, index=False)
    log.info(f"✓ Guardado: {out_path} ({len(df):,} partidos)")
    
    # Estadísticas
    log.info(f"\nDistribución por torneo (top 10):")
    top_t = df.groupby("tournament").size().sort_values(ascending=False).head(10)
    for t, n in top_t.items():
        log.info(f"  {t:<45} {n:>5}")
    
    wc_df = df[df["tournament"] == "FIFA World Cup"]
    log.info(f"\nSolo World Cup: {len(wc_df)} partidos (desde {wc_df['date'].dt.year.min()})")
    
    return df

CONFEDERATION_MAP = {
    # UEFA
    "UEFA": ["Germany", "France", "Spain", "Italy", "England", "Portugal",
             "Netherlands", "Belgium", "Croatia", "Serbia", "Poland", "Switzerland",
             "Austria", "Denmark", "Sweden", "Norway", "Czech Republic", "Slovakia",
             "Hungary", "Romania", "Bulgaria", "Greece", "Turkey", "Ukraine",
             "Russia", "Scotland", "Wales", "Northern Ireland", "Republic of Ireland",
             "Albania", "Bosnia and Herzegovina", "Kosovo", "North Macedonia",
             "Montenegro", "Slovenia", "Finland", "Iceland", "Luxembourg",
             "Armenia", "Azerbaijan", "Georgia", "Kazakhstan", "Moldova"],
    # CONMEBOL
    "CONMEBOL": ["Brazil", "Argentina", "Uruguay", "Colombia", "Chile", "Peru",
                 "Ecuador", "Bolivia", "Paraguay", "Venezuela"],
    # CONCACAF
    "CONCACAF": ["United States", "Mexico", "Canada", "Costa Rica", "Honduras",
                 "Guatemala", "El Salvador", "Jamaica", "Panama", "Trinidad and Tobago",
                 "Curacao", "Haiti", "Cuba", "Nicaragua", "Belize"],
    # CAF
    "CAF": ["Morocco", "Senegal", "Nigeria", "Ghana", "Egypt", "Cameroon",
            "Ivory Coast", "Algeria", "Tunisia", "South Africa", "Mali",
            "Burkina Faso", "Guinea", "DR Congo", "Zambia", "Zimbabwe",
            "Tanzania", "Uganda", "Kenya", "Ethiopia", "Mozambique",
            "Angola", "Benin", "Togo", "Rwanda", "Cape Verde"],
    # AFC
    "AFC": ["Japan", "South Korea", "Iran", "Saudi Arabia", "Australia",
            "Qatar", "UAE", "Jordan", "Iraq", "Oman", "Bahrain",
            "China", "Uzbekistan", "Kyrgyzstan", "Tajikistan", "India",
            "Indonesia", "Vietnam", "Thailand", "Malaysia"],
    # OFC
    "OFC": ["New Zealand", "Fiji", "Papua New Guinea", "Solomon Islands"],
}

TEAM_TO_CONF = {}
for conf, teams in CONFEDERATION_MAP.items():
    for t in teams:
        TEAM_TO_CONF[t] = conf

def add_confederation(df):
    df["home_conf"] = df["home_team"].map(TEAM_TO_CONF).fillna("OTHER")
    df["away_conf"] = df["away_team"].map(TEAM_TO_CONF).fillna("OTHER")
    return df

if __name__ == "__main__":
    df = download_and_clean()
    print(f"\n✓ Dataset listo: {len(df)} partidos en {OUT/'results.csv'}")
