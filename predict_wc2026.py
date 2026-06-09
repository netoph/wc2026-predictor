"""
wc2026/predict_wc2026.py
════════════════════════════════════════════════════════════════
PREDICTOR MUNDIAL 2026 — Modelo completo
Datos: 49,445 partidos históricos + amistosos pre-torneo Jun 2026
Pesos: decay temporal ξ=0.003 + boost x1.8 en últimos 7 días
════════════════════════════════════════════════════════════════
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from model.elo_system import ELOSystem
import warnings; warnings.filterwarnings("ignore")

# ─── Carga y preprocesamiento ────────────────────────────────────────────────
df = pd.read_csv(ROOT / "data/international/results.csv", parse_dates=["date"])
df = df.dropna(subset=["home_score","away_score"])
df["home_score"] = df["home_score"].astype(int)
df["away_score"] = df["away_score"].astype(int)

TODAY = pd.Timestamp("2026-06-08")
df_train = df[df["date"] <= TODAY].copy()
df_train["days_ago"] = (TODAY - df_train["date"]).dt.days

# Pesos: decay exponencial + boost reciente
df_train["weight"] = np.exp(-0.003 * df_train["days_ago"])
df_train.loc[df_train["days_ago"] <= 7,  "weight"] *= 1.8  # amistosos pre-WC
df_train.loc[df_train["days_ago"] <= 30, "weight"] *= 1.2  # último mes
df_train.loc[df_train["tournament"] == "FIFA World Cup", "weight"] *= 1.5
df_train.loc[df_train["tournament"].str.contains("qualification", case=False, na=False), "weight"] *= 0.9
df_train.loc[df_train["tournament"] == "Friendly", "weight"] *= 0.75

print(f"{'='*60}")
print(f"  🏆 PREDICTOR MUNDIAL 2026")
print(f"  Datos: {len(df_train):,} partidos hasta {TODAY.date()}")
amistosos_pre = df_train[df_train['days_ago'] <= 7]
print(f"  Amistosos pre-torneo (últimos 7 días): {len(amistosos_pre)}")
print(f"{'='*60}\n")

# ─── Entrenar ELO con todos los datos ────────────────────────────────────────
print("Entrenando ELO dinámico...")
elo = ELOSystem()
elo.fit_historical(df_train, from_year=1990, verbose=False)
print(f"  ✓ ELO actualizado al {TODAY.date()}")
print(f"  Top 10:")
top10 = sorted(elo.ratings.items(), key=lambda x:-x[1])[:10]
for i, (t, r) in enumerate(top10, 1):
    # Forma reciente (últimos 5 partidos)
    rec = df_train[(df_train.home_team==t)|(df_train.away_team==t)].nlargest(5,'date')
    pts = 0
    for _, row in rec.iterrows():
        hg, ag = int(row.home_score), int(row.away_score)
        is_home = row.home_team == t
        gf = hg if is_home else ag
        gc = ag if is_home else hg
        pts += 3 if gf > gc else (1 if gf == gc else 0)
    print(f"    {i:>2}. {t:<22} ELO:{r:.0f}  Forma:{pts}/15")

# ─── Últimos amistosos pre-Mundial ──────────────────────────────────────────
print(f"\n📋 AMISTOSOS ÚLTIMOS 7 DÍAS (info de forma fresca):")
print(f"  {'Fecha':<10} {'Local':<22} {'Marc':<6} {'Visitante':<22} {'Torneo'}")
print(f"  {'-'*78}")
for _, r in amistosos_pre.sort_values('date', ascending=False).head(40).iterrows():
    print(f"  {str(r.date.date()):<10} {r.home_team:<22} "
          f"{int(r.home_score)}-{int(r.away_score):<4} {r.away_team:<22} {r.tournament}")

# ─── Función de predicción ───────────────────────────────────────────────────
def ev(p_model, momio_american):
    dec = momio_american/100+1 if momio_american>0 else 100/abs(momio_american)+1
    return round((p_model*(dec-1)-(1-p_model))*100,1)

def predict(home, away, neutral=True, altitude_h=0):
    """Retorna prob calibradas + goles esperados."""
    p = elo.predict_match(home, away, neutral=neutral, altitude=altitude_h)
    ph, pd_, pa = p["home"], p["draw"], p["away"]

    # Goles esperados desde ELO diferencial
    elo_diff = p["elo_diff"]
    lam_base = 1.20 + 0.0003 * elo_diff   # local
    mu_base  = 1.10 - 0.0003 * elo_diff   # visitante
    lam_base = max(0.3, lam_base)
    mu_base  = max(0.3, mu_base)

    from scipy.stats import poisson
    lambda_total = lam_base + mu_base
    p_over25 = 1 - sum(poisson.pmf(k, lambda_total) for k in range(3))
    p_btts   = (1 - poisson.pmf(0, lam_base)) * (1 - poisson.pmf(0, mu_base))

    return {
        "p_home": ph, "p_draw": pd_, "p_away": pa,
        "lambda": round(lam_base, 2), "mu": round(mu_base, 2),
        "p_over25": round(p_over25, 3), "p_btts": round(p_btts, 3),
        "elo_h": p["elo_home"], "elo_a": p["elo_away"],
    }

# ─── FASE DE GRUPOS — Primeros partidos (Jun 11-18) ─────────────────────────
FIXTURES = [
    # Día 1 — 11 junio
    {"date":"2026-06-11","home":"Mexico",        "away":"Ecuador",      "neutral":False,"stadium":"Estadio Azteca","altitude":2240},
    {"date":"2026-06-11","home":"United States", "away":"Honduras",     "neutral":False,"stadium":"Rose Bowl",   "altitude":270},
    # Día 2 — 12 junio
    {"date":"2026-06-12","home":"Canada",        "away":"Uruguay",      "neutral":False,"stadium":"BMO Field",   "altitude":76},
    {"date":"2026-06-12","home":"Germany",       "away":"Saudi Arabia", "neutral":True, "stadium":"MetLife",     "altitude":4},
    # Día 3 — 13 junio
    {"date":"2026-06-13","home":"Spain",         "away":"Ivory Coast",  "neutral":True, "stadium":"SoFi",        "altitude":60},
    {"date":"2026-06-13","home":"Argentina",     "away":"Algeria",      "neutral":True, "stadium":"Gillette",    "altitude":18},
    {"date":"2026-06-13","home":"Japan",         "away":"Senegal",      "neutral":True, "stadium":"Lumen Field", "altitude":8},
    # Día 4 — 14 junio
    {"date":"2026-06-14","home":"Brazil",        "away":"Scotland",     "neutral":True, "stadium":"Allegiant",   "altitude":600},
    {"date":"2026-06-14","home":"France",        "away":"Senegal",      "neutral":True, "stadium":"AT&T",        "altitude":171},
    {"date":"2026-06-14","home":"England",       "away":"Croatia",      "neutral":True, "stadium":"Lincoln",     "altitude":267},
    {"date":"2026-06-14","home":"Portugal",      "away":"DR Congo",     "neutral":True, "stadium":"Arrowhead",   "altitude":304},
    # Más partidos...
    {"date":"2026-06-15","home":"Netherlands",   "away":"Uzbekistan",   "neutral":True, "stadium":"MetLife",     "altitude":4},
    {"date":"2026-06-15","home":"Colombia",      "away":"South Korea",  "neutral":True, "stadium":"NRG",         "altitude":23},
    {"date":"2026-06-15","home":"Morocco",       "away":"Scotland",     "neutral":True, "stadium":"Empower",     "altitude":1029},
    {"date":"2026-06-18","home":"Mexico",        "away":"South Korea",  "neutral":False,"stadium":"Estadio Azteca","altitude":2240},
]

print(f"\n{'='*75}")
print(f"  🗓️  PREDICCIONES FASE DE GRUPOS — MUNDIAL 2026")
print(f"{'='*75}")
print(f"  {'Fecha':<11} {'Local':<22} {'Vis':<22} {'P H/D/A':^18} {'Goles':^9} {'O2.5':>5} {'BTTS':>5}")
print(f"  {'-'*75}")

for m in FIXTURES:
    p = predict(m["home"], m["away"], m["neutral"], m.get("altitude",0))
    alt_info = f" ⛰️{m['altitude']}m" if m.get("altitude",0) > 1000 else ""
    print(f"  {m['date']} {m['home']:<22} {m['away']:<22} "
          f"{p['p_home']:.2f}/{p['p_draw']:.2f}/{p['p_away']:.2f}  "
          f"{p['lambda']:.1f}+{p['mu']:.1f}={p['lambda']+p['mu']:.1f}  "
          f"{p['p_over25']:>5.1%}  {p['p_btts']:>5.1%}{alt_info}")

# ─── PARTIDOS DE ALTO INTERÉS CON EV ────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  💰 VALUE BETS — FASE DE GRUPOS (momios Caliente estimados)")
print(f"{'='*65}")

HIGH_INTEREST = [
    {"match":"Mexico vs Ecuador","home":"Mexico","away":"Ecuador",
     "neutral":False,"altitude":2240,
     "odds":{"home_win":+110,"draw":+230,"away_win":+260,
             "over25":+105,"btts":-115}},
    {"match":"Argentina vs Algeria","home":"Argentina","away":"Algeria",
     "neutral":True,"altitude":0,
     "odds":{"home_win":-350,"draw":+380,"away_win":+900,
             "over25":-130,"btts":-105}},
    {"match":"Brazil vs Scotland","home":"Brazil","away":"Scotland",
     "neutral":True,"altitude":600,
     "odds":{"home_win":-280,"draw":+320,"away_win":+700,
             "over25":-120,"btts":-110}},
    {"match":"USA vs Honduras","home":"United States","away":"Honduras",
     "neutral":False,"altitude":270,
     "odds":{"home_win":-230,"draw":+320,"away_win":+600,
             "over25":+110,"btts":+110}},
    {"match":"Germany vs Saudi Arabia","home":"Germany","away":"Saudi Arabia",
     "neutral":True,"altitude":0,
     "odds":{"home_win":-400,"draw":+420,"away_win":+1100,
             "over25":-140,"btts":-120}},
]

for m in HIGH_INTEREST:
    p = predict(m["home"], m["away"], m["neutral"], m.get("altitude",0))
    odds = m["odds"]
    print(f"\n  ⚽ {m['match']}")
    print(f"     ELO: {elo.get(m['home']):.0f} vs {elo.get(m['away']):.0f}  |  "
          f"Goles esperados: {p['lambda']:.2f}+{p['mu']:.2f}={p['lambda']+p['mu']:.2f}")
    print(f"  {'Mercado':<20} {'P(modelo)':>10}  {'Momio':>8}  {'EV/100':>8}  Veredicto")
    print(f"  {'-'*60}")

    bets = [
        ("Local gana",   p["p_home"],  odds["home_win"]),
        ("Empate",       p["p_draw"],  odds["draw"]),
        ("Visitante",    p["p_away"],  odds["away_win"]),
        ("Over 2.5",     p["p_over25"],odds["over25"]),
        ("BTTS Sí",      p["p_btts"],  odds["btts"]),
    ]
    for nombre, prob, momio in bets:
        ev_val = ev(prob, momio)
        v = "✅ VALOR" if ev_val > 4 else ("⚠️  neutro" if ev_val > -5 else "❌")
        print(f"  {nombre:<20} {prob*100:>9.1f}%  {momio:>+8}  {ev_val:>+7.1f}   {v}")

print(f"\n{'='*65}")
print("  ⚠️  Nota: momios son estimados. Verificar en Caliente antes de apostar.")
print(f"{'='*65}")
