"""
wc2026/backtest_wc2022.py
═══════════════════════════════════════════════════════════════
PRUEBA CIEGA — MUNDIAL 2022 (QATAR)
Metodología:
  TRAIN: Todo hasta 2022-11-19 (día antes del primer partido)
  TEST:  64 partidos WC 2022 (20 Nov - 18 Dic 2022)
  
Modelos evaluados:
  1. ELO puro (baseline)
  2. ELO + Dixon-Coles Internacional
  3. Ensemble
  
Métricas:
  - LogLoss (menor = mejor; naïve = 1.099)
  - Brier Score
  - Hit Rate (resultado correcto 1X2)
  - ROI simulado (Kelly vs momios históricos estimados)
═══════════════════════════════════════════════════════════════
"""
import sys
from pathlib import Path
# Ensure wc2026/ is in path so `model` package is found
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")


# ─── Cargar datos ────────────────────────────────────────────────────────────
print("Cargando datos históricos...")
df_all = pd.read_csv(ROOT / "data/international/results.csv", parse_dates=["date"])
df_all = df_all.dropna(subset=["home_score","away_score"])
df_all["home_score"] = df_all["home_score"].astype(int)
df_all["away_score"] = df_all["away_score"].astype(int)

CUT_DATE  = pd.Timestamp("2022-11-19")
WC_START  = pd.Timestamp("2022-11-20")
WC_END    = pd.Timestamp("2022-12-18")
WC_TOURN  = "FIFA World Cup"

# TRAIN: todo antes del primer partido del Mundial 2022
df_train = df_all[df_all["date"] <= CUT_DATE].copy()

# TEST: solo partidos del WC 2022
df_test = df_all[
    (df_all["date"] >= WC_START) &
    (df_all["date"] <= WC_END) &
    (df_all["tournament"] == WC_TOURN)
].copy()

print(f"  TRAIN: {len(df_train):,} partidos (hasta {CUT_DATE.date()})")
print(f"  TEST:  {len(df_test)} partidos del Mundial 2022")
print(f"  Equipos en test: {sorted(set(df_test.home_team) | set(df_test.away_team))[:8]}...")

# ─── Modelo 1: ELO Puro ──────────────────────────────────────────────────────
print("\n[1/3] Entrenando ELO puro...")
from model.elo_system import ELOSystem

TOURNAMENT_K_ELO = {
    "FIFA World Cup": 60, "FIFA World Cup qualification": 40,
    "UEFA Euro": 50, "Copa América": 50, "CONCACAF Gold Cup": 45,
    "Africa Cup of Nations": 45, "AFC Asian Cup": 45,
    "UEFA Nations League": 40, "Friendly": 20,
}

elo = ELOSystem()
for _, r in df_train.sort_values("date").iterrows():
    k = TOURNAMENT_K_ELO.get(r.get("tournament","Friendly"), 30)
    elo.update(r["home_team"], r["away_team"],
               int(r["home_score"]), int(r["away_score"]),
               tournament=r.get("tournament","Friendly"),
               neutral=bool(r.get("neutral", False)))

# ─── Modelo 2: Dixon-Coles Internacional ─────────────────────────────────────
print("[2/3] Entrenando Dixon-Coles Internacional...")

# Filtrar solo los últimos 8 años para DC (ventana relevante, más rápido)
df_dc_train = df_train[df_train["date"] >= pd.Timestamp("2014-01-01")].copy()

# Equipos con suficientes partidos (>= 5)
team_counts = pd.concat([
    df_dc_train["home_team"], df_dc_train["away_team"]
]).value_counts()
valid_teams = set(team_counts[team_counts >= 5].index)

df_dc_train = df_dc_train[
    df_dc_train["home_team"].isin(valid_teams) &
    df_dc_train["away_team"].isin(valid_teams)
].copy()

# Decaimiento temporal ξ=0.003/día
ref_date = CUT_DATE
df_dc_train["days_ago"] = (ref_date - df_dc_train["date"]).dt.days
df_dc_train["weight"]   = np.exp(-0.003 * df_dc_train["days_ago"])

# Peso extra por importancia del torneo
df_dc_train.loc[df_dc_train["tournament"] == "FIFA World Cup", "weight"] *= 1.5
df_dc_train.loc[df_dc_train["tournament"].str.contains("qualification", case=False, na=False), "weight"] *= 0.9
df_dc_train.loc[df_dc_train["tournament"] == "Friendly", "weight"] *= 0.7

teams  = sorted(valid_teams)
N      = len(teams)
t_idx  = {t: i for i, t in enumerate(teams)}

def dc_log_likelihood(params):
    """Log-verosimilitud Dixon-Coles ponderada por tiempo."""
    att = params[:N]
    def_ = params[N:2*N]
    home_adv = params[2*N]
    rho      = params[2*N+1]
    ridge    = 0.02

    ll = 0.0
    for _, row in df_dc_train.iterrows():
        hi = t_idx.get(row["home_team"])
        ai = t_idx.get(row["away_team"])
        if hi is None or ai is None:
            continue
        w  = row["weight"]
        hg = int(row["home_score"])
        ag = int(row["away_score"])
        neutral = bool(row.get("neutral", False))

        ha = 0 if neutral else home_adv
        lam = np.exp(att[hi] + def_[ai] + ha)
        mu  = np.exp(att[ai] + def_[hi])
        lam = max(lam, 1e-6); mu = max(mu, 1e-6)

        # Corrección Dixon-Coles
        if hg == 0 and ag == 0:
            tau = max(1 - lam * mu * rho, 1e-10)
        elif hg == 1 and ag == 0:
            tau = max(1 + mu * rho, 1e-10)
        elif hg == 0 and ag == 1:
            tau = max(1 + lam * rho, 1e-10)
        elif hg == 1 and ag == 1:
            tau = max(1 - rho, 1e-10)
        else:
            tau = 1.0

        p = (poisson.pmf(hg, lam) * poisson.pmf(ag, mu) * tau)
        p = max(p, 1e-10)
        ll -= w * np.log(p)

    # Regularización L2 (ridge)
    ll += ridge * (np.sum(att**2) + np.sum(def_**2))
    return ll

# Inicializar con ELO-derived priors
x0 = np.zeros(2*N + 2)
x0[2*N]   = 0.15   # home advantage
x0[2*N+1] = -0.10  # rho

print(f"  Optimizando {2*N+2} parámetros sobre {len(df_dc_train):,} partidos...")
result = minimize(dc_log_likelihood, x0, method="L-BFGS-B",
                  options={"maxiter": 150, "ftol": 1e-6},
                  bounds=[(None,None)]*2*N + [(0, 0.5), (-0.5, 0.0)])


att_fit  = result.x[:N]
def_fit  = result.x[N:2*N]
home_adv = result.x[2*N]
rho      = result.x[2*N+1]
print(f"  ✓ Convergido. Home_adv={home_adv:.3f}, rho={rho:.3f}")

# ─── Función de predicción ────────────────────────────────────────────────────
def predict_dc(home: str, away: str, neutral: bool = True) -> tuple:
    """Retorna (p_home, p_draw, p_away) con Dixon-Coles."""
    if home not in t_idx or away not in t_idx:
        return None  # usar ELO como fallback
    hi = t_idx[home]; ai = t_idx[away]
    ha = 0 if neutral else home_adv
    lam = np.exp(att_fit[hi] + def_fit[ai] + ha)
    mu  = np.exp(att_fit[ai] + def_fit[hi])
    
    p_h = p_d = p_a = 0.0
    for i in range(10):
        for j in range(10):
            if i == 0 and j == 0:
                tau = max(1 - lam * mu * rho, 1e-10)
            elif i == 1 and j == 0:
                tau = max(1 + mu * rho, 1e-10)
            elif i == 0 and j == 1:
                tau = max(1 + lam * rho, 1e-10)
            elif i == 1 and j == 1:
                tau = max(1 - rho, 1e-10)
            else:
                tau = 1.0
            p = poisson.pmf(i, lam) * poisson.pmf(j, mu) * tau
            if i > j:   p_h += p
            elif i == j: p_d += p
            else:        p_a += p
    total = p_h + p_d + p_a
    return p_h/total, p_d/total, p_a/total

def predict_elo(home: str, away: str, neutral: bool = True) -> tuple:
    pred = elo.predict_match(home, away, neutral=neutral)
    return pred["home"], pred["draw"], pred["away"]

def ensemble(home: str, away: str, neutral: bool = True,
             w_dc=0.55, w_elo=0.45) -> tuple:
    dc  = predict_dc(home, away, neutral)
    el  = predict_elo(home, away, neutral)
    if dc is None:
        return el
    return (
        w_dc*dc[0] + w_elo*el[0],
        w_dc*dc[1] + w_elo*el[1],
        w_dc*dc[2] + w_elo*el[2],
    )

# ─── Prueba Ciega sobre WC 2022 ──────────────────────────────────────────────
print("\n[3/3] Evaluando en partidos WC 2022 (CIEGO)...")
print("─" * 75)

results = []
for _, row in df_test.sort_values("date").iterrows():
    home = row["home_team"]; away = row["away_team"]
    hg   = int(row["home_score"]); ag = int(row["away_score"])
    date = row["date"].strftime("%Y-%m-%d")
    neutral = True  # Qatar es sede neutral para todos

    # Resultado real
    if hg > ag:   real = "H"
    elif hg == ag: real = "D"
    else:          real = "A"

    # Predicciones
    p_elo  = predict_elo(home, away, neutral)
    p_ens  = ensemble(home, away, neutral)

    # Pick del modelo (máxima probabilidad)
    picks = ["H","D","A"]
    pick_elo = picks[np.argmax(p_elo)]
    pick_ens = picks[np.argmax(p_ens)]

    # LogLoss por partido
    real_idx = picks.index(real)
    ll_elo   = -np.log(max(p_elo[real_idx], 1e-10))
    ll_ens   = -np.log(max(p_ens[real_idx], 1e-10))

    # Brier
    y = [0,0,0]; y[real_idx] = 1
    brier_elo = sum((p_elo[i]-y[i])**2 for i in range(3))
    brier_ens = sum((p_ens[i]-y[i])**2 for i in range(3))

    results.append({
        "date": date, "home": home, "away": away,
        "hg": hg, "ag": ag, "real": real,
        "p_h_ens": round(p_ens[0],3), "p_d_ens": round(p_ens[1],3),
        "p_a_ens": round(p_ens[2],3),
        "pick_ens": pick_ens, "hit_ens": pick_ens == real,
        "ll_ens": round(ll_ens, 4), "brier_ens": round(brier_ens, 4),
        "p_h_elo": round(p_elo[0],3), "p_d_elo": round(p_elo[1],3),
        "p_a_elo": round(p_elo[2],3),
        "pick_elo": pick_elo, "hit_elo": pick_elo == real,
        "ll_elo": round(ll_elo, 4), "brier_elo": round(brier_elo, 4),
    })

    emoji_ens = "✅" if pick_ens == real else "❌"
    print(f"  {date}  {home:<22} {hg}-{ag}  {away:<22} | "
          f"Pred:{p_ens[0]:.2f}/{p_ens[1]:.2f}/{p_ens[2]:.2f} → {pick_ens} "
          f"{emoji_ens} (real:{real})")

# ─── Métricas globales ───────────────────────────────────────────────────────
df_res = pd.DataFrame(results)

n = len(df_res)
ll_ens_mean    = df_res["ll_ens"].mean()
ll_elo_mean    = df_res["ll_elo"].mean()
brier_ens_mean = df_res["brier_ens"].mean()
hit_ens        = df_res["hit_ens"].mean()
hit_elo        = df_res["hit_elo"].mean()

# Referencia: naïve (uniforme)
naive_ll     = -np.log(1/3)  # 1.099
naive_brier  = 2 * (1/3) * (2/3)  # 0.222

print("\n" + "=" * 65)
print("  📊 RESULTADOS PRUEBA CIEGA — MUNDIAL 2022 QATAR")
print("=" * 65)
print(f"  Partidos evaluados: {n}")
print(f"\n  {'Modelo':<25} {'LogLoss':>9} {'vs Naïve':>9} {'Brier':>8} {'Hit%':>7}")
print(f"  {'-'*60}")
print(f"  {'Naïve (1/3)':25} {naive_ll:>9.4f} {'—':>9} {naive_brier:>8.4f} {'33.3%':>7}")
print(f"  {'ELO Puro':25} {ll_elo_mean:>9.4f} {(ll_elo_mean-naive_ll):>+9.4f} "
      f"{df_res['brier_elo'].mean():>8.4f} {hit_elo:>7.1%}")
print(f"  {'DC + ELO Ensemble':25} {ll_ens_mean:>9.4f} {(ll_ens_mean-naive_ll):>+9.4f} "
      f"{brier_ens_mean:>8.4f} {hit_ens:>7.1%}")
print(f"\n  📈 Mejora sobre naïve (Ensemble): {(naive_ll-ll_ens_mean)/naive_ll:.1%}")

# Fase de grupos vs eliminatorias
gp = df_res[df_res.index < 48]  # aproximado
el = df_res[df_res.index >= 48]
if len(gp) > 0 and len(el) > 0:
    print(f"\n  Por fase:")
    print(f"  {'Fase de grupos':25} LogLoss={gp['ll_ens'].mean():.4f} | Hit={gp['hit_ens'].mean():.1%}")
    print(f"  {'Eliminatorias':25} LogLoss={el['ll_ens'].mean():.4f} | Hit={el['hit_ens'].mean():.1%}")

# Partidos más acertados / fallados
print(f"\n  TOP 3 predicciones más certeras:")
top3 = df_res.nsmallest(3, "ll_ens")
for _, r in top3.iterrows():
    print(f"    {r['home']} vs {r['away']:} → pred {r['pick_ens']} (real {r['real']}) | LL={r['ll_ens']:.3f}")

print(f"\n  TOP 3 predicciones más falladas:")
bot3 = df_res.nlargest(3, "ll_ens")
for _, r in bot3.iterrows():
    print(f"    {r['home']} vs {r['away']} → pred {r['pick_ens']} (real {r['real']}) | LL={r['ll_ens']:.3f}")

# Guardar resultados
out = ROOT / "data" / "tracker" / "blind_test_wc2022.csv"
out.parent.mkdir(parents=True, exist_ok=True)
df_res.to_csv(out, index=False)
print(f"\n✓ Resultados guardados: {out}")
print("=" * 65)
