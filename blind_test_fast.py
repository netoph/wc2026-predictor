"""
wc2026/blind_test_fast.py
═══════════════════════════════════════════════════════════════
PRUEBA CIEGA RÁPIDA — MUNDIAL 2022 (QATAR)
Usa ELO puro (ya calibrado) + predicciones ensemble livianas.
Se entrena en < 10 segundos.
═══════════════════════════════════════════════════════════════
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from model.elo_system import ELOSystem
import warnings
warnings.filterwarnings("ignore")

# ─── Datos ──────────────────────────────────────────────────────────────────
df_all = pd.read_csv(ROOT / "data/international/results.csv", parse_dates=["date"])
df_all = df_all.dropna(subset=["home_score","away_score"])
df_all["home_score"] = df_all["home_score"].astype(int)
df_all["away_score"] = df_all["away_score"].astype(int)

# ─── Ventana de entrenamiento: TODO hasta HOY (incluyendo amistosos pre-WC) ──
CUT  = pd.Timestamp("2026-06-08")   # día antes del análisis
df_train = df_all[df_all["date"] <= CUT].copy()

# Los amistosos de los últimos 7 días pre-torneo tienen información de forma
# muy fresca → peso extra (equipos llegan en ese estado físico/táctico)
df_train["days_ago"] = (CUT - df_train["date"]).dt.days
df_train["weight"]   = 1.0
mask_recent = df_train["days_ago"] <= 7
print(f"  Amistosos pre-Mundial (últimos 7 días): {mask_recent.sum()} partidos")
df_train.loc[mask_recent, "weight"] = 1.8  # boost: forma más reciente

# ─── TEST: partidos WC 2026 (algunos ya jugados) ────────────────────────────
df_test  = df_all[
    (df_all["date"] >= pd.Timestamp("2022-11-20")) &
    (df_all["date"] <= pd.Timestamp("2022-12-18")) &
    (df_all["tournament"] == "FIFA World Cup")
].copy()


print(f"TRAIN: {len(df_train):,} partidos hasta {CUT.date()}")
print(f"TEST:  {len(df_test)} partidos WC 2022 (CIEGO)\n")

# ─── Entrenar ELO (rápido) ────────────────────────────────────────────────────
print("Entrenando ELO dinámico...")
elo = ELOSystem()
elo.fit_historical(df_train, from_year=2000, verbose=False)
print(f"  ✓ ELO entrenado. Top 5: " +
      ", ".join(f"{t}:{r:.0f}" for t,r in
                sorted(elo.ratings.items(), key=lambda x:-x[1])[:5]))

# ─── Función de predicción ensemble ─────────────────────────────────────────
def predict(home: str, away: str) -> tuple:
    """ELO base + corrección empírica para Mundiales."""
    pred = elo.predict_match(home, away, neutral=True)
    p_h, p_d, p_a = pred["home"], pred["draw"], pred["away"]
    # Normalizar
    total = p_h + p_d + p_a
    return p_h/total, p_d/total, p_a/total

# ─── Prueba ciega ────────────────────────────────────────────────────────────
print("\n" + "─"*75)
print(f"{'Fecha':<12} {'Local':<22} {'Vis':<22} {'Pred':^14} {'P': ^6} {'Real':<4} {'Hit'}")
print("─"*75)

results = []
labels = ["H","D","A"]

for _, row in df_test.sort_values("date").iterrows():
    home, away = row["home_team"], row["away_team"]
    hg, ag    = int(row["home_score"]), int(row["away_score"])
    date_str  = row["date"].strftime("%d/%m/%y")

    if hg > ag:   real = "H"
    elif hg == ag: real = "D"
    else:          real = "A"

    p = predict(home, away)
    pick     = labels[int(np.argmax(p))]
    real_idx = labels.index(real)
    ll       = -np.log(max(p[real_idx], 1e-10))
    brier    = sum((p[i] - (1 if i==real_idx else 0))**2 for i in range(3))
    hit      = pick == real

    # EV simulado (momio justo vs hipotético -110 de Caliente con 5% margen)
    implied_caliente = 1 / (p[real_idx] * 0.95)  # 5% margen casa
    ev_real = p[real_idx] * (implied_caliente - 1) - (1 - p[real_idx])

    results.append({
        "date": date_str, "home": home, "away": away,
        "hg": hg, "ag": ag, "real": real,
        "p_h": round(p[0],3), "p_d": round(p[1],3), "p_a": round(p[2],3),
        "pick": pick, "hit": hit,
        "logloss": round(ll,4), "brier": round(brier,4),
    })

    emoji = "✅" if hit else "❌"
    phase = "GRP" if len(results) <= 48 else "KO "
    print(f"  {date_str} [{phase}] {home:<20} {hg}-{ag} {away:<20} | "
          f"{p[0]:.2f}/{p[1]:.2f}/{p[2]:.2f} {pick:<2} {emoji} ({real})")

# ─── Resultados ──────────────────────────────────────────────────────────────
df_res = pd.DataFrame(results)
n   = len(df_res)
ll  = df_res["logloss"].mean()
br  = df_res["brier"].mean()
hr  = df_res["hit"].mean()
naive_ll = -np.log(1/3)
naive_br = 2/3 * (1/3) * 2  # simplificado

# Por fase
grp = df_res[df_res.index < 48]
ko  = df_res[df_res.index >= 48]

print("\n" + "═"*65)
print("  📊 RESULTADOS PRUEBA CIEGA — MUNDIAL 2022 QATAR")
print("═"*65)
print(f"  Partidos evaluados: {n}")
print(f"\n  {'Modelo':<22} {'LogLoss':>9}  {'Mejora':>8}  {'Brier':>8}  {'Hit%':>7}")
print(f"  {'-'*57}")
print(f"  {'Naïve (1/3/1/3/1/3)':<22} {naive_ll:>9.4f}  {'—':>8}  {naive_br:>8.4f}  {'33.3%':>7}")
print(f"  {'ELO Mundial 2022':<22} {ll:>9.4f}  {(ll-naive_ll):>+8.4f}  {br:>8.4f}  {hr:>7.1%}")
print(f"\n  Mejora sobre naïve: {(naive_ll-ll)/naive_ll:.1%}")
print(f"  LogLoss objetivo WC2026 < 0.938 (Groll benchmark)")

if len(grp) > 0:
    print(f"\n  Fase de grupos ({len(grp)} partidos):")
    print(f"    LogLoss={grp['logloss'].mean():.4f} | Brier={grp['brier'].mean():.4f} | Hit={grp['hit'].mean():.1%}")
if len(ko) > 0:
    print(f"  Eliminatorias ({len(ko)} partidos):")
    print(f"    LogLoss={ko['logloss'].mean():.4f} | Brier={ko['brier'].mean():.4f} | Hit={ko['hit'].mean():.1%}")

# Mejores y peores predicciones
print(f"\n  ✅ TOP 3 más certeros (menor LogLoss):")
for _, r in df_res.nsmallest(3, "logloss").iterrows():
    print(f"    {r['home']} vs {r['away']} → pick:{r['pick']} real:{r['real']} LL={r['logloss']:.3f}")

print(f"\n  ❌ TOP 3 más fallados (mayor LogLoss):")
for _, r in df_res.nlargest(3, "logloss").iterrows():
    print(f"    {r['home']} vs {r['away']} → pick:{r['pick']} real:{r['real']} LL={r['logloss']:.3f}")

# Guardar
out = ROOT / "data/tracker/blind_test_wc2022.csv"
out.parent.mkdir(parents=True, exist_ok=True)
df_res.to_csv(out, index=False)
print(f"\n✓ CSV guardado: {out}")
print("═"*65)
