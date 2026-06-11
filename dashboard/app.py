"""
wc2026/dashboard/app.py
FastAPI backend del dashboard Mundial 2026.
Sirve predicciones en tiempo real y el simulador de apuestas.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import json
import asyncio
import uvicorn
import warnings; warnings.filterwarnings("ignore")

app = FastAPI(title="WC2026 Predictor", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ─── Cargar modelos al startup ────────────────────────────────────────────────
from model.elo_system import ELOSystem
from betting.kelly import KellyBettor
from betting.auto_scanner import AutoEVScanner
from betting.hit_tracker import HitTracker
from scrapers.espn_live import fetch_and_update as espn_fetch

df_hist = None
elo     = None
bettor  = KellyBettor(initial_bankroll=1000.0)
hit_tracker = HitTracker()
ev_scanner  = None   # se inicializa en startup con compute_prediction

# Estado del scraper
scraper_state = {
    "last_fetch": None,
    "last_result": {},
    "is_running": False,
    "fetch_count": 0,
    "live_matches": 0,
    "live_odds": {},   # home_vs_away → odds dict
}

async def espn_background_task():
    """Background task: scrapea ESPN cada 60s durante partidos, 5min si no."""
    while True:
        try:
            scraper_state["is_running"] = True
            result = espn_fetch()
            scraper_state["last_fetch"] = pd.Timestamp.now().isoformat()
            scraper_state["last_result"] = {
                "updated": result.get("updated", 0),
                "fetched": result.get("total_fetched", 0),
                "odds": result.get("odds_logged", 0),
            }
            scraper_state["fetch_count"] += 1

            # Guardar odds en memoria para el dashboard
            live_count = 0
            for m in result.get("matches", []):
                key = f"{m['home']} vs {m['away']}"
                if m.get("odds"):
                    scraper_state["live_odds"][key] = m["odds"]
                if m["status"] == "live":
                    live_count += 1

                # Auto-hit-track finished matches
                if m["status"] == "finished" and m["home_goals"] is not None:
                    already = any(r["home"]==m["home"] and r["away"]==m["away"] for r in hit_tracker.records)
                    if not already:
                        pred = compute_prediction(m["home"], m["away"], True, 0)
                        hit_tracker.record_match(
                            m["home"], m["away"],
                            pred["p_home"], pred["p_draw"], pred["p_away"],
                            m["home_goals"], m["away_goals"],
                            p_over25=pred.get("p_over25"),
                            p_btts=pred.get("p_btts"),
                            total_goals_expected=pred.get("total_goals_expected"),
                        )
                        # Auto-settle bets for this match
                        _auto_settle_match(m["home"], m["away"],
                                          m["home_goals"], m["away_goals"])

            scraper_state["live_matches"] = live_count

            # Auto-EV scan when odds available
            if ev_scanner and scraper_state["live_odds"]:
                scan_result = ev_scanner.scan_all(scraper_state["live_odds"])
                if scan_result["placed"] > 0:
                    print(f"  🎯 Auto-bet: {scan_result['placed']} apuestas colocadas")

            # Recargar fixtures si hubo actualizaciones
            if result.get("updated", 0) > 0:
                global fixtures_df
                fixtures_df = None  # Force reload on next request

            scraper_state["is_running"] = False
            # Persist state to disk after each cycle
            _save_state()
            # During WC (Jun 11 - Jul 19), always scrape fast
            from datetime import datetime
            is_wc = datetime(2026,6,11) <= datetime.now() <= datetime(2026,7,20)
            interval = 60 if (live_count > 0 or is_wc) else 300
            print(f"  📡 ESPN: {result.get('total_fetched',0)} partidos | "
                  f"{result.get('updated',0)} actualizados | "
                  f"{result.get('odds_logged',0)} odds | "
                  f"next in {interval}s")
        except Exception as e:
            scraper_state["is_running"] = False
            print(f"  ⚠️  ESPN scraper error: {e}")
            interval = 120

        await asyncio.sleep(interval)

@app.on_event("startup")
async def startup():
    global df_hist, elo
    print("Cargando datos y modelos...")
    df_hist = pd.read_csv(ROOT / "data/international/results.csv", parse_dates=["date"])
    df_hist = df_hist.dropna(subset=["home_score","away_score"])
    elo = ELOSystem()
    elo.fit_historical(df_hist, from_year=2000, verbose=False)
    print(f"✓ ELO entrenado. Top ELO: {sorted(elo.ratings.items(), key=lambda x:-x[1])[:3]}")
    # Arrancar scraper en background
    asyncio.create_task(espn_background_task())
    print("✓ ESPN scraper arrancado (background task)")
    # Entrenar NegBin
    global negbin_model
    from model.negative_binomial import NegBinGoalModel
    negbin_model = NegBinGoalModel(ridge=0.02)
    negbin_model.fit(df_hist, elo_ratings=elo.ratings, from_year=2014, verbose=True)
    print("✓ NegBin entrenado")
    # Calibración
    global calibrator
    from model.calibration import TemperatureScaler
    calibrator = _fit_calibrator(df_hist)
    # Auto EV scanner
    global ev_scanner
    ev_scanner = AutoEVScanner(bettor, compute_prediction)
    print("✓ Auto-EV scanner activado")
    # Restaurar estado previo (si existe)
    _restore_state()
    # Evaluar modelo en train/test/blind
    global eval_cache
    from model.evaluate import evaluate_model
    print("Evaluando modelo en train/test/blind...")
    eval_cache = evaluate_model(elo, negbin_model, df_hist, calibrator)
    for split in ["train","test","blind"]:
        d = eval_cache.get(split, {})
        if "error" not in d:
            nb = d.get("negbin", {})
            print(f"  {split}: n={d.get('n',0):,} | ELO LL={d.get('elo',{}).get('logloss','?')} HR={d.get('elo',{}).get('hit_rate','?')} | NB LL={nb.get('logloss','?')} HR={nb.get('hit_rate','?')}")
    print("✓ Evaluación completa")

@app.on_event("shutdown")
async def shutdown():
    _save_state()
    print("✓ Estado guardado a disco")

negbin_model = None
calibrator   = None
eval_cache   = None

STATE_DIR = ROOT / "data" / "state"

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)

def _save_state():
    """Guarda estado del bot, tracker y scanner a disco."""
    import json
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(STATE_DIR / "bettor.json", "w") as f:
            json.dump({"bets": bettor.bets, "bankroll": bettor.bankroll}, f, cls=NumpyEncoder)
        with open(STATE_DIR / "tracker.json", "w") as f:
            json.dump(hit_tracker.records, f, cls=NumpyEncoder)
        if ev_scanner:
            with open(STATE_DIR / "scanner.json", "w") as f:
                json.dump({"bet_log": ev_scanner.bet_log,
                           "scan_log": ev_scanner.scan_log[-100:],
                           "scanned_matches": list(ev_scanner.scanned_matches)}, f, cls=NumpyEncoder)
    except Exception as e:
        print(f"  ⚠️  Error guardando estado: {e}")

def _restore_state():
    """Restaura estado previo del bot si existe."""
    import json
    if not STATE_DIR.exists():
        print("  ℹ️  Sin estado previo, arrancando limpio")
        return
    try:
        bpath = STATE_DIR / "bettor.json"
        if bpath.exists():
            with open(bpath) as f:
                d = json.load(f)
            bettor.bets = d.get("bets", [])
            bettor.bankroll = d.get("bankroll", 1000.0)
            print(f"  ✓ Restaurado bettor: {len(bettor.bets)} apuestas, bankroll=${bettor.bankroll:.2f}")

        tpath = STATE_DIR / "tracker.json"
        if tpath.exists():
            with open(tpath) as f:
                hit_tracker.records = json.load(f)
            print(f"  ✓ Restaurado tracker: {len(hit_tracker.records)} partidos")

        spath = STATE_DIR / "scanner.json"
        if spath.exists() and ev_scanner:
            with open(spath) as f:
                d = json.load(f)
            ev_scanner.bet_log = d.get("bet_log", [])
            ev_scanner.scan_log = d.get("scan_log", [])
            ev_scanner.scanned_matches = set(d.get("scanned_matches", []))
            print(f"  ✓ Restaurado scanner: {len(ev_scanner.bet_log)} bets, {len(ev_scanner.scanned_matches)} matches escaneados")

        # Sincronizar bet_log de scanner con las apuestas liquidadas en bettor
        if ev_scanner and bettor.bets:
            synced = 0
            for b in bettor.bets:
                if b["result"] is not None:
                    for entry in ev_scanner.bet_log:
                        if entry.get("bet_id") == b["id"] and entry.get("result") is None:
                            entry["result"] = b["result"]
                            entry["pnl"] = b["pnl"]
                            synced += 1
            if synced > 0:
                print(f"  ✓ Sincronizados {synced} resultados de apuestas en scanner")
                _save_state()

    except Exception as e:
        print(f"  ⚠️  Error restaurando estado: {e}")

def _fit_calibrator(df):
    """Ajusta Temperature Scaler con últimos 2 años de resultados."""
    from model.calibration import TemperatureScaler
    recent = df[df["date"] >= pd.Timestamp("2024-01-01")].copy()
    recent = recent.dropna(subset=["home_score","away_score"])
    if len(recent) < 50:
        print("  ⚠️  Calibración: pocos datos, usando T=1.0")
        return TemperatureScaler(T=1.0)
    # Generar predicciones para calibrar
    import numpy as np
    probs = []
    labels = []
    for _, r in recent.iterrows():
        try:
            p = elo.predict_match(r["home_team"], r["away_team"], neutral=True)
            probs.append([p["home"], p["draw"], p["away"]])
            hg, ag = int(r["home_score"]), int(r["away_score"])
            labels.append(0 if hg > ag else (1 if hg == ag else 2))
        except:
            pass
    probs = np.array(probs)
    labels = np.array(labels)
    ts = TemperatureScaler()
    ts.fit(probs, labels)
    print(f"  ✓ Calibración: T={ts.T:.3f} | LogLoss {ts.train_logloss_before:.4f}→{ts.train_logloss_after:.4f}")
    return ts

def _auto_settle_match(home, away, hg, ag):
    """Auto-settle todas las apuestas de un partido finalizado."""
    match_key = f"{home} vs {away}"
    total_goals = hg + ag
    for bet in bettor.bets:
        if bet["result"] is not None:
            continue
        if bet["match"] != match_key:
            continue
        won = False
        sel = bet.get("selection","").lower()
        mkt = bet.get("market","").lower()
        if "1x2" in mkt:
            if sel == "home" and hg > ag: won = True
            elif sel == "draw" and hg == ag: won = True
            elif sel == "away" and hg < ag: won = True
        elif "over" in mkt:
            if total_goals > 2.5: won = True
        elif "under" in mkt:
            if total_goals < 2.5: won = True
        bettor.settle_bet(bet["id"], won)
        # Sync bot log
        if ev_scanner:
            ev_scanner.settle_bot_bet(bet["id"], won)

# ─── Utilidades ───────────────────────────────────────────────────────────────
def compute_prediction(home: str, away: str,
                        neutral: bool = True, altitude: float = 0,
                        home_goals: int = None, away_goals: int = None,
                        minute: int = None):
    """
    Predicción completa: 1X2, O/U, BTTS, córners.
    Usa NegBin si disponible, fallback a ELO lineal.
    Si se pasan (home_goals, away_goals, minute) → modo VIVO.
    """
    from scipy.stats import poisson

    pred_elo = elo.predict_match(home, away, neutral=neutral, altitude=altitude)
    ph, pd_, pa = pred_elo["home"], pred_elo["draw"], pred_elo["away"]
    elo_diff = pred_elo["elo_diff"]

    # NegBin goals (preferred) vs ELO linear (fallback)
    if negbin_model and negbin_model.is_fitted and home in negbin_model.att and away in negbin_model.att:
        lam_full, mu_full = negbin_model.expected_goals(home, away, neutral, altitude)
        model_used = "NegBin"
    else:
        lam_full = max(0.4, 1.18 + 0.00025 * elo_diff + (0.00015 * altitude))
        mu_full  = max(0.4, 1.08 - 0.00025 * elo_diff + (0.00008 * altitude))
        model_used = "ELO"

    if home_goals is not None and minute is not None and minute > 0:
        # Modo VIVO: recalcular con goles restantes
        frac = max((90 - minute) / 90, 0.0)
        # Boost presión según situación global
        global_h = (home_goals or 0)
        global_a = (away_goals or 0)
        pressure_h = 1.0 + 0.12 * (global_a > global_h)   # local va perdiendo
        pressure_a = 1.0 + 0.10 * (global_h > global_a)   # visitante va perdiendo
        lam_rem = lam_full * frac * pressure_h
        mu_rem  = mu_full  * frac * pressure_a

        # Recalcular 1X2 dado marcador actual
        max_g = 6
        joint = np.zeros((max_g, max_g))
        for i in range(max_g):
            for j in range(max_g):
                joint[i,j] = poisson.pmf(i, lam_rem) * poisson.pmf(j, mu_rem)
        joint /= joint.sum()

        final_h = np.array([[home_goals + i for j in range(max_g)] for i in range(max_g)])
        final_a = np.array([[away_goals + j for j in range(max_g)] for i in range(max_g)])
        ph = float(np.sum((final_h > final_a) * joint))
        pd_ = float(np.sum((final_h == final_a) * joint))
        pa  = float(np.sum((final_h < final_a) * joint))
        total = ph + pd_ + pa
        ph /= total; pd_ /= total; pa /= total

        lam_use = lam_rem
        mu_use  = mu_rem
        note = f"VIVO min:{minute} | {home_goals}-{away_goals}"
    else:
        lam_use = lam_full
        mu_use  = mu_full
        note = "Pre-partido"

    lambda_total  = lam_use + mu_use
    p_over15 = 1 - sum(poisson.pmf(k, lambda_total) for k in range(2))
    p_over25 = 1 - sum(poisson.pmf(k, lambda_total) for k in range(3))
    p_over35 = 1 - sum(poisson.pmf(k, lambda_total) for k in range(4))
    p_btts   = (1 - poisson.pmf(0, lam_use)) * (1 - poisson.pmf(0, mu_use))

    lam_corners = 5.2 + 3.1 * (lam_full + mu_full)  # siempre basado en partido completo
    p_corn85 = 1 - sum(poisson.pmf(k, lam_corners) for k in range(9))
    p_corn95 = 1 - sum(poisson.pmf(k, lam_corners) for k in range(10))
    p_corn105= 1 - sum(poisson.pmf(k, lam_corners) for k in range(11))

    return {
        "home": home, "away": away,
        "p_home": round(ph, 4), "p_draw": round(pd_, 4), "p_away": round(pa, 4),
        "lambda": round(lam_full, 3), "mu": round(mu_full, 3),
        "total_goals_expected": round(lam_full + mu_full, 3),
        "p_over15": round(p_over15, 4), "p_over25": round(p_over25, 4),
        "p_over35": round(p_over35, 4),
        "p_under25": round(1 - p_over25, 4),
        "p_btts": round(p_btts, 4),
        "lambda_corners": round(lam_corners, 2),
        "p_corners_over85": round(p_corn85, 4),
        "p_corners_over95": round(p_corn95, 4),
        "p_corners_over105": round(p_corn105, 4),
        "elo_home": round(pred_elo["elo_home"], 0),
        "elo_away": round(pred_elo["elo_away"], 0),
        "elo_diff": round(pred_elo["elo_diff"], 0),
        "note": note,
    }

# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/api/predict")
async def predict(home: str, away: str,
                  neutral: bool = True, altitude: float = 0):
    result = compute_prediction(home, away, neutral, altitude)
    return JSONResponse(result)

@app.post("/api/live_update")
async def live_update(request: Request):
    """Actualiza marcador en vivo y recalcula probabilidades."""
    data = await request.json()
    home     = data.get("home")
    away     = data.get("away")
    hg       = int(data.get("home_goals", 0))
    ag       = int(data.get("away_goals", 0))
    minute   = int(data.get("minute", 45))
    neutral  = bool(data.get("neutral", True))
    altitude = float(data.get("altitude", 0))

    result = compute_prediction(home, away, neutral, altitude, hg, ag, minute)
    return JSONResponse(result)

@app.post("/api/bet")
async def place_bet(request: Request):
    """Coloca apuesta simulada con Kelly Criterion."""
    data    = await request.json()
    match   = data.get("match", "")
    market  = data.get("market", "")
    sel     = data.get("selection", "")
    p_model = float(data.get("p_model", 0))
    momio   = float(data.get("momio", 0))
    date_s  = data.get("date", "")

    bet = bettor.place_bet(match, market, sel, p_model, momio, date_s)
    if bet:
        return JSONResponse({"status": "placed", "bet": bet, "bankroll": bettor.bankroll})
    return JSONResponse({"status": "no_value", "bankroll": bettor.bankroll})

@app.post("/api/settle")
async def settle(request: Request):
    data   = await request.json()
    bet_id = int(data.get("bet_id"))
    won    = bool(data.get("won"))
    pnl    = bettor.settle_bet(bet_id, won)
    return JSONResponse({"pnl": pnl, "bankroll": bettor.bankroll})

@app.get("/api/simulator/stats")
async def sim_stats():
    return JSONResponse(bettor.stats())

@app.get("/api/simulator/bets")
async def sim_bets():
    return JSONResponse(bettor.bets)

@app.get("/api/elo_rankings")
async def elo_rankings():
    top = sorted(elo.ratings.items(), key=lambda x: -x[1])[:48]
    return JSONResponse([{"rank": i+1, "team": t, "elo": round(r)} for i, (t,r) in enumerate(top)])

# ─── Hit Tracker endpoints ───────────────────────────────────────────────────
@app.get("/api/tracker/stats")
async def tracker_stats():
    return JSONResponse(hit_tracker.stats())

@app.get("/api/tracker/records")
async def tracker_records():
    return JSONResponse(hit_tracker.get_records())

# ─── Bot / Auto Scanner endpoints ────────────────────────────────────────────
@app.get("/api/bot/stats")
async def bot_stats():
    """Estadísticas del bot por mercado."""
    if not ev_scanner:
        return JSONResponse({"total_bets": 0, "by_market": {}})
    return JSONResponse(ev_scanner.get_bot_stats())

@app.get("/api/bot/bets")
async def bot_bets():
    """Historial de apuestas del bot."""
    if not ev_scanner:
        return JSONResponse([])
    return JSONResponse(ev_scanner.get_bet_log())

@app.get("/api/bot/scan_log")
async def bot_scan_log():
    """Log de escaneos (incluye skips)."""
    if not ev_scanner:
        return JSONResponse([])
    return JSONResponse(ev_scanner.get_scan_summary())

@app.get("/api/bot/force_scan")
async def force_scan():
    """Forzar escaneo de EV."""
    if not ev_scanner:
        return JSONResponse({"error": "scanner not ready"})
    result = ev_scanner.scan_all(scraper_state.get("live_odds", {}))
    return JSONResponse(result)

# ─── Calibration endpoint ────────────────────────────────────────────────────
@app.get("/api/calibration")
async def calibration_info():
    if not calibrator:
        return JSONResponse({"fitted": False})
    return JSONResponse({
        "fitted": calibrator.fitted,
        "T": round(calibrator.T, 4),
        "logloss_before": round(calibrator.train_logloss_before, 4),
        "logloss_after": round(calibrator.train_logloss_after, 4),
        "brier_before": round(calibrator.train_brier_before, 4),
        "brier_after": round(calibrator.train_brier_after, 4),
        "n_samples": calibrator.n_fit_samples,
    })

@app.get("/api/model/eval")
async def model_eval():
    """KPIs del modelo en train/test/blind."""
    if not eval_cache:
        return JSONResponse({"error": "evaluation not ready"})
    return JSONResponse(eval_cache)

@app.get("/api/scraper/status")
async def scraper_status():
    """Estado del scraper ESPN."""
    return JSONResponse(scraper_state)

@app.get("/api/scraper/odds")
async def live_odds():
    """Momios DraftKings en vivo (via ESPN)."""
    return JSONResponse(scraper_state.get("live_odds", {}))

@app.get("/api/scraper/force")
async def force_scrape():
    """Forzar un scrape inmediato."""
    result = espn_fetch()
    scraper_state["last_fetch"] = pd.Timestamp.now().isoformat()
    scraper_state["fetch_count"] += 1
    for m in result.get("matches", []):
        key = f"{m['home']} vs {m['away']}"
        if m.get("odds"):
            scraper_state["live_odds"][key] = m["odds"]
    return JSONResponse({"status": "ok", **result})

# ─── Fixtures del Mundial ─────────────────────────────────────────────────────
fixtures_df = None

def load_fixtures():
    global fixtures_df
    fixtures_df = pd.read_csv(ROOT / "data/wc2026/fixtures.csv")
    fixtures_df["home_goals"] = fixtures_df["home_goals"].where(fixtures_df["home_goals"].notna(), None)
    fixtures_df["away_goals"] = fixtures_df["away_goals"].where(fixtures_df["away_goals"].notna(), None)
    fixtures_df["minute"]     = fixtures_df["minute"].where(fixtures_df["minute"].notna(), None)

@app.get("/api/fixtures")
async def get_fixtures(group: str = None):
    """Retorna fixtures con predicciones del modelo."""
    if fixtures_df is None:
        load_fixtures()
    df = fixtures_df.copy()
    if group:
        df = df[df["group"] == group]

    results = []
    for _, row in df.iterrows():
        home = row["home"].strip()
        away = row["away"].strip()
        alt  = float(row.get("altitude", 0) or 0)
        status = str(row.get("status", "scheduled")).strip()
        def safe_int(v):
            if v is None or pd.isna(v): return None
            try: return int(float(str(v).strip()))
            except: return None
        hg = safe_int(row.get("home_goals"))
        ag = safe_int(row.get("away_goals"))
        mn = safe_int(row.get("minute"))

        if status == "live" and hg is not None and mn is not None:
            pred = compute_prediction(home, away, True, alt, hg, ag, mn)
        else:
            pred = compute_prediction(home, away, True, alt)

        results.append({
            "date": row["date"], "time": row.get("time",""),
            "group": row["group"],
            "home": home, "away": away,
            "venue": row.get("venue",""), "altitude": alt,
            "status": status,
            "home_goals": hg, "away_goals": ag, "minute": mn,
            **{k: pred[k] for k in [
                "p_home","p_draw","p_away","lambda","mu",
                "total_goals_expected","p_over25","p_btts",
                "lambda_corners","p_corners_over95",
                "elo_home","elo_away","elo_diff"
            ]},
        })
    return JSONResponse(results)

@app.post("/api/fixtures/update")
async def update_fixture(request: Request):
    """Actualiza marcador en vivo de un fixture."""
    if fixtures_df is None:
        load_fixtures()
    data = await request.json()
    home = data.get("home","").strip()
    away = data.get("away","").strip()
    mask = (fixtures_df["home"].str.strip() == home) & (fixtures_df["away"].str.strip() == away)
    if mask.any():
        idx = fixtures_df[mask].index[0]
        fixtures_df.at[idx, "home_goals"] = int(data.get("home_goals", 0))
        fixtures_df.at[idx, "away_goals"] = int(data.get("away_goals", 0))
        fixtures_df.at[idx, "minute"]     = int(data.get("minute", 0))
        fixtures_df.at[idx, "status"]     = data.get("status", "live")
        # Save to disk
        fixtures_df.to_csv(ROOT / "data/wc2026/fixtures.csv", index=False)
        return JSONResponse({"status": "updated", "home": home, "away": away})
    return JSONResponse({"status": "not_found"}, status_code=404)

@app.get("/api/standings")
async def standings():
    """Tabla de posiciones por grupo basada en resultados reales."""
    if fixtures_df is None:
        load_fixtures()
    df = fixtures_df[fixtures_df["status"].str.strip() == "finished"].copy()
    teams = {}
    for _, r in df.iterrows():
        g = r["group"]
        h, a = r["home"].strip(), r["away"].strip()
        hg, ag = int(r["home_goals"]), int(r["away_goals"])
        for t in [h, a]:
            if t not in teams:
                teams[t] = {"team":t,"group":g,"pts":0,"gf":0,"gc":0,"w":0,"d":0,"l":0,"pld":0}
        teams[h]["pld"]+=1; teams[a]["pld"]+=1
        teams[h]["gf"]+=hg; teams[h]["gc"]+=ag
        teams[a]["gf"]+=ag; teams[a]["gc"]+=hg
        if hg>ag:
            teams[h]["w"]+=1; teams[h]["pts"]+=3; teams[a]["l"]+=1
        elif hg==ag:
            teams[h]["d"]+=1; teams[h]["pts"]+=1; teams[a]["d"]+=1; teams[a]["pts"]+=1
        else:
            teams[a]["w"]+=1; teams[a]["pts"]+=3; teams[h]["l"]+=1
    for t in teams.values():
        t["gd"] = t["gf"] - t["gc"]
    by_group = {}
    for t in teams.values():
        g = t["group"]
        if g not in by_group: by_group[g] = []
        by_group[g].append(t)
    for g in by_group:
        by_group[g].sort(key=lambda x: (-x["pts"],-x["gd"],-x["gf"]))
    return JSONResponse(by_group)

@app.get("/api/recent_form")
async def recent_form(team: str, n: int = 5):
    """Últimos N partidos de un equipo."""
    mask = (df_hist["home_team"] == team) | (df_hist["away_team"] == team)
    recent = df_hist[mask].nlargest(n, "date")
    results = []
    for _, r in recent.iterrows():
        is_home = r["home_team"] == team
        gf = int(r["home_score"]) if is_home else int(r["away_score"])
        gc = int(r["away_score"]) if is_home else int(r["home_score"])
        rival = r["away_team"] if is_home else r["home_team"]
        res = "W" if gf > gc else ("D" if gf == gc else "L")
        results.append({
            "date": str(r["date"].date()),
            "rival": rival, "gf": gf, "gc": gc,
            "result": res, "venue": "H" if is_home else "A",
            "tournament": r.get("tournament","")
        })
    return JSONResponse(results)

# Servir el dashboard HTML
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8026))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
