"""
wc2026/model/evaluate.py
═══════════════════════════════════════════════════════════════
Evaluación exhaustiva del modelo en 3 splits:
  - TRAIN:  2014-2023 (ajuste de parámetros)
  - TEST:   2024-ene a 2025-jun (validación temporal)
  - BLIND:  2025-jul a 2026+ (datos nunca vistos, amistosos pre-WC)

Calcula LogLoss, Brier, Hit Rate 1X2, O/U accuracy, xG MAE.
═══════════════════════════════════════════════════════════════
"""
import numpy as np
import pandas as pd
from scipy.stats import poisson
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def evaluate_model(elo, negbin, df_all, calibrator=None):
    """
    Evalúa el modelo en train/test/blind splits.
    Returns dict with all KPIs.
    """
    df = df_all.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # Define splits
    splits = {
        "train": df[(df["date"] >= "2014-01-01") & (df["date"] < "2024-01-01")],
        "test":  df[(df["date"] >= "2024-01-01") & (df["date"] < "2025-07-01")],
        "blind": df[df["date"] >= "2025-07-01"],
    }

    results = {}
    for split_name, split_df in splits.items():
        if len(split_df) < 10:
            results[split_name] = {"n": len(split_df), "error": "insufficient data"}
            continue

        metrics = _eval_split(split_df, elo, negbin, calibrator)
        results[split_name] = metrics

    # Model metadata
    results["model_info"] = {
        "elo_teams": len(elo.ratings) if elo else 0,
        "negbin_teams": len(negbin.att) if negbin and negbin.is_fitted else 0,
        "negbin_home_adv": round(negbin.home_adv, 4) if negbin and negbin.is_fitted else 0,
        "calibration_T": round(calibrator.T, 4) if calibrator and calibrator.fitted else 1.0,
        "top_attack": _top_n(negbin.att, 10, reverse=True) if negbin and negbin.is_fitted else [],
        "top_defense": _top_n(negbin.def_, 10, reverse=False) if negbin and negbin.is_fitted else [],
        "top_elo": sorted(elo.ratings.items(), key=lambda x: -x[1])[:10] if elo else [],
    }

    return results


def _top_n(d, n, reverse=True):
    items = sorted(d.items(), key=lambda x: x[1], reverse=reverse)[:n]
    return [{"team": t, "value": round(v, 3)} for t, v in items]


def _eval_split(df, elo, negbin, calibrator):
    """Evaluate model on a data split."""
    n = len(df)
    probs_raw = []
    probs_nb = []
    labels = []
    xg_errors = []
    ou25_preds = []
    ou25_actuals = []
    btts_preds = []
    btts_actuals = []
    match_details = []
    skipped = 0

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        hg = int(row["home_score"])
        ag = int(row["away_score"])
        total = hg + ag

        # True label
        if hg > ag: label = 0
        elif hg == ag: label = 1
        else: label = 2

        # ELO prediction
        try:
            pred_elo = elo.predict_match(home, away, neutral=True)
            p_elo = [pred_elo["home"], pred_elo["draw"], pred_elo["away"]]
        except:
            skipped += 1
            continue

        probs_raw.append(p_elo)
        labels.append(label)

        # NegBin prediction
        if negbin and negbin.is_fitted and home in negbin.att and away in negbin.att:
            lam, mu = negbin.expected_goals(home, away, neutral=True)
            # NB-based 1X2
            max_g = 8
            joint = np.zeros((max_g, max_g))
            for i in range(max_g):
                for j in range(max_g):
                    joint[i, j] = poisson.pmf(i, lam) * poisson.pmf(j, mu)
            joint /= joint.sum()
            ph = float(np.tril(joint, -1).sum())
            pd_ = float(np.trace(joint))
            pa = float(np.triu(joint, 1).sum())
            t = ph + pd_ + pa
            probs_nb.append([ph/t, pd_/t, pa/t])

            # O/U 2.5
            p_over25 = 1 - sum(poisson.pmf(k, lam + mu) for k in range(3))
            ou25_preds.append(p_over25)
            ou25_actuals.append(total > 2.5)

            # BTTS
            p_btts = (1 - poisson.pmf(0, lam)) * (1 - poisson.pmf(0, mu))
            btts_preds.append(p_btts)
            btts_actuals.append(hg > 0 and ag > 0)

            # xG error
            xg_errors.append(abs((lam + mu) - total))

            match_details.append({
                "home": home, "away": away,
                "hg": hg, "ag": ag,
                "p_home": round(ph/t, 3), "p_draw": round(pd_/t, 3), "p_away": round(pa/t, 3),
                "xg": round(lam + mu, 2),
                "correct": (["H","D","A"][np.argmax([ph,pd_,pa])] == ["H","D","A"][label]),
            })

    probs_raw = np.array(probs_raw)
    labels = np.array(labels)

    # Core metrics - ELO
    result = {
        "n": n,
        "skipped": skipped,
        "evaluated": len(labels),
    }

    if len(labels) > 0:
        # ELO metrics
        p_clip = np.clip(probs_raw, 1e-12, 1.0)
        elo_logloss = -np.mean(np.log(p_clip[np.arange(len(labels)), labels]))
        one_hot = np.eye(3)[labels]
        elo_brier = float(np.mean(np.sum((probs_raw - one_hot) ** 2, axis=1)))
        elo_preds = probs_raw.argmax(axis=1)
        elo_hit = float(np.mean(elo_preds == labels))

        result["elo"] = {
            "logloss": round(elo_logloss, 4),
            "brier": round(elo_brier, 4),
            "hit_rate": round(elo_hit, 4),
            "naive_logloss": round(-np.log(1/3), 4),
        }

        # Calibrated ELO
        if calibrator and calibrator.fitted:
            p_cal = calibrator.transform(probs_raw)
            p_cal_clip = np.clip(p_cal, 1e-12, 1.0)
            cal_logloss = -np.mean(np.log(p_cal_clip[np.arange(len(labels)), labels]))
            cal_brier = float(np.mean(np.sum((p_cal - one_hot) ** 2, axis=1)))
            result["elo_calibrated"] = {
                "logloss": round(cal_logloss, 4),
                "brier": round(cal_brier, 4),
            }

    # NegBin metrics
    if probs_nb:
        probs_nb = np.array(probs_nb)
        nb_labels = labels[:len(probs_nb)]
        p_nb_clip = np.clip(probs_nb, 1e-12, 1.0)
        nb_logloss = -np.mean(np.log(p_nb_clip[np.arange(len(nb_labels)), nb_labels]))
        one_hot_nb = np.eye(3)[nb_labels]
        nb_brier = float(np.mean(np.sum((probs_nb - one_hot_nb) ** 2, axis=1)))
        nb_preds = probs_nb.argmax(axis=1)
        nb_hit = float(np.mean(nb_preds == nb_labels))

        result["negbin"] = {
            "logloss": round(nb_logloss, 4),
            "brier": round(nb_brier, 4),
            "hit_rate": round(nb_hit, 4),
            "n_evaluated": len(nb_labels),
        }

    # O/U 2.5
    if ou25_preds:
        ou_preds_bin = [p > 0.5 for p in ou25_preds]
        ou_hit = sum(1 for p, a in zip(ou_preds_bin, ou25_actuals) if p == a) / len(ou25_preds)
        ou_logloss = -np.mean([
            a * np.log(max(p, 1e-12)) + (1-a) * np.log(max(1-p, 1e-12))
            for p, a in zip(ou25_preds, ou25_actuals)
        ])
        result["over_under"] = {
            "hit_rate": round(ou_hit, 4),
            "logloss": round(ou_logloss, 4),
            "n": len(ou25_preds),
            "actual_over_pct": round(sum(ou25_actuals) / len(ou25_actuals), 4),
        }

    # BTTS
    if btts_preds:
        btts_preds_bin = [p > 0.5 for p in btts_preds]
        btts_hit = sum(1 for p, a in zip(btts_preds_bin, btts_actuals) if p == a) / len(btts_preds)
        result["btts"] = {
            "hit_rate": round(btts_hit, 4),
            "n": len(btts_preds),
            "actual_btts_pct": round(sum(btts_actuals) / len(btts_actuals), 4),
        }

    # xG
    if xg_errors:
        result["xg"] = {
            "mae": round(np.mean(xg_errors), 3),
            "rmse": round(np.sqrt(np.mean(np.array(xg_errors)**2)), 3),
            "median_ae": round(np.median(xg_errors), 3),
        }

    # Confidence breakdown
    if len(labels) > 0 and len(probs_raw) > 0:
        max_p = probs_raw.max(axis=1)
        correct = (probs_raw.argmax(axis=1) == labels)
        bins = [(0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.0)]
        conf_breakdown = []
        for lo, hi in bins:
            mask = (max_p >= lo) & (max_p < hi)
            if mask.sum() > 0:
                conf_breakdown.append({
                    "range": f"{lo:.0%}-{hi:.0%}",
                    "n": int(mask.sum()),
                    "accuracy": round(float(correct[mask].mean()), 4),
                    "avg_confidence": round(float(max_p[mask].mean()), 4),
                })
        result["confidence_bins"] = conf_breakdown

    # Sample predictions (last 15 for blind)
    if match_details:
        result["sample_predictions"] = match_details[-15:]

    return result
