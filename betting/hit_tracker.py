"""
wc2026/betting/hit_tracker.py
═══════════════════════════════════════════════════════════════
Hit Rate Tracker: evalúa la precisión del modelo partido por
partido. Calcula LogLoss, Brier Score, y hit rate acumulado.
═══════════════════════════════════════════════════════════════
"""
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent


class HitTracker:
    """
    Registra predicciones vs resultados reales.
    Calcula métricas de calibración en tiempo real.
    """
    def __init__(self):
        self.records = []

    def record_match(self, home: str, away: str,
                     p_home: float, p_draw: float, p_away: float,
                     home_goals: int, away_goals: int,
                     p_over25: float = None, p_btts: float = None,
                     total_goals_expected: float = None):
        """Registra un partido finalizado con su predicción y resultado."""
        # Resultado real
        if home_goals > away_goals:
            result = "H"
            result_idx = 0
        elif home_goals == away_goals:
            result = "D"
            result_idx = 1
        else:
            result = "A"
            result_idx = 2

        # Predicción del modelo
        probs = np.array([p_home, p_draw, p_away])
        pred = ["H", "D", "A"][np.argmax(probs)]
        correct = (pred == result)

        # LogLoss para este partido
        p_clipped = np.clip(probs, 1e-12, 1.0)
        logloss = -np.log(p_clipped[result_idx])

        # Brier Score
        one_hot = np.zeros(3)
        one_hot[result_idx] = 1.0
        brier = float(np.sum((probs - one_hot) ** 2))

        # Over/Under
        total = home_goals + away_goals
        over25_correct = None
        if p_over25 is not None:
            over25_actual = total > 2.5
            over25_pred = p_over25 > 0.5
            over25_correct = (over25_pred == over25_actual)

        # BTTS
        btts_correct = None
        if p_btts is not None:
            btts_actual = (home_goals > 0 and away_goals > 0)
            btts_pred = p_btts > 0.5
            btts_correct = (btts_pred == btts_actual)

        # Goal error
        goal_error = abs((total_goals_expected or 0) - total) if total_goals_expected else None

        record = {
            "timestamp":  datetime.now().isoformat(),
            "home": home, "away": away,
            "p_home": round(p_home, 4), "p_draw": round(p_draw, 4), "p_away": round(p_away, 4),
            "home_goals": home_goals, "away_goals": away_goals,
            "total_goals": total,
            "result": result,
            "prediction": pred,
            "correct_1x2": correct,
            "logloss": round(logloss, 4),
            "brier": round(brier, 4),
            "p_over25": p_over25,
            "over25_correct": over25_correct,
            "p_btts": p_btts,
            "btts_correct": btts_correct,
            "xg_expected": total_goals_expected,
            "goal_error": round(goal_error, 2) if goal_error is not None else None,
        }
        self.records.append(record)

        emoji = "✅" if correct else "❌"
        print(f"  {emoji} HIT: {home} {home_goals}-{away_goals} {away} | "
              f"Pred:{pred} Real:{result} | LL:{logloss:.3f} Brier:{brier:.3f}")
        return record

    def stats(self) -> dict:
        """Retorna métricas acumuladas."""
        if not self.records:
            return {"matches_tracked": 0}

        n = len(self.records)
        correct_1x2 = sum(1 for r in self.records if r["correct_1x2"])
        avg_logloss = np.mean([r["logloss"] for r in self.records])
        avg_brier   = np.mean([r["brier"] for r in self.records])

        # Over 2.5
        over_records = [r for r in self.records if r["over25_correct"] is not None]
        over_hit = sum(1 for r in over_records if r["over25_correct"]) / len(over_records) if over_records else 0

        # BTTS
        btts_records = [r for r in self.records if r["btts_correct"] is not None]
        btts_hit = sum(1 for r in btts_records if r["btts_correct"]) / len(btts_records) if btts_records else 0

        # Goal MAE
        goal_errors = [r["goal_error"] for r in self.records if r["goal_error"] is not None]
        goal_mae = np.mean(goal_errors) if goal_errors else None

        return {
            "matches_tracked": n,
            "hit_rate_1x2": round(correct_1x2 / n, 4),
            "avg_logloss": round(avg_logloss, 4),
            "avg_brier": round(avg_brier, 4),
            "over25_hit_rate": round(over_hit, 4),
            "btts_hit_rate": round(btts_hit, 4),
            "goal_mae": round(goal_mae, 2) if goal_mae else None,
            "naïve_logloss": round(-np.log(1/3), 4),  # baseline
        }

    def get_records(self) -> list:
        return self.records

    def save(self, path: Path = None):
        path = path or ROOT / "data" / "tracker" / "hit_tracker.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.records).to_csv(path, index=False)
        print(f"✓ Hit tracker guardado: {path}")
