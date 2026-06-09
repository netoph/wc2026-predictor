"""
wc2026/betting/auto_scanner.py
═══════════════════════════════════════════════════════════════
Smart Auto-EV Scanner v2: prioriza mercados donde el modelo
tiene más edge estadístico.

Jerarquía de confianza:
  1. Over/Under goles  → Poisson/NegBin directo (alta confianza)
  2. BTTS              → producto de probabilidades marginales
  3. Córners O/U       → correlación empírica calibrada
  4. 1X2               → más difícil, solo con EV grande (>10%)

Cada mercado tiene su propio MIN_EV threshold.
═══════════════════════════════════════════════════════════════
"""
import numpy as np
from datetime import datetime


# Confianza del modelo por mercado (más alto = más fácil de predecir)
MARKET_CONFIG = {
    "Over 2.5":    {"min_ev": 0.04, "confidence": 0.85, "kelly_mult": 1.0,  "priority": 1},
    "Under 2.5":   {"min_ev": 0.04, "confidence": 0.85, "kelly_mult": 1.0,  "priority": 1},
    "Over 1.5":    {"min_ev": 0.05, "confidence": 0.80, "kelly_mult": 0.8,  "priority": 2},
    "BTTS":        {"min_ev": 0.05, "confidence": 0.75, "kelly_mult": 0.8,  "priority": 2},
    "Corners O9.5":{"min_ev": 0.06, "confidence": 0.70, "kelly_mult": 0.6,  "priority": 3},
    "1X2_Home":    {"min_ev": 0.10, "confidence": 0.55, "kelly_mult": 0.5,  "priority": 4},
    "1X2_Draw":    {"min_ev": 0.15, "confidence": 0.40, "kelly_mult": 0.3,  "priority": 5},
    "1X2_Away":    {"min_ev": 0.10, "confidence": 0.55, "kelly_mult": 0.5,  "priority": 4},
}


class AutoEVScanner:
    """
    Bot inteligente: escanea momios DraftKings, compara con el modelo,
    y solo apuesta donde tiene edge real. Prioriza O/U > BTTS > 1X2.
    """

    def __init__(self, bettor, compute_fn):
        self.bettor = bettor
        self.compute = compute_fn
        self.scanned_matches = set()
        self.scan_log = []
        self.bet_log = []        # historial completo del bot
        self.stats_by_market = {}

    @staticmethod
    def parse_american(odds_str) -> int:
        try:
            return int(str(odds_str).strip().replace("+", ""))
        except:
            return 0

    @staticmethod
    def american_to_decimal(m: int) -> float:
        if m > 0: return m / 100 + 1
        if m < 0: return 100 / abs(m) + 1
        return 1.0

    @staticmethod
    def decimal_to_implied(dec: float) -> float:
        return 1 / dec if dec > 0 else 0

    def _calc_ev(self, p_model: float, momio: int) -> float:
        dec = self.american_to_decimal(momio)
        return p_model * (dec - 1) - (1 - p_model)

    def _calc_edge_quality(self, ev: float, market: str, p_model: float) -> dict:
        """Evalúa la calidad del edge considerando el mercado."""
        cfg = MARKET_CONFIG.get(market, {"min_ev": 0.10, "confidence": 0.50, "kelly_mult": 0.5, "priority": 5})

        # Adjusted EV considering model confidence
        adj_ev = ev * cfg["confidence"]

        # Grade
        if adj_ev >= 0.12: grade = "A+"
        elif adj_ev >= 0.08: grade = "A"
        elif adj_ev >= 0.05: grade = "B+"
        elif adj_ev >= 0.03: grade = "B"
        elif adj_ev >= cfg["min_ev"]: grade = "C"
        else: grade = "SKIP"

        return {
            "raw_ev": round(ev, 4),
            "adj_ev": round(adj_ev, 4),
            "min_ev": cfg["min_ev"],
            "confidence": cfg["confidence"],
            "kelly_mult": cfg["kelly_mult"],
            "priority": cfg["priority"],
            "grade": grade,
            "bet": grade != "SKIP" and ev >= cfg["min_ev"],
        }

    def scan_match(self, home: str, away: str, odds: dict,
                   neutral: bool = True, altitude: float = 0,
                   match_date: str = None) -> list:
        """Escanea un partido, evalúa todos los mercados, apuesta inteligentemente."""
        match_key = f"{home} vs {away}"
        if match_key in self.scanned_matches:
            return []

        pred = self.compute(home, away, neutral, altitude)
        placed = []
        date_str = match_date or datetime.now().strftime("%Y-%m-%d")

        # Todos los mercados disponibles
        markets = []

        # O/U 2.5 — highest confidence
        if odds.get("over25"):
            momio = self.parse_american(odds["over25"])
            if momio: markets.append(("Over 2.5", "Over 2.5", pred.get("p_over25", 0), momio))
        if odds.get("under25"):
            momio = self.parse_american(odds["under25"])
            if momio: markets.append(("Under 2.5", "Under 2.5", pred.get("p_under25", 0), momio))

        # 1X2 — lowest confidence, highest threshold
        if odds.get("ml_home"):
            momio = self.parse_american(odds["ml_home"])
            if momio: markets.append(("1X2_Home", "1X2", pred.get("p_home", 0), momio))
        if odds.get("ml_draw"):
            momio = self.parse_american(odds["ml_draw"])
            if momio: markets.append(("1X2_Draw", "1X2", pred.get("p_draw", 0), momio))
        if odds.get("ml_away"):
            momio = self.parse_american(odds["ml_away"])
            if momio: markets.append(("1X2_Away", "1X2", pred.get("p_away", 0), momio))

        # Sort by priority (O/U first, 1X2 last)
        markets.sort(key=lambda x: MARKET_CONFIG.get(x[0], {}).get("priority", 99))

        for market_key, market_name, p_model, momio in markets:
            if not p_model or p_model <= 0.01:
                continue

            ev = self._calc_ev(p_model, momio)
            quality = self._calc_edge_quality(ev, market_key, p_model)
            p_implied = self.decimal_to_implied(self.american_to_decimal(momio))
            edge_vs_market = p_model - p_implied

            scan_entry = {
                "timestamp": datetime.now().isoformat(),
                "match": match_key,
                "market": market_key,
                "market_name": market_name,
                "selection": market_key.split("_")[-1] if "_" in market_key else market_key,
                "p_model": round(p_model, 4),
                "p_implied": round(p_implied, 4),
                "edge": round(edge_vs_market, 4),
                "momio": momio,
                "ev": quality["raw_ev"],
                "adj_ev": quality["adj_ev"],
                "grade": quality["grade"],
                "confidence": quality["confidence"],
                "action": "skip",
                "stake": 0,
                "bet_id": None,
                "result": None,
                "pnl": None,
            }

            if quality["bet"]:
                # Adjust Kelly fraction by market confidence
                orig_fraction = self.bettor.KELLY_FRACTION
                self.bettor.KELLY_FRACTION = orig_fraction * quality["kelly_mult"]

                bet = self.bettor.place_bet(
                    match_key, market_name,
                    scan_entry["selection"],
                    p_model, momio, date_str
                )

                self.bettor.KELLY_FRACTION = orig_fraction  # restore

                if bet:
                    scan_entry["action"] = "BET"
                    scan_entry["stake"] = bet["stake"]
                    scan_entry["bet_id"] = bet["id"]
                    placed.append(bet)

                    # Track in bot log
                    self.bet_log.append({
                        **scan_entry,
                        "home": home, "away": away,
                        "lambda": pred.get("lambda", 0),
                        "mu": pred.get("mu", 0),
                        "total_xg": pred.get("total_goals_expected", 0),
                    })

            self.scan_log.append(scan_entry)

        if placed:
            self.scanned_matches.add(match_key)

        return placed

    def scan_all(self, live_odds: dict, fixtures: list = None) -> dict:
        """Escanea todos los momios disponibles."""
        total_placed = []
        for match_key, odds in live_odds.items():
            parts = match_key.split(" vs ")
            if len(parts) != 2:
                continue
            home, away = parts[0].strip(), parts[1].strip()

            alt = 0
            if fixtures:
                for f in fixtures:
                    if f.get("home", "").strip() == home and f.get("away", "").strip() == away:
                        alt = float(f.get("altitude", 0) or 0)
                        break

            placed = self.scan_match(home, away, odds, neutral=True, altitude=alt)
            total_placed.extend(placed)

        return {
            "scanned": len(live_odds),
            "placed": len(total_placed),
            "bets": total_placed,
        }

    def settle_bot_bet(self, bet_id: int, won: bool):
        """Actualiza el resultado en el log del bot."""
        for entry in self.bet_log:
            if entry.get("bet_id") == bet_id:
                entry["result"] = "WIN" if won else "LOSS"
                bet = next((b for b in self.bettor.bets if b["id"] == bet_id), None)
                if bet:
                    entry["pnl"] = bet.get("pnl", 0)
                break

    def get_bot_stats(self) -> dict:
        """Estadísticas completas del bot por mercado."""
        settled = [b for b in self.bet_log if b.get("result") is not None]
        pending = [b for b in self.bet_log if b.get("result") is None]

        if not self.bet_log:
            return {
                "total_bets": 0, "settled": 0, "pending": len(pending),
                "by_market": {}, "bankroll": self.bettor.bankroll,
            }

        by_market = {}
        for b in self.bet_log:
            mkt = b["market"]
            if mkt not in by_market:
                by_market[mkt] = {"bets": 0, "settled": 0, "wins": 0,
                                  "pnl": 0, "avg_ev": 0, "avg_edge": 0,
                                  "total_staked": 0, "evs": [], "edges": []}
            by_market[mkt]["bets"] += 1
            by_market[mkt]["total_staked"] += b.get("stake", 0)
            by_market[mkt]["evs"].append(b.get("ev", 0))
            by_market[mkt]["edges"].append(b.get("edge", 0))
            if b.get("result"):
                by_market[mkt]["settled"] += 1
                if b["result"] == "WIN":
                    by_market[mkt]["wins"] += 1
                by_market[mkt]["pnl"] += b.get("pnl", 0) or 0

        for mkt, d in by_market.items():
            d["hit_rate"] = round(d["wins"] / d["settled"], 4) if d["settled"] else 0
            d["roi"] = round(d["pnl"] / d["total_staked"], 4) if d["total_staked"] else 0
            d["avg_ev"] = round(np.mean(d["evs"]), 4) if d["evs"] else 0
            d["avg_edge"] = round(np.mean(d["edges"]), 4) if d["edges"] else 0
            del d["evs"], d["edges"]

        total_pnl = sum(b.get("pnl", 0) or 0 for b in settled)
        total_staked = sum(b.get("stake", 0) for b in settled)

        return {
            "total_bets": len(self.bet_log),
            "settled": len(settled),
            "pending": len(pending),
            "wins": sum(1 for b in settled if b["result"] == "WIN"),
            "hit_rate": round(sum(1 for b in settled if b["result"] == "WIN") / len(settled), 4) if settled else 0,
            "total_pnl": round(total_pnl, 2),
            "total_staked": round(total_staked, 2),
            "roi": round(total_pnl / total_staked, 4) if total_staked else 0,
            "bankroll": round(self.bettor.bankroll, 2),
            "by_market": by_market,
        }

    def get_scan_summary(self) -> list:
        return self.scan_log[-50:]

    def get_bet_log(self) -> list:
        return self.bet_log
