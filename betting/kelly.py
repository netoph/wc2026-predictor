"""
wc2026/betting/kelly.py
Kelly Criterion fraccionado + EV scanner para el Mundial 2026.
Registra todas las apuestas simuladas y trackea el ROI.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent

class KellyBettor:
    """
    Simulador de apuestas con Kelly Criterion fraccionado.
    Bankroll inicial: $1000 USD (simulado).
    Fracción Kelly: 25% (conservador, reduce varianza).
    EV mínimo para apostar: 3%.
    """
    KELLY_FRACTION  = 0.25
    MIN_EV          = 0.03     # 3% mínimo de ventaja
    MAX_BET_PCT     = 0.05     # máximo 5% del bankroll por apuesta
    STOP_LOSS_PCT   = 0.30     # pausa si pierde 30% del bankroll

    def __init__(self, initial_bankroll: float = 1000.0):
        self.bankroll  = initial_bankroll
        self.initial   = initial_bankroll
        self.peak      = initial_bankroll
        self.bets      = []
        self.paused    = False

    # ── Conversión de momios ─────────────────────────────────────────────────
    @staticmethod
    def american_to_decimal(momio: float) -> float:
        if momio > 0:
            return momio / 100 + 1
        return 100 / abs(momio) + 1

    @staticmethod
    def decimal_to_implied(decimal_odds: float) -> float:
        return 1 / decimal_odds

    @staticmethod
    def ev(p_model: float, decimal_odds: float) -> float:
        """Valor esperado por unidad apostada."""
        return p_model * (decimal_odds - 1) - (1 - p_model)

    # ── Kelly Criterion ──────────────────────────────────────────────────────
    def kelly_stake(self, p_model: float, decimal_odds: float) -> float:
        """Retorna el stake óptimo en $ (Kelly fraccionado)."""
        if self.paused:
            return 0.0

        b = decimal_odds - 1
        f_star = (b * p_model - (1 - p_model)) / b  # Kelly completo
        f_kelly = f_star * self.KELLY_FRACTION         # Kelly 25%

        # Caps
        f_kelly = max(0.0, min(f_kelly, self.MAX_BET_PCT))

        ev_val = self.ev(p_model, decimal_odds)
        if ev_val < self.MIN_EV:
            return 0.0

        return round(self.bankroll * f_kelly, 2)

    # ── Registrar apuesta ────────────────────────────────────────────────────
    def place_bet(self, match: str, market: str, selection: str,
                  p_model: float, momio_american: float,
                  match_date: str = None) -> dict | None:
        """Coloca una apuesta simulada si hay EV suficiente."""
        decimal_odds = self.american_to_decimal(momio_american)
        ev_val  = self.ev(p_model, decimal_odds)
        stake   = self.kelly_stake(p_model, decimal_odds)

        if stake <= 0:
            return None

        bet = {
            "id":           len(self.bets) + 1,
            "date":         match_date or datetime.now().strftime("%Y-%m-%d"),
            "match":        match,
            "market":       market,
            "selection":    selection,
            "p_model":      round(p_model, 4),
            "p_implied":    round(self.decimal_to_implied(decimal_odds), 4),
            "momio":        momio_american,
            "decimal_odds": round(decimal_odds, 3),
            "ev":           round(ev_val, 4),
            "stake":        stake,
            "bankroll_before": round(self.bankroll, 2),
            "result":       None,   # se rellena con settle_bet()
            "pnl":          None,
        }
        self.bets.append(bet)
        print(f"  🎯 BET #{bet['id']}: {match} | {market}:{selection} | "
              f"EV={ev_val:+.1%} | Stake=${stake:.2f}")
        return bet

    # ── Liquidar apuesta ─────────────────────────────────────────────────────
    def settle_bet(self, bet_id: int, won: bool) -> float:
        """Liquida una apuesta con el resultado real. Retorna P&L."""
        bet = next((b for b in self.bets if b["id"] == bet_id), None)
        if bet is None or bet["result"] is not None:
            return 0.0

        if won:
            pnl = bet["stake"] * (bet["decimal_odds"] - 1)
            bet["result"] = "WIN"
        else:
            pnl = -bet["stake"]
            bet["result"] = "LOSS"

        bet["pnl"] = round(pnl, 2)
        self.bankroll += pnl
        self.peak = max(self.peak, self.bankroll)

        # Check stop-loss
        if self.bankroll < self.initial * (1 - self.STOP_LOSS_PCT):
            self.paused = True
            print(f"  ⚠️  STOP-LOSS activado. Bankroll: ${self.bankroll:.2f}")

        emoji = "✅" if won else "❌"
        print(f"  {emoji} Settled #{bet_id}: {bet['match']} | P&L={pnl:+.2f} | "
              f"Bankroll=${self.bankroll:.2f}")
        return pnl

    # ── Estadísticas ─────────────────────────────────────────────────────────
    def stats(self) -> dict:
        settled = [b for b in self.bets if b["result"] is not None]
        if not settled:
            return {"bets": 0}

        wins     = [b for b in settled if b["result"] == "WIN"]
        total_staked = sum(b["stake"] for b in settled)
        total_pnl    = sum(b["pnl"] for b in settled)
        roi          = total_pnl / total_staked if total_staked > 0 else 0
        hit_rate     = len(wins) / len(settled) if settled else 0
        max_dd       = (self.peak - self.bankroll) / self.peak

        # Por mercado
        markets = {}
        for b in settled:
            m = b["market"]
            if m not in markets:
                markets[m] = {"bets": 0, "wins": 0, "pnl": 0.0}
            markets[m]["bets"] += 1
            markets[m]["wins"] += (1 if b["result"] == "WIN" else 0)
            markets[m]["pnl"]  += b["pnl"]

        return {
            "total_bets":    len(settled),
            "pending_bets":  len(self.bets) - len(settled),
            "hit_rate":      round(hit_rate, 4),
            "total_staked":  round(total_staked, 2),
            "total_pnl":     round(total_pnl, 2),
            "roi":           round(roi, 4),
            "bankroll":      round(self.bankroll, 2),
            "max_drawdown":  round(max_dd, 4),
            "by_market":     markets,
        }

    def print_stats(self):
        s = self.stats()
        print("\n" + "=" * 55)
        print(f"  💰 SIMULADOR WC 2026 — REPORTE")
        print("=" * 55)
        print(f"  Bankroll inicial:  ${self.initial:>10,.2f}")
        print(f"  Bankroll actual:   ${s['bankroll']:>10,.2f}")
        print(f"  P&L total:         ${s['total_pnl']:>+10,.2f}")
        print(f"  ROI:               {s['roi']:>+10.1%}")
        print(f"  Hit Rate:          {s['hit_rate']:>10.1%}")
        print(f"  Total apostado:    ${s['total_staked']:>10,.2f}")
        print(f"  Apuestas:          {s['total_bets']:>10}")
        print(f"  Max Drawdown:      {s['max_drawdown']:>10.1%}")
        print(f"\n  Por mercado:")
        for mkt, d in s.get("by_market", {}).items():
            pct = d["wins"] / d["bets"] if d["bets"] else 0
            print(f"    {mkt:<18} {d['bets']:>3} bets | "
                  f"{pct:.0%} hit | P&L: ${d['pnl']:>+.2f}")
        print("=" * 55)

    def save(self, path: Path = None):
        path = path or ROOT / "data" / "tracker" / "bets.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.bets).to_csv(path, index=False)
        print(f"✓ Apuestas guardadas: {path}")


if __name__ == "__main__":
    # Test rápido
    bot = KellyBettor(initial_bankroll=1000)

    # Simular algunas apuestas
    b1 = bot.place_bet("Brazil vs Argentina", "1X2", "Brazil",
                       p_model=0.52, momio_american=+140, match_date="2026-06-15")
    b2 = bot.place_bet("France vs Germany", "Over 2.5", "Over",
                       p_model=0.55, momio_american=+120, match_date="2026-06-16")
    b3 = bot.place_bet("Spain vs Morocco", "1X2", "Spain",
                       p_model=0.65, momio_american=-180, match_date="2026-06-17")

    # Liquidar
    if b1: bot.settle_bet(b1["id"], won=True)
    if b2: bot.settle_bet(b2["id"], won=False)
    if b3: bot.settle_bet(b3["id"], won=True)

    bot.print_stats()
