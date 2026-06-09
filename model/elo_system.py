"""
wc2026/model/elo_system.py
Sistema ELO dinámico para selecciones nacionales.
K-factor adaptativo según importancia del partido y diferencia de ratings.
Basado en World Football ELO methodology + ajustes Groll 2022.
"""
import numpy as np
import pandas as pd
from pathlib import Path

# ─── Importancia de torneos (K base) ─────────────────────────────────────────
TOURNAMENT_K = {
    "FIFA World Cup": 60,
    "FIFA World Cup qualification": 40,
    "UEFA Euro": 50,
    "UEFA European Championship": 50,
    "Copa América": 50,
    "CONCACAF Gold Cup": 45,
    "Africa Cup of Nations": 45,
    "AFC Asian Cup": 45,
    "UEFA Nations League": 40,
    "CONCACAF Nations League": 40,
    "FIFA Confederations Cup": 45,
    "Friendly": 20,
}

# ELO ratings de arranque para las 48 selecciones del Mundial 2026
# Fuente: worldfootballelo.com (pre-torneo)
INITIAL_ELO = {
    # Top tier
    "France": 2102, "Spain": 2075, "England": 2043, "Brazil": 2036,
    "Argentina": 2141, "Portugal": 2000, "Netherlands": 1985,
    "Belgium": 1960, "Germany": 1975, "Italy": 1950,
    # UEFA segunda línea
    "Croatia": 1920, "Denmark": 1910, "Switzerland": 1900,
    "Austria": 1870, "Serbia": 1850, "Poland": 1840,
    "Ukraine": 1830, "Turkey": 1820, "Hungary": 1800,
    "Scotland": 1790, "Czech Republic": 1810, "Slovakia": 1780,
    "Romania": 1760, "Slovenia": 1770, "Albania": 1740,
    # CONMEBOL
    "Uruguay": 1910, "Colombia": 1885, "Ecuador": 1860,
    "Chile": 1840, "Peru": 1820, "Bolivia": 1770,
    "Paraguay": 1780, "Venezuela": 1760,
    # CONCACAF
    "United States": 1870, "Mexico": 1855, "Canada": 1850,
    "Costa Rica": 1800, "Honduras": 1760, "Panama": 1780,
    "Jamaica": 1720, "El Salvador": 1710, "Guatemala": 1690,
    # CAF
    "Morocco": 1880, "Senegal": 1850, "Nigeria": 1840,
    "Ivory Coast": 1830, "Ghana": 1810, "Egypt": 1820,
    "Algeria": 1800, "Cameroon": 1790, "Mali": 1760,
    "South Africa": 1750, "Tunisia": 1780, "DR Congo": 1740,
    "Burkina Faso": 1730, "Cape Verde": 1720, "Tanzania": 1680,
    # AFC
    "Japan": 1870, "South Korea": 1840, "Iran": 1830,
    "Saudi Arabia": 1810, "Australia": 1800, "Jordan": 1760,
    "Iraq": 1770, "Oman": 1720, "Qatar": 1730,
    "UAE": 1710, "Uzbekistan": 1750,
    # OFC
    "New Zealand": 1680,
    # Default
    "DEFAULT": 1500,
}

class ELOSystem:
    def __init__(self, initial_ratings: dict = None):
        self.ratings = {**INITIAL_ELO}
        if initial_ratings:
            self.ratings.update(initial_ratings)
        self.history = []

    def get(self, team: str) -> float:
        return self.ratings.get(team, self.ratings.get("DEFAULT", 1500))

    def expected_score(self, elo_a: float, elo_b: float,
                       neutral: bool = False, altitude: float = 0) -> float:
        """P(A gana) desde ELO, con ajuste por altitud y sede."""
        home_advantage = 0 if neutral else 100  # +100 puntos ELO si es local
        altitude_bonus  = altitude * 0.0002 * 100  # ~+20 puntos por 1000m
        return 1 / (1 + 10 ** (-(elo_a + home_advantage + altitude_bonus - elo_b) / 400))

    def k_factor(self, tournament: str, elo_diff: float, goal_diff: int) -> float:
        """K-factor adaptativo."""
        k_base = TOURNAMENT_K.get(tournament, 30)

        # Multiplicador por diferencia de goles
        if goal_diff == 1:
            goal_mult = 1.0
        elif goal_diff == 2:
            goal_mult = 1.5
        else:
            goal_mult = 1.75 + (goal_diff - 3) * 0.1

        # Corrección por diferencia de ELO (resultados sorpresa = más K)
        elo_correction = 1.0
        if elo_diff > 200:
            elo_correction = 0.85  # favorito que gana ≠ tanta info
        elif elo_diff < -200:
            elo_correction = 1.20  # outsider que gana = más info

        return k_base * goal_mult * elo_correction

    def update(self, home_team: str, away_team: str,
               home_goals: int, away_goals: int,
               tournament: str = "Friendly",
               neutral: bool = False,
               altitude: float = 0) -> tuple:
        """Actualiza ELO con resultado del partido. Retorna (delta_home, delta_away)."""
        elo_h = self.get(home_team)
        elo_a = self.get(away_team)

        p_home = self.expected_score(elo_h, elo_a, neutral, altitude)

        # Score real: 1=gana, 0.5=empate, 0=pierde
        if home_goals > away_goals:
            score = 1.0
        elif home_goals == away_goals:
            score = 0.5
        else:
            score = 0.0

        goal_diff = abs(home_goals - away_goals)
        elo_diff  = elo_h - elo_a
        k = self.k_factor(tournament, elo_diff, goal_diff)

        delta_home =  k * (score - p_home)
        delta_away = -delta_home

        self.ratings[home_team] = elo_h + delta_home
        self.ratings[away_team] = elo_a + delta_away

        self.history.append({
            "home_team": home_team, "away_team": away_team,
            "home_goals": home_goals, "away_goals": away_goals,
            "tournament": tournament,
            "elo_before_home": elo_h, "elo_before_away": elo_a,
            "delta_home": round(delta_home, 1),
            "delta_after_home": round(self.ratings[home_team], 1),
        })
        return delta_home, delta_away

    def fit_historical(self, df: pd.DataFrame,
                       from_year: int = 2000,
                       verbose: bool = True) -> "ELOSystem":
        """Entrena ELO con histórico completo desde un año base."""
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"].dt.year >= from_year].sort_values("date")
        df = df.dropna(subset=["home_score","away_score"])

        for _, row in df.iterrows():
            self.update(
                home_team  = row["home_team"],
                away_team  = row["away_team"],
                home_goals = int(row["home_score"]),
                away_goals = int(row["away_score"]),
                tournament = row.get("tournament", "Friendly"),
                neutral    = bool(row.get("neutral", False)),
            )

        if verbose:
            top = sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)[:20]
            print(f"\nTop 20 ELO al {df['date'].max().date()}:")
            for i, (t, r) in enumerate(top, 1):
                print(f"  {i:>2}. {t:<30} {r:.0f}")

        return self

    def predict_match(self, home_team: str, away_team: str,
                      neutral: bool = False, altitude: float = 0) -> dict:
        """Retorna probabilidades 1X2 puras desde ELO."""
        p_home = self.expected_score(
            self.get(home_team), self.get(away_team), neutral, altitude
        )
        # Ajuste empírico para incluir empate (Ley de los tres resultados)
        # Basado en distribución histórica de p_home → p_draw
        p_draw  = 0.265 - 0.30 * abs(p_home - 0.5)
        p_away  = 1 - p_home - p_draw

        # Asegurar positividad
        p_draw  = max(p_draw,  0.05)
        p_away  = max(p_away,  0.05)
        total   = p_home + p_draw + p_away
        return {
            "home": p_home / total,
            "draw": p_draw / total,
            "away": p_away / total,
            "elo_home": self.get(home_team),
            "elo_away": self.get(away_team),
            "elo_diff": self.get(home_team) - self.get(away_team),
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [(t, r) for t, r in self.ratings.items()],
            columns=["team", "elo"]
        ).sort_values("elo", ascending=False)


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    df = pd.read_csv(ROOT / "data" / "international" / "results.csv")
    print(f"Entrenando ELO con {len(df):,} partidos...")
    elo = ELOSystem()
    elo.fit_historical(df, from_year=2000)

    # Test: Brasil vs Argentina
    print("\nBrasil vs Argentina (neutral):")
    pred = elo.predict_match("Brazil", "Argentina", neutral=True)
    print(f"  Brasil: {pred['home']:.1%} | Empate: {pred['draw']:.1%} | Argentina: {pred['away']:.1%}")
    print(f"  ELO Brasil: {pred['elo_home']:.0f} | ELO Argentina: {pred['elo_away']:.0f}")

    # Guardar ratings
    out = ROOT / "data" / "wc2026" / "elo_ratings.csv"
    out.parent.mkdir(exist_ok=True)
    elo.to_dataframe().to_csv(out, index=False)
    print(f"\n✓ ELO ratings guardados: {out}")
