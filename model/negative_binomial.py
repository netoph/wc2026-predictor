"""
wc2026/model/negative_binomial.py
═══════════════════════════════════════════════════════════════
Negative Binomial Goal Model — adaptado de Groll et al. 2022
Maneja overdispersión en selecciones nacionales.
Parámetros calibrados con backtest WC 2010-2022.
═══════════════════════════════════════════════════════════════
"""
import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson
from scipy.optimize import minimize
from scipy.special import gammaln
import warnings; warnings.filterwarnings("ignore")


# Dispersión por confederación (empírico, Groll 2022)
PHI_CONF = {
    "UEFA":     1.40,
    "CONMEBOL": 1.80,
    "CONCACAF": 2.10,
    "CAF":      2.30,
    "AFC":      1.90,
    "OFC":      2.50,
    "OTHER":    2.00,
}

TEAM_CONF = {
    # UEFA
    **{t: "UEFA" for t in [
        "Spain","France","Germany","England","Portugal","Netherlands",
        "Belgium","Italy","Croatia","Serbia","Poland","Switzerland",
        "Austria","Denmark","Sweden","Norway","Czech Republic","Slovakia",
        "Hungary","Romania","Turkey","Ukraine","Russia","Scotland",
        "Wales","Northern Ireland","Republic of Ireland","Albania",
        "Bosnia and Herzegovina","Kosovo","North Macedonia","Montenegro",
        "Slovenia","Finland","Iceland","Greece","Bulgaria",
    ]},
    # CONMEBOL
    **{t: "CONMEBOL" for t in [
        "Brazil","Argentina","Uruguay","Colombia","Chile","Peru",
        "Ecuador","Bolivia","Paraguay","Venezuela",
    ]},
    # CONCACAF
    **{t: "CONCACAF" for t in [
        "United States","Mexico","Canada","Costa Rica","Honduras",
        "Guatemala","El Salvador","Jamaica","Panama",
        "Trinidad and Tobago","Curacao","Curaçao","Haiti","Cuba",
    ]},
    # CAF
    **{t: "CAF" for t in [
        "Morocco","Senegal","Nigeria","Ghana","Egypt","Cameroon",
        "Ivory Coast","Algeria","Tunisia","South Africa","Mali",
        "Burkina Faso","Guinea","DR Congo","Zambia","Cape Verde",
        "Tanzania","Uganda","Kenya","Ethiopia","Angola",
    ]},
    # AFC
    **{t: "AFC" for t in [
        "Japan","South Korea","Iran","Saudi Arabia","Australia",
        "Qatar","UAE","Jordan","Iraq","Oman","Bahrain",
        "China","Uzbekistan","Indonesia","Vietnam","Thailand",
    ]},
    # OFC
    **{t: "OFC" for t in ["New Zealand","Fiji","Papua New Guinea"]},
}

def get_conf(team: str) -> str:
    return TEAM_CONF.get(team, "OTHER")

def get_phi(team_home: str, team_away: str) -> float:
    """Dispersión como media geométrica de ambas confederaciones."""
    phi_h = PHI_CONF.get(get_conf(team_home), 2.0)
    phi_a = PHI_CONF.get(get_conf(team_away), 2.0)
    return np.sqrt(phi_h * phi_a)


class NegBinGoalModel:
    """
    Modelo Negative Binomial jerárquico para goles internacionales.
    
    P(X=k | mu, phi) = NB(mu, phi)
    Var(X) = mu + mu²/phi  →  phi grande = menos overdispersión
    
    Parámetros:
      att[i]    : fuerza atacante del equipo i
      def_[j]   : fuerza defensora del equipo j (negativo = buena defensa)
      home_adv  : ventaja de local (≈0.15 para selecciones)
      neutral   : penalización campo neutro (≈-0.15)
    """
    def __init__(self, ridge: float = 0.02):
        self.ridge = ridge
        self.att  = {}
        self.def_ = {}
        self.home_adv  = 0.15
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, elo_ratings: dict = None,
            from_year: int = 2014, xi: float = 0.003,
            verbose: bool = True):
        """Ajusta el modelo NegBin con L-BFGS-B vectorizado + regularización ridge."""

        # Filtrar y ponderar
        ref = df["date"].max()
        df = df[df["date"] >= pd.Timestamp(f"{from_year}-01-01")].copy()
        df = df.dropna(subset=["home_score","away_score"])
        df["home_score"] = df["home_score"].astype(int)
        df["away_score"] = df["away_score"].astype(int)

        # Peso temporal + torneo
        df["days_ago"] = (ref - df["date"]).dt.days
        df["w"] = np.exp(-xi * df["days_ago"])
        df.loc[df["tournament"] == "FIFA World Cup", "w"] *= 1.5
        df.loc[df["tournament"].str.contains("qualification", case=False, na=False), "w"] *= 0.9
        df.loc[df["tournament"] == "Friendly", "w"] *= 0.75
        df.loc[df["days_ago"] <= 7, "w"] *= 1.8

        # Equipos con >= 5 partidos
        counts = pd.concat([df.home_team, df.away_team]).value_counts()
        valid  = set(counts[counts >= 5].index)
        df = df[df.home_team.isin(valid) & df.away_team.isin(valid)].copy()

        teams  = sorted(valid)
        N      = len(teams)
        t_idx  = {t: i for i, t in enumerate(teams)}
        self._teams = teams
        self._tidx  = t_idx

        if verbose:
            print(f"  NegBin: {N} equipos, {len(df):,} partidos")

        # Pre-compute arrays for vectorized log-likelihood
        hi_arr  = df["home_team"].map(t_idx).values.astype(int)
        ai_arr  = df["away_team"].map(t_idx).values.astype(int)
        hg_arr  = df["home_score"].values.astype(float)
        ag_arr  = df["away_score"].values.astype(float)
        w_arr   = df["w"].values.astype(float)
        neutral_arr = df.get("neutral", pd.Series(False, index=df.index)).fillna(False).values.astype(bool)

        # Phi por confederación (vectorizado)
        phi_h_arr = np.array([PHI_CONF.get(get_conf(t), 2.0) for t in df["home_team"]])
        phi_a_arr = np.array([PHI_CONF.get(get_conf(t), 2.0) for t in df["away_team"]])

        M = len(df)

        # Prior ELO
        if elo_ratings:
            mean_elo = np.mean(list(elo_ratings.values()))
            att0 = np.array([0.3 * np.log(elo_ratings.get(t, mean_elo) / mean_elo) for t in teams])
        else:
            att0 = np.zeros(N)

        def0 = np.zeros(N)
        x0   = np.concatenate([att0, def0, [0.15]])

        # Vectorized NB log-likelihood
        def neg_ll(params):
            att  = params[:N]
            def_ = params[N:2*N]
            ha   = params[2*N]

            home_bonus = np.where(neutral_arr, 0.0, ha)
            mu_h = np.exp(att[hi_arr] + def_[ai_arr] + home_bonus)
            mu_a = np.exp(att[ai_arr] + def_[hi_arr])
            mu_h = np.maximum(mu_h, 1e-6)
            mu_a = np.maximum(mu_a, 1e-6)

            # NB log-pmf vectorizado
            ll_h = (gammaln(hg_arr + phi_h_arr) - gammaln(phi_h_arr) - gammaln(hg_arr + 1)
                    + phi_h_arr * np.log(phi_h_arr / (phi_h_arr + mu_h))
                    + hg_arr * np.log(mu_h / (phi_h_arr + mu_h)))

            ll_a = (gammaln(ag_arr + phi_a_arr) - gammaln(phi_a_arr) - gammaln(ag_arr + 1)
                    + phi_a_arr * np.log(phi_a_arr / (phi_a_arr + mu_a))
                    + ag_arr * np.log(mu_a / (phi_a_arr + mu_a)))

            total_ll = -np.sum(w_arr * (ll_h + ll_a))
            total_ll += self.ridge * (np.sum(att**2) + np.sum(def_**2))
            return total_ll

        bounds = [(None,None)]*2*N + [(0.0, 0.5)]

        if verbose:
            print(f"  Optimizando {2*N+1} params (L-BFGS-B vectorizado, maxiter=200)...")
            import time; t0 = time.time()

        res = minimize(neg_ll, x0, method="L-BFGS-B",
                       options={"maxiter": 200, "ftol": 1e-6},
                       bounds=bounds)

        self.att      = {t: res.x[i]   for t, i in t_idx.items()}
        self.def_     = {t: res.x[N+i] for t, i in t_idx.items()}
        self.home_adv = res.x[2*N]
        self.is_fitted = True

        if verbose:
            dt = time.time() - t0
            print(f"  ✓ NegBin ajustado en {dt:.1f}s. home_adv={self.home_adv:.3f}")
            top_att = sorted(self.att.items(), key=lambda x:-x[1])[:5]
            top_def = sorted(self.def_.items(), key=lambda x:x[1])[:5]
            print(f"  Top ataque:  {', '.join(f'{t}({v:.2f})' for t,v in top_att)}")
            print(f"  Top defensa: {', '.join(f'{t}({v:.2f})' for t,v in top_def)}")

        return self

    def expected_goals(self, home: str, away: str,
                       neutral: bool = True,
                       altitude: float = 0) -> tuple:
        """Retorna (lambda_home, lambda_away) — goles esperados."""
        att_h = self.att.get(home, 0.0)
        def_h = self.def_.get(home, 0.0)
        att_a = self.att.get(away, 0.0)
        def_a = self.def_.get(away, 0.0)

        ha    = 0.0 if neutral else self.home_adv
        alt   = altitude * 0.00015  # +1.5% goles por 1000m

        lam = np.exp(att_h + def_a + ha + alt)
        mu  = np.exp(att_a + def_h + alt * 0.5)  # menor efecto visitante
        return lam, mu

    def predict(self, home: str, away: str,
                neutral: bool = True,
                altitude: float = 0,
                max_goals: int = 10) -> dict:
        """
        Retorna distribución completa de probabilidades y mercados.
        """
        lam, mu = self.expected_goals(home, away, neutral, altitude)
        phi_h   = PHI_CONF.get(get_conf(home), 2.0)
        phi_a   = PHI_CONF.get(get_conf(away), 2.0)
        phi     = np.sqrt(phi_h * phi_a)  # mixto

        # Distribución conjunta via Poisson (NegBin → Poisson cuando phi grande)
        # Para selecciones usamos phi directamente
        p_h_goals = np.array([nbinom.pmf(k, phi_h, phi_h/(phi_h+lam)) for k in range(max_goals)])
        p_a_goals = np.array([nbinom.pmf(k, phi_a, phi_a/(phi_a+mu )) for k in range(max_goals)])

        joint = np.outer(p_h_goals, p_a_goals)
        joint /= joint.sum()  # renormalizar

        p_home = float(np.tril(joint, -1).sum())
        p_draw = float(np.trace(joint))
        p_away = float(np.triu(joint,  1).sum())

        total = p_home + p_draw + p_away
        p_home /= total; p_draw /= total; p_away /= total

        # Mercados O/U
        lambda_total = lam + mu
        markets = {}
        for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
            p_over  = float(sum(
                joint[i,j] for i in range(max_goals)
                for j in range(max_goals) if i+j > line
            ))
            markets[f"over_{str(line).replace('.','p')}"]  = round(p_over, 4)
            markets[f"under_{str(line).replace('.','p')}"] = round(1-p_over, 4)

        # BTTS
        p_btts = float(1 - p_h_goals[0] - p_a_goals[0] + joint[0,0])
        markets["btts"] = round(p_btts, 4)

        # Córners: correlación empírica con goles esperados
        # λ_corners ≈ 5.0 + 1.5 * total_goals_expected (calibrado con la media de ~9.0 córners en mundiales)
        lambda_corners = 5.0 + 1.5 * lambda_total
        for line in [8.5, 9.5, 10.5, 11.5]:
            p_c = 1 - sum(poisson.pmf(k, lambda_corners) for k in range(int(line)+1))
            markets[f"corners_over_{str(line).replace('.','p')}"] = round(p_c, 4)
        markets["lambda_corners"] = round(lambda_corners, 2)

        return {
            "home": home, "away": away,
            "p_home": round(p_home, 4),
            "p_draw": round(p_draw, 4),
            "p_away": round(p_away, 4),
            "lambda": round(lam, 3),
            "mu":     round(mu,  3),
            "total_goals": round(lam + mu, 3),
            "phi": round(phi, 2),
            **markets,
        }

    def odds_table(self, home: str, away: str,
                   neutral: bool = True, altitude: float = 0) -> str:
        """Imprime tabla de mercados con momios justos."""
        pred = self.predict(home, away, neutral, altitude)
        lines = [
            f"\n  {'='*60}",
            f"  ⚽ {home}  vs  {away}",
            f"  Goles esperados: {pred['lambda']:.2f} + {pred['mu']:.2f} = {pred['total_goals']:.2f}",
            f"  Overdispersión φ = {pred['phi']:.1f} ({get_conf(home)}/{get_conf(away)})",
            f"  {'='*60}",
            f"  {'Mercado':<25} {'P%':>7}  {'Momio justo':>12}",
            f"  {'-'*47}",
        ]
        markets_show = [
            ("Local gana",     pred["p_home"]),
            ("Empate",         pred["p_draw"]),
            ("Visitante",      pred["p_away"]),
            ("Over 1.5",       pred["over_1p5"]),
            ("Over 2.5",       pred["over_2p5"]),
            ("Over 3.5",       pred["over_3p5"]),
            ("BTTS Sí",        pred["btts"]),
            ("Córners O8.5",   pred["corners_over_8p5"]),
            ("Córners O9.5",   pred["corners_over_9p5"]),
            ("Córners O10.5",  pred["corners_over_10p5"]),
        ]
        for nombre, p in markets_show:
            if p <= 0: continue
            dec = 1/p
            amer = int((dec-1)*100) if dec >= 2 else int(-100/(dec-1))
            sign = "+" if amer >= 0 else ""
            lines.append(f"  {nombre:<25} {p*100:>6.1f}%  {sign}{amer:>11}")
        lines.append(f"  {'='*60}")
        return "\n".join(lines)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT))

    import sys
    df = pd.read_csv(ROOT / "data/international/results.csv", parse_dates=["date"])
    df = df.dropna(subset=["home_score","away_score"])

    # Cargar ELO
    from model.elo_system import ELOSystem
    elo = ELOSystem()
    elo.fit_historical(df, from_year=2000, verbose=False)

    model = NegBinGoalModel(ridge=0.02)
    model.fit(df, elo_ratings=elo.ratings, from_year=2014, verbose=True)

    # Predicciones clave
    for h, a in [
        ("Mexico", "Ecuador"),
        ("Argentina", "Algeria"),
        ("Spain", "Ivory Coast"),
        ("Brazil", "Scotland"),
        ("Germany", "Saudi Arabia"),
        ("France", "Senegal"),
    ]:
        neutral = h not in ["Mexico","United States","Canada"]
        alt = 2240 if h == "Mexico" else 0
        print(model.odds_table(h, a, neutral=neutral, altitude=alt))
