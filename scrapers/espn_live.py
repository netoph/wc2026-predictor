"""
wc2026/scrapers/espn_live.py
═══════════════════════════════════════════════════════════════
Scraper automático de ESPN para marcadores en vivo + momios.
Corre como background task en FastAPI cada 60 segundos.
═══════════════════════════════════════════════════════════════
"""
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import time

ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
ROOT = Path(__file__).resolve().parent.parent

# Mapeo de nombres ESPN → nombres en nuestro fixtures.csv
ESPN_TO_LOCAL = {
    "Czechia": "Czech Republic",
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "Türkiye": "Turkey",
    "Curacao": "Curaçao",
    "Korea DPR": "North Korea",
    "USA": "United States",
    "Congo DR": "DR Congo",
    "Cape Verde Islands": "Cape Verde",
    "Saudi": "Saudi Arabia",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
}

def normalize_team(name: str) -> str:
    return ESPN_TO_LOCAL.get(name, name)


def fetch_espn_scores() -> list:
    """Fetch live/recent scores + odds from ESPN API."""
    try:
        resp = requests.get(ESPN_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  ⚠️  ESPN fetch error: {e}")
        return []

    matches = []
    for event in data.get("events", []):
        comp = event.get("competitions", [{}])[0]
        status_obj = comp.get("status", {})
        status_type = status_obj.get("type", {})
        state = status_type.get("state", "pre")  # pre, in, post
        clock = status_obj.get("displayClock", "0'")

        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            continue

        # ESPN: competitors[0]=home, competitors[1]=away
        home_data = competitors[0]
        away_data = competitors[1]
        home_name = normalize_team(home_data.get("team", {}).get("displayName", ""))
        away_name = normalize_team(away_data.get("team", {}).get("displayName", ""))
        home_score = int(home_data.get("score", "0") or "0")
        away_score = int(away_data.get("score", "0") or "0")

        # Parse minute from clock
        minute = 0
        try:
            clock_str = clock.replace("'", "").strip()
            if clock_str:
                minute = int(clock_str)
        except:
            minute = 0

        # Map state to our status
        if state == "in":
            match_status = "live"
        elif state == "post":
            match_status = "finished"
        else:
            match_status = "scheduled"

        # Extract odds (DraftKings via ESPN)
        odds_data = {}
        for odd in comp.get("odds", []):
            ml = odd.get("moneyline", {})
            if ml:
                home_ml = ml.get("home", {}).get("close", {}).get("odds")
                away_ml = ml.get("away", {}).get("close", {}).get("odds")
                draw_ml = ml.get("draw", {}).get("close", {}).get("odds")
                if home_ml: odds_data["ml_home"] = home_ml
                if away_ml: odds_data["ml_away"] = away_ml
                if draw_ml: odds_data["ml_draw"] = draw_ml

            total = odd.get("total", {})
            if total:
                over_odds = total.get("over", {}).get("close", {}).get("odds")
                under_odds = total.get("under", {}).get("close", {}).get("odds")
                if over_odds:  odds_data["over25"]  = over_odds
                if under_odds: odds_data["under25"] = under_odds

        matches.append({
            "home": home_name,
            "away": away_name,
            "home_goals": home_score,
            "away_goals": away_score,
            "minute": minute,
            "status": match_status,
            "clock": clock,
            "venue": comp.get("venue", {}).get("fullName", ""),
            "odds": odds_data,
        })

    return matches


def update_fixtures_from_espn(matches: list) -> dict:
    """Actualiza fixtures.csv con los datos de ESPN."""
    fixtures_path = ROOT / "data/wc2026/fixtures.csv"
    if not fixtures_path.exists():
        return {"updated": 0, "error": "fixtures.csv not found"}

    df = pd.read_csv(fixtures_path)
    updated = 0

    for m in matches:
        if m["status"] == "scheduled":
            continue  # Solo actualizar partidos en vivo o finalizados

        mask = (
            df["home"].str.strip() == m["home"]
        ) & (
            df["away"].str.strip() == m["away"]
        )

        if mask.any():
            idx = df[mask].index[0]
            df.at[idx, "home_goals"] = m["home_goals"]
            df.at[idx, "away_goals"] = m["away_goals"]
            df.at[idx, "minute"]     = m["minute"]
            df.at[idx, "status"]     = m["status"]
            updated += 1

    if updated > 0:
        df.to_csv(fixtures_path, index=False)

    return {"updated": updated, "total_fetched": len(matches)}


def fetch_and_update() -> dict:
    """Fetch from ESPN and update local fixtures. Returns summary."""
    matches = fetch_espn_scores()
    result = update_fixtures_from_espn(matches)

    # Log odds for EV analysis
    odds_log = []
    for m in matches:
        if m.get("odds"):
            odds_log.append({
                "timestamp": datetime.now().isoformat(),
                "home": m["home"], "away": m["away"],
                "status": m["status"], "minute": m["minute"],
                "score": f"{m['home_goals']}-{m['away_goals']}",
                **m["odds"],
            })

    if odds_log:
        odds_path = ROOT / "data" / "odds" / "espn_odds_live.csv"
        odds_path.parent.mkdir(parents=True, exist_ok=True)
        df_odds = pd.DataFrame(odds_log)
        if odds_path.exists():
            df_odds.to_csv(odds_path, mode="a", header=False, index=False)
        else:
            df_odds.to_csv(odds_path, index=False)

    return {**result, "odds_logged": len(odds_log), "matches": matches}


if __name__ == "__main__":
    print("Fetching ESPN WC2026 scores...")
    result = fetch_and_update()
    print(f"  Updated: {result['updated']} | Fetched: {result['total_fetched']} | Odds logged: {result['odds_logged']}")
    for m in result.get("matches", []):
        status_icon = "🔴" if m["status"]=="live" else "✅" if m["status"]=="finished" else "⏳"
        odds_str = ""
        if m.get("odds"):
            o = m["odds"]
            odds_str = f" | ML: {o.get('ml_home','?')}/{o.get('ml_draw','?')}/{o.get('ml_away','?')} | O/U2.5: {o.get('over25','?')}/{o.get('under25','?')}"
        print(f"  {status_icon} {m['home']:<20} {m['home_goals']}-{m['away_goals']} {m['away']:<20} {m['clock']:>5}{odds_str}")
