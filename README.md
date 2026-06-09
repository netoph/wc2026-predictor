# 🏆 WC 2026 Predictor

Autonomous prediction and betting simulation system for the 2026 FIFA World Cup.

## Stack

| Component | Description |
|---|---|
| **ELO Rating** | Dynamic rating system for 323 national teams (2000-2026) |
| **Negative Binomial** | Goal modeling with overdispersion, 275 attack/defense params |
| **Temperature Scaling** | Probability calibration (Guo et al., 2017) |
| **Kelly Criterion** | Optimal bet sizing with market-specific confidence |
| **ESPN Scraper** | Real-time scores and DraftKings odds every 60s |
| **Auto-EV Scanner** | Smart bot that bets when model edge > market threshold |

## Blind Test KPIs (875 matches, never seen)

| Metric | Value |
|---|---|
| 1X2 Hit Rate | **65.4%** |
| LogLoss | **0.791** (naïve: 1.099) |
| O/U 2.5 Hit | **65.5%** |
| BTTS Hit | **64.5%** |
| xG MAE | **1.276** |

## Quick Start

```bash
# Local
cd dashboard && python3 -c "import uvicorn; uvicorn.run('app:app', host='0.0.0.0', port=8026)"

# Docker (recommended for server)
docker compose up -d

# Access
open http://localhost:8026
```

## Architecture

```
wc2026/
├── model/
│   ├── elo_system.py          # ELO rating with variable K-factor
│   ├── negative_binomial.py   # NegBin goal model (vectorized L-BFGS-B)
│   ├── calibration.py         # Temperature scaling
│   └── evaluate.py            # Train/test/blind evaluation
├── betting/
│   ├── kelly.py               # Kelly Criterion bettor
│   ├── auto_scanner.py        # Smart EV scanner with market priorities
│   └── hit_tracker.py         # Prediction accuracy tracker
├── scrapers/
│   └── espn_live.py           # ESPN + DraftKings odds scraper
├── dashboard/
│   ├── app.py                 # FastAPI backend
│   └── static/index.html      # Single-page dashboard
├── data/
│   ├── international/         # Historical results (49K+ matches)
│   └── wc2026/                # Fixtures and ELO ratings
├── Dockerfile
├── docker-compose.yml         # Auto-restart + persistence
└── fly.toml                   # Fly.io deployment config
```

## Resilience

- **Auto-restart**: `docker compose` with `restart: always`
- **State persistence**: Bot bets, tracker, and scanner state saved to `data/state/` after every scrape cycle
- **Health checks**: Docker health endpoint at `/api/scraper/status`
- **Graceful shutdown**: State saved on SIGTERM

## References

1. Dixon & Coles (1997). JRSS-C, 46(2), 265-280
2. Elo (1978). The Rating of Chessplayers
3. Guo et al. (2017). ICML — On Calibration of Modern Neural Networks
4. Kelly (1956). Bell System Technical Journal, 35(4), 917-926

## License

MIT
