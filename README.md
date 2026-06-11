# Fantasy Football Projection Tool

A Streamlit app that pulls **real NFL stats automatically from nflverse** (free, public, no API key) and turns them into fantasy projections, value-over-replacement rankings, tiers, and a draft cheat sheet. No CSV needed.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at http://localhost:8501. The first load downloads a few seasons of stats (takes a moment), then it's cached for 24 hours.

## How projections work

1. Downloads weekly player stats for the seasons you select (sidebar slider)
2. Aggregates to per-game rates for each player-season
3. Combines seasons with recency weighting (last season counts 2x the one before, 4x the one before that, also weighted by games played)
4. Multiplies by your projected games setting (default 16 to bake in typical injury risk)
5. Filters to players active in the most recent season with enough games for a reliable sample

## Features

- **Live data** — real stats from nflverse, refreshed daily; sample-data mode works offline
- **Custom scoring** — PPR / Half PPR / Standard presets or fully custom values
- **VOR rankings** — value over the replacement-level player, based on your league size and roster (flex demand split 45/45/10 across RB/WR/TE)
- **Tiers** — gap-based tiering shows where the talent cliffs are at each position
- **Compare players** and **position scarcity** views
- **Cheat sheet export** — CSV download for draft day

## Limitations worth knowing

These are stat-based baseline projections. They don't know about offseason trades, rookies (no NFL stats yet), new coaching schemes, or suspensions. Treat them as a starting point and adjust with your own judgment.
