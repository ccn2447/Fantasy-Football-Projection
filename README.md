# Fantasy Football Projection Tool

A Streamlit app that builds fantasy projections from real NFL data and generates a draft strategy tailored to your exact league settings — including superflex.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Keep all four Python/CSV files in the same folder. First load downloads a few seasons of stats; everything is cached for 12–24 hours.

## Data sources (all free, no API keys)

| Source | Provides |
|---|---|
| nflverse | Real weekly NFL stats, upcoming-season schedules, official injury reports |
| Sleeper API | Current injury status and body part, news recency |
| FantasyFootballCalculator | Market ADP (PPR / Half / Standard / 2QB-superflex) |

Each source degrades gracefully — if one is unreachable, its columns are hidden and the rest of the app works.

## Projection model

1. Weekly stats → per-game rates per player-season
2. Recency-weighted blend across your chosen seasons (last season counts 2× the prior one)
3. **TD regression**: touchdown rates are the noisiest stat year over year, so each player's TD rate is pulled partway toward the position average (adjustable slider, default 30%)
4. Scaled to your projected-games setting (default 16 to bake in injury risk)
5. Filtered to players active last season with enough games for a reliable sample

## League-aware valuation

- **Superflex**: superflex slots route ~80% of their demand to QB, which drops QB replacement level and surfaces QBs much higher in the rankings — exactly how real superflex drafts behave
- **Scoring**: full custom scoring including a TE-premium reception bonus; every downstream number (rankings, VOR, tiers, defense ratings, strategy) uses *your* scoring
- **VOR**: value over the replacement-level player given your league size and roster

## Tabs

- **Rankings** — VOR-ranked board with ADP, value-vs-ADP, injury status, and SOS columns
- **Draft Strategy** — auto-generated from your settings: position priority by scarcity, superflex QB timing, tier cliffs by round, market values and reaches vs ADP, injury flags in your draft range
- **News & Injuries** — current status (Sleeper), weeks missed last season (nflverse), and a news search link per player
- **Compare / Positions** — head-to-head charts and tier visualizations with the replacement line
- **Teams** — each team's pass/run rate and a position-by-position strength-of-schedule heatmap
- **Cheat Sheet** — one CSV with everything, plus your strategy notes as a Markdown download

## Honest limitations

Stat-based projections can't see offseason trades, rookies (no NFL stats yet), or coaching changes. SOS uses last season's defenses, and team tendencies can shift with new coordinators. The Teams and News tabs exist precisely so you can sanity-check the numbers against the real-world situation.
