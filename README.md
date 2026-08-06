# Fantasy Football Projection Tool

A Streamlit app that builds fantasy projections from real NFL data and generates a draft strategy tailored to your exact league settings — including superflex.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Keep all the Python files in the same folder. First load downloads a few seasons of stats; everything is cached for 12–24 hours.

## Data sources (all free, no API keys)

| Source | Provides |
|---|---|
| nflverse | Real weekly NFL stats, upcoming-season schedules, official injury reports |
| Sleeper API | Current injury status and body part, news recency |
| FantasyFootballCalculator | Market ADP (PPR / Half / Standard / 2QB-superflex) |
| Open-Meteo | Kickoff temperature, wind and precipitation for outdoor games (~2 weeks out) |
| nflverse draft picks + rosters | Draft capital and current teams, used to project rookies |
| FantasyPros ECR (via ffverse) | Expert consensus ranks, expert disagreement, and weekly expert projections |

Each source degrades gracefully — if one is unreachable, its columns are hidden and the rest of the app works.

## Projection model

1. Weekly stats → per-game rates per player-season
2. Recency-weighted blend across your chosen seasons (last season counts 2× the prior one)
3. **TD regression**: touchdown rates are the noisiest stat year over year, so each player's TD rate is pulled partway toward the position average (adjustable slider, default 30%)
4. Scaled to your projected-games setting (default 16 to bake in injury risk)
5. Filtered to players active last season with enough games for a reliable sample

## Rookies

A stats model can't see rookies — they have no NFL stats. They get a prior built from the one thing
known before week 1: draft capital.

The curve is fitted, not assumed. For every rookie class in the training window the app totals what
that class actually produced and divides by the games it could have played, giving the expected
per-game production of a pick in each range. About 15% of drafted rookies never record a stat, and
because they enter the average as zeros, bust risk is priced in rather than ignored. Thin cells (a
three-rookie sample of top-16 running backs) shrink toward the position's own draft-capital trend
line rather than its mean — shrinking to the mean would drag elite picks down to day-3 production.

Market ADP then separates rookies taken in the same range, since ADP knows the depth chart a rookie
landed in. The **Rookie market weight** slider controls that: 0 is draft capital alone, 1 follows
the market. Rookies carry an **R** in the rankings and flow into everything downstream — draft
strategy, the mock draft pool, and weekly projections.

Expect roughly one rookie in the top 50 and a handful inside the top 200 — which is about where
rookie classes actually land. Treat these as a wide range rather than a number.

## Expert consensus

FantasyPros ECR, mirrored by ffverse — free and keyless, using the superflex board automatically
when your league is superflex. About 95% of the consensus top 100 matches the board by name; the
misses are rookies, which the draft-capital model covers.

**It is deliberately not the backbone of the projections.** Consensus and ADP are close to the same
signal, since drafters read these rankings. Build projections out of ECR and the "value vs ADP"
column compares the market against itself, the mock draft's projections-vs-ADP slider has the market
on both ends, and none of it respects your scoring settings. The **Trust experts** slider (default
0.3) controls how far consensus moves the board order, and your own rank stays visible in its own
column at every setting. Rookies lean on the experts regardless of the slider — a draft-capital
prior is an average over a pick range, while the experts watched that specific player's camp.

The **vs Experts** tab carries the parts that add information rather than re-weighting what you
already have:

- **Where you disagree**, in both directions, filtered to players someone would actually draft — a
  200-place gap between rank 450 and rank 250 is not a disagreement anyone acts on
- **Expert certainty**, measured as a player's spread against the typical spread of their ECR
  neighbours. Raw spread just re-ranks the board, and dividing by rank makes every elite player look
  contested. Normalised properly, it separates a genuine argument (a QB ranked anywhere from 27th to
  98th) from a settled one (a top-30 QB with a spread of two places)
- **Consensus movement**, the closest thing here to a breakout or fade signal. FantasyPros only
  populates this on some scrapes, so it may be empty in the preseason and fills in during the season

In-season, the Weekly tab can compare your weekly numbers against FantasyPros' own weekly
projections, sorted by disagreement. That file only refreshes during the season, so it shows a
staleness warning out of season.

## League-aware valuation

- **Superflex**: superflex slots route ~80% of their demand to QB, which drops QB replacement level and surfaces QBs much higher in the rankings — exactly how real superflex drafts behave
- **Scoring**: full custom scoring including a TE-premium reception bonus; every downstream number (rankings, VOR, tiers, defense ratings, strategy) uses *your* scoring
- **VOR**: value over the replacement-level player given your league size and roster

## Tabs

- **Rankings** — VOR-ranked board with ADP, value-vs-ADP, injury status, and SOS columns
- **Mock Draft** — simulate your draft from your real slot, either start-to-finish or pick-by-pick
- **Weekly** — week-by-week projections driven by matchup, game script, injuries and weather
- **vs Experts** — where your model and expert consensus disagree, and how certain the experts are
- **Draft Strategy** — auto-generated from your settings: position priority by scarcity, superflex QB timing, tier cliffs by round, market values and reaches vs ADP, injury flags in your draft range
- **News & Injuries** — current status (Sleeper), weeks missed last season (nflverse), and a news search link per player
- **Compare / Positions** — head-to-head charts and tier visualizations with the replacement line
- **Teams** — each team's pass/run rate and a position-by-position strength-of-schedule heatmap
- **Cheat Sheet** — one CSV with everything, plus your strategy notes as a Markdown download

## Mock draft

The other managers draft off a blended board — market ADP mixed with your own projections — plus
Gaussian noise and roster-need logic. Two sliders control the room: how much it trusts ADP vs your
numbers, and how unpredictable managers are. Because the projections side of the board is already
superflex- and scoring-aware, a superflex league produces a superflex-shaped draft with no special
casing (QBs go in the first three rounds instead of the fifth).

- **Simulate the whole draft** — your roster and best legal starting lineup, a draft grade against
  the other teams, the full board, and every pick with its reach/value vs ADP
- **Draft interactively** — the sim pauses on every one of your picks; take the player you want,
  auto-pick, or reset
- **Monte Carlo availability** — run 10–200 drafts and see the probability each player is still on
  the board at each of *your* picks. Anyone near 50% is the real decision point

## Weekly projections

Each player's season per-game average is moved week to week by multipliers you can see and turn off:

| Factor | Source |
|---|---|
| Matchup | Opponent's fantasy points allowed to that position vs league average |
| Game script | Vegas spread — underdogs throw more, favorites run more |
| Implied total | Vegas total + spread → expected points for that offense |
| Team pass-rate outlook | Your own editable pass-rate projection per team (new OC or QB) |
| Injuries | Sleeper status for the player, applied to week 1 or all weeks |
| Vacated volume | Points freed up by injured teammates at the same position, redistributed |
| Weather | Wind, cold and precipitation at kickoff, outdoor games only |
| Home / away | Small fixed edge |

Weather and game-script effects are applied against each player's passing/receiving vs rushing
split, so wind hurts a pocket passer far more than a goal-line back. Byes come straight from the
schedule and show as zero. Every factor falls back to 1.0 when its source is unavailable.

## In-season mode

Nothing needs editing when the season starts. The app reads the live schedule, works out which week
is current, and offers an **in-season toggle** in the sidebar from the moment week 1 kicks off. With
it on:

- The season slider extends to the current year, so games being played now enter the projection
  blend. Partial seasons are handled correctly — the model weights by games played, so three weeks
  of data carries three weeks of influence
- Matchup ratings blend current-year defenses into last year's, weighted `weeks_played /
  (weeks_played + 6)` — about 45% current-year at week 6, 74% by week 17. Six games of defensive
  data is too noisy to trust on its own
- Projected games defaults to the weeks left on the schedule, so season totals become
  rest-of-season totals
- **The board updates itself every week.** The preseason projection becomes a prior, and each
  player's actual production this season is blended in with weight `games / (games + k)` — half
  after `k` games (default 4, adjustable). Nothing needs rebuilding; the ranking re-weights on its
  own as games are played
- Rookies converge fastest, which is the point: a player whose entire projection was a draft-capital
  guess is running mostly on real production within a month
- Players who appear this season but weren't on the preseason board — undrafted breakouts, late
  signings — are added on their current production once they have two games
- The rankings show **GP**, **Live %** (how much of the number is this season) and **Δ vs preseason**
- The Weekly tab opens on the current week and hides the weeks already played
- A **rest-of-season table** totals every weekly adjustment across your remaining games — the
  in-season equivalent of the Rankings tab, already accounting for byes, matchups and injuries
- Players who haven't appeared yet this season stay on the board instead of being filtered out as
  inactive

Toggle it off at any point to get the preseason draft board back.

## Honest limitations

Stat-based projections can't see offseason trades or coaching changes — the editable pass-rate
outlook in the Weekly tab is the one lever for the latter. Rookie projections are draft-capital
averages, not scouting: they cannot tell you which specific rookie breaks out, only what a pick in
that range is typically worth. A rookie's floor and ceiling are much further apart than a veteran's,
and the single number in the rankings hides that.

Expert consensus is a community scrape of FantasyPros rather than an official API: the refresh
cadence isn't yours to control, matching is by player name, and the weekly file is a single
current-week snapshot rather than a history. Their weekly point projections assume standard PPR, so
in an unusual scoring format compare the shape of a disagreement rather than the raw gap. SOS uses last season's defenses, and team tendencies can shift with new coordinators. The Teams and News tabs exist precisely so you can sanity-check the numbers against the real-world
situation. Weather forecasts only reach about two weeks out, so late-season weather columns are
empty until the season is underway, and defensive matchup ratings are last season's — week 14 is a
sketch, not a lineup decision.
