"""
Data sources for the fantasy projection app.
All loaders degrade gracefully — if a source is unreachable, they return None
and the app simply hides that feature.
"""

import re

import numpy as np
import pandas as pd
import requests
import streamlit as st

POSITIONS = ["QB", "RB", "WR", "TE"]

STAT_COLUMNS = [
    "pass_yds", "pass_td", "interceptions",
    "rush_yds", "rush_td",
    "receptions", "rec_yds", "rec_td",
    "fumbles",
    "pass_att", "carries", "targets",
]


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def name_key(s: pd.Series) -> pd.Series:
    """Normalize player names for cross-source matching."""
    return (
        s.astype(str).str.lower()
        .str.replace(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", regex=True)
        .str.replace(r"[^a-z]", "", regex=True)
    )


# ----------------------------------------------------------------------------
# nflverse: weekly stats → player-season aggregates (keeps opponent for defense ratings)
# ----------------------------------------------------------------------------
@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def load_nflverse_weekly(seasons: tuple[int, ...]) -> pd.DataFrame:
    import nflreadpy as nfl

    weekly = nfl.load_player_stats(list(seasons)).to_pandas()
    if "season_type" in weekly.columns:
        weekly = weekly[weekly["season_type"] == "REG"]

    name_c = pick_col(weekly, ["player_display_name", "player_name"])
    team_c = pick_col(weekly, ["team", "recent_team"])
    int_c = pick_col(weekly, ["passing_interceptions", "interceptions"])
    opp_c = pick_col(weekly, ["opponent_team", "opponent"])

    rename = {
        name_c: "player",
        "passing_yards": "pass_yds",
        "passing_tds": "pass_td",
        int_c: "interceptions",
        "rushing_yards": "rush_yds",
        "rushing_tds": "rush_td",
        "receiving_yards": "rec_yds",
        "receiving_tds": "rec_td",
        "attempts": "pass_att",
    }
    weekly = weekly.rename(columns={k: v for k, v in rename.items() if k})
    weekly["team"] = weekly[team_c] if team_c else "—"
    weekly["opponent"] = weekly[opp_c] if opp_c else None

    fparts = [c for c in
              ["sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost"]
              if c in weekly.columns]
    weekly["fumbles"] = weekly[fparts].sum(axis=1) if fparts else 0.0

    for col in STAT_COLUMNS:
        if col not in weekly.columns:
            weekly[col] = 0.0
        weekly[col] = pd.to_numeric(weekly[col], errors="coerce").fillna(0.0)

    if "position" not in weekly.columns:
        rosters = nfl.load_rosters(list(seasons)).to_pandas()
        rid = pick_col(rosters, ["gsis_id", "player_id"])
        rosters = rosters[[rid, "position"]].drop_duplicates(rid)
        weekly = weekly.merge(
            rosters.rename(columns={rid: "player_id"}), on="player_id", how="left"
        )

    weekly["position"] = weekly["position"].astype(str).str.upper().str.strip()
    weekly = weekly[weekly["position"].isin(POSITIONS)]
    return weekly


def aggregate_seasons(weekly: pd.DataFrame) -> pd.DataFrame:
    weekly = weekly.sort_values(["season", "week"])  # so team=("team", "last") is the latest team
    return (
        weekly.groupby(["player_id", "player", "position", "season"])
        .agg(
            games=("week", "nunique"),
            team=("team", "last"),
            **{c: (c, "sum") for c in STAT_COLUMNS},
        )
        .reset_index()
    )


# ----------------------------------------------------------------------------
# Defense ratings: fantasy points allowed per game, by position, latest season
# ----------------------------------------------------------------------------
def defense_ratings(weekly: pd.DataFrame, season: int, score_fn) -> pd.DataFrame | None:
    w = weekly[(weekly["season"] == season) & weekly["opponent"].notna()].copy()
    if w.empty:
        return None
    w["fpts"] = score_fn(w)
    per_game = (
        w.groupby(["opponent", "position", "week"])["fpts"].sum().reset_index()
        .groupby(["opponent", "position"])["fpts"].mean().reset_index()
        .rename(columns={"opponent": "def_team", "fpts": "fpts_allowed_pg"})
    )
    # Percentile within position: 100 = most generous defense (easiest matchup)
    per_game["def_pctl"] = per_game.groupby("position")["fpts_allowed_pg"].rank(pct=True) * 100
    return per_game


# ----------------------------------------------------------------------------
# Team offensive philosophy: pass vs run rate, latest season
# ----------------------------------------------------------------------------
def team_tendencies(weekly: pd.DataFrame, season: int) -> pd.DataFrame:
    w = weekly[weekly["season"] == season]
    t = w.groupby("team").agg(pass_att=("pass_att", "sum"), carries=("carries", "sum")).reset_index()
    t["plays"] = t["pass_att"] + t["carries"]
    t = t[t["plays"] > 100]
    t["pass_rate"] = (t["pass_att"] / t["plays"] * 100).round(1)
    return t[["team", "pass_rate", "pass_att", "carries"]].sort_values("pass_rate", ascending=False)


# ----------------------------------------------------------------------------
# Strength of schedule for the upcoming season
# ----------------------------------------------------------------------------
@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def load_schedule_opponents(season: int) -> pd.DataFrame | None:
    """Map team → list of opponents for the given season. None if not released yet."""
    import nflreadpy as nfl

    try:
        sched = nfl.load_schedules([season]).to_pandas()
    except Exception:
        return None
    if sched.empty:
        return None
    if "game_type" in sched.columns:  # `sched.get(...)` returns a scalar when absent → bad mask
        sched = sched[sched["game_type"] == "REG"]
    home = sched[["home_team", "away_team"]].rename(columns={"home_team": "team", "away_team": "opponent"})
    away = sched[["away_team", "home_team"]].rename(columns={"away_team": "team", "home_team": "opponent"})
    return pd.concat([home, away], ignore_index=True)


def compute_sos(schedule: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """Average opponent generosity per (team, position). Higher = easier schedule."""
    merged = schedule.merge(ratings, left_on="opponent", right_on="def_team")
    sos = (
        merged.groupby(["team", "position"])["def_pctl"].mean().reset_index()
        .rename(columns={"def_pctl": "sos_pctl"})
    )
    sos["sos_pctl"] = sos["sos_pctl"].round(0)
    return sos


# ----------------------------------------------------------------------------
# Sleeper: current injury status + news recency (free, no key)
# ----------------------------------------------------------------------------
@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def load_sleeper_injuries() -> pd.DataFrame | None:
    try:
        r = requests.get("https://api.sleeper.app/v1/players/nfl", timeout=45)
        r.raise_for_status()
        players = r.json()
    except Exception:
        return None

    rows = []
    for p in players.values():
        if not isinstance(p, dict) or p.get("position") not in POSITIONS:
            continue
        rows.append({
            "name_key_": re.sub(r"[^a-z]", "", str(p.get("full_name", "")).lower()),
            "position": p.get("position"),
            "injury_status": p.get("injury_status"),
            "injury_part": p.get("injury_body_part"),
            "injury_notes": p.get("injury_notes"),
            "news_updated": p.get("news_updated"),
            "sleeper_age": p.get("age"),
            "years_exp": p.get("years_exp"),
        })
    df = pd.DataFrame(rows)
    df["injury"] = np.where(
        df["injury_status"].notna(),
        df["injury_status"].astype(str)
        + np.where(df["injury_part"].notna(), " (" + df["injury_part"].astype(str) + ")", ""),
        None,
    )
    df["news_updated"] = pd.to_datetime(df["news_updated"], unit="ms", errors="coerce")
    return df.drop_duplicates(["name_key_", "position"])


# ----------------------------------------------------------------------------
# nflverse: last-season injury report history
# ----------------------------------------------------------------------------
@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def load_injury_history(season: int) -> pd.DataFrame | None:
    import nflreadpy as nfl

    try:
        inj = nfl.load_injuries([season]).to_pandas()
    except Exception:
        return None
    if inj.empty:
        return None
    pid = pick_col(inj, ["gsis_id", "player_id"])
    status = pick_col(inj, ["report_status", "injury_status"])
    if not pid or not status:
        return None
    inj["was_out"] = inj[status].astype(str).str.lower().isin(["out", "doubtful", "injured reserve", "ir"])
    hist = (
        inj.groupby(pid)
        .agg(injury_weeks=("week", "nunique"), weeks_out=("was_out", "sum"))
        .reset_index()
        .rename(columns={pid: "player_id"})
    )
    return hist


# ----------------------------------------------------------------------------
# ADP from FantasyFootballCalculator (free, no key)
# ----------------------------------------------------------------------------
@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def load_adp(scoring_format: str, teams: int, year: int) -> pd.DataFrame | None:
    fmt = {"PPR": "ppr", "Half PPR": "half-ppr", "Standard": "standard",
           "Superflex": "2qb"}.get(scoring_format, "ppr")
    # FFC only publishes 8/10/12/14-team ADP — snap to the nearest supported size
    teams = min((8, 10, 12, 14), key=lambda t: abs(t - int(teams)))
    url = f"https://fantasyfootballcalculator.com/api/v1/adp/{fmt}?teams={teams}&year={year}"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json().get("players", [])
    except Exception:
        return None
    if not data:
        return None
    adp = pd.DataFrame(data)
    if "name" not in adp.columns or "adp" not in adp.columns:
        return None
    adp["name_key_"] = name_key(adp["name"])
    adp = adp[adp["position"].isin(POSITIONS)]
    return adp[["name_key_", "position", "adp"]].drop_duplicates(["name_key_", "position"])


# ----------------------------------------------------------------------------
# Week-by-week schedule + venue info (for the Weekly Projections tab)
# ----------------------------------------------------------------------------
# Approximate stadium coordinates, used only to pull a weather forecast.
STADIUM_COORDS = {
    "ARI": (33.53, -112.26), "ATL": (33.76, -84.40), "BAL": (39.28, -76.62),
    "BUF": (42.77, -78.79), "CAR": (35.23, -80.85), "CHI": (41.86, -87.62),
    "CIN": (39.10, -84.52), "CLE": (41.51, -81.70), "DAL": (32.75, -97.09),
    "DEN": (39.74, -105.02), "DET": (42.34, -83.05), "GB": (44.50, -88.06),
    "HOU": (29.68, -95.41), "IND": (39.76, -86.16), "JAX": (30.32, -81.64),
    "KC": (39.05, -94.48), "LA": (33.95, -118.34), "LAC": (33.95, -118.34),
    "LV": (36.09, -115.18), "MIA": (25.96, -80.24), "MIN": (44.97, -93.26),
    "NE": (42.09, -71.26), "NO": (29.95, -90.08), "NYG": (40.81, -74.07),
    "NYJ": (40.81, -74.07), "PHI": (39.90, -75.17), "PIT": (40.45, -80.02),
    "SEA": (47.60, -122.33), "SF": (37.40, -121.97), "TB": (27.98, -82.50),
    "TEN": (36.17, -86.77), "WAS": (38.91, -76.86),
}

# Neutral-site / international venues, matched on the schedule's stadium name.
NEUTRAL_VENUES = {
    "melbourne": (-37.82, 144.98), "maracana": (-22.91, -43.23),
    "tottenham": (51.60, -0.07), "wembley": (51.56, -0.28),
    "stade de france": (48.92, 2.36), "bernabeu": (40.45, -3.69),
    "munich": (48.22, 11.62), "banorte": (19.30, -99.15), "azteca": (19.30, -99.15),
}

# Teams whose home games are indoors (or under a closed/fixed roof most weeks).
INDOOR_TEAMS = {"ARI", "ATL", "DAL", "DET", "HOU", "IND", "LA", "LAC", "LV", "MIN", "NO"}


def _venue_coords(home_team: str, stadium: str) -> tuple[float | None, float | None]:
    s = str(stadium).lower()
    for key, coords in NEUTRAL_VENUES.items():
        if key in s:
            return coords
    return STADIUM_COORDS.get(home_team, (None, None))


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def load_schedule_games(season: int) -> pd.DataFrame | None:
    """One row per team-game for the regular season, with venue + kickoff info.

    Columns: team, opponent, week, is_home, game_id, kickoff, roof, indoor,
             stadium, lat, lon, total_line, spread_line, implied_total
    """
    import nflreadpy as nfl

    try:
        sched = nfl.load_schedules([season]).to_pandas()
    except Exception:
        return None
    if sched is None or sched.empty:
        return None
    if "game_type" in sched.columns:
        sched = sched[sched["game_type"] == "REG"].copy()
    if sched.empty:
        return None

    for c in ["stadium", "roof", "location", "total_line", "spread_line", "gametime"]:
        if c not in sched.columns:
            sched[c] = np.nan

    kickoff = pd.to_datetime(
        sched["gameday"].astype(str) + " " + sched["gametime"].fillna("13:00").astype(str),
        errors="coerce",
    )
    sched["kickoff"] = kickoff

    rows = []
    for is_home, (team_c, opp_c) in [(True, ("home_team", "away_team")),
                                     (False, ("away_team", "home_team"))]:
        part = pd.DataFrame({
            "team": sched[team_c],
            "opponent": sched[opp_c],
            "week": sched["week"].astype(int),
            "is_home": is_home and (sched["location"].fillna("Home") == "Home"),
            "game_id": sched.get("game_id", pd.Series(range(len(sched)))),
            "kickoff": sched["kickoff"],
            "roof": sched["roof"],
            "stadium": sched["stadium"],
            "home_team": sched["home_team"],
            "total_line": pd.to_numeric(sched["total_line"], errors="coerce"),
            "spread_line": pd.to_numeric(sched["spread_line"], errors="coerce"),
        })
        # nflverse spread_line is from the home team's perspective (positive = home favored)
        part["team_spread"] = sched["spread_line"] if is_home else -sched["spread_line"]
        rows.append(part)

    g = pd.concat(rows, ignore_index=True)
    g["team_spread"] = pd.to_numeric(g["team_spread"], errors="coerce")
    g["implied_total"] = g["total_line"] / 2 + g["team_spread"] / 2

    roof = g["roof"].astype("string").str.lower().fillna("")
    g["indoor"] = roof.isin(["dome", "closed"]) | (
        (roof == "") & g["home_team"].isin(INDOOR_TEAMS)
    )
    coords = g.apply(lambda r: _venue_coords(r["home_team"], r["stadium"]), axis=1)
    g["lat"] = [c[0] for c in coords]
    g["lon"] = [c[1] for c in coords]
    return g.sort_values(["team", "week"]).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Weather: Open-Meteo forecast (free, no key). Only covers ~16 days out, so
# games further away simply come back without a forecast and get no adjustment.
# ----------------------------------------------------------------------------
@st.cache_data(ttl=60 * 60 * 3, show_spinner=False)
def load_game_weather(games: pd.DataFrame) -> pd.DataFrame | None:
    """Forecast temp (°F), wind (mph) and precipitation chance at kickoff.

    Returns one row per game_id that is outdoors, within the forecast window,
    and has known coordinates. None if nothing is fetchable.
    """
    if games is None or games.empty:
        return None

    out_games = games[
        (~games["indoor"]) & games["lat"].notna() & games["kickoff"].notna()
    ].drop_duplicates("game_id").copy()
    if out_games.empty:
        return None

    now = pd.Timestamp.utcnow().tz_localize(None).normalize()
    horizon = now + pd.Timedelta(days=15)
    out_games = out_games[(out_games["kickoff"] >= now) & (out_games["kickoff"] <= horizon)]
    if out_games.empty:
        return None

    # One batched request: Open-Meteo accepts comma-separated coordinate lists.
    lats = ",".join(f"{v:.2f}" for v in out_games["lat"])
    lons = ",".join(f"{v:.2f}" for v in out_games["lon"])
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lats, "longitude": lons,
                "hourly": "temperature_2m,wind_speed_10m,precipitation_probability",
                "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
                "start_date": out_games["kickoff"].min().strftime("%Y-%m-%d"),
                "end_date": out_games["kickoff"].max().strftime("%Y-%m-%d"),
                "timezone": "UTC",
            },
            timeout=30,
        )
        r.raise_for_status()
        payload = r.json()
    except Exception:
        return None

    blocks = payload if isinstance(payload, list) else [payload]
    if len(blocks) != len(out_games):
        return None

    rows = []
    for (_, game), block in zip(out_games.iterrows(), blocks):
        hourly = block.get("hourly", {})
        times = pd.to_datetime(pd.Series(hourly.get("time", [])), errors="coerce")
        if times.empty:
            continue
        idx = int((times - game["kickoff"]).abs().idxmin())

        def at(key):
            vals = hourly.get(key) or []
            return float(vals[idx]) if idx < len(vals) and vals[idx] is not None else np.nan

        rows.append({
            "game_id": game["game_id"],
            "temp_f": at("temperature_2m"),
            "wind_mph": at("wind_speed_10m"),
            "precip_pct": at("precipitation_probability"),
        })
    return pd.DataFrame(rows) if rows else None
