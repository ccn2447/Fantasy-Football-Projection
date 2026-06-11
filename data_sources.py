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
    sched = sched[sched.get("game_type", "REG") == "REG"]
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
