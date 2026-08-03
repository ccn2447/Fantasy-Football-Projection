"""
Week-by-week projections.

Season projections give a per-game baseline; this module moves that baseline
around, week by week, with multipliers you can see and switch off:

  matchup   — opponent's fantasy points allowed to this position vs league average
  script    — Vegas spread: underdogs throw more, favorites run more
  volume    — Vegas implied team total vs league average
  team      — your own projected pass-rate change for a team (new OC, new QB)
  injury    — the player's own current status
  vacancy   — volume vacated by injured teammates at the same position
  weather   — wind / cold / precipitation at kickoff, outdoor games only

Every factor degrades to 1.0 when its data source is missing, so the tab still
works with nothing but a schedule.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Own-status multipliers (Sleeper injury_status values)
INJURY_MULT = {
    "out": 0.0, "ir": 0.0, "ir-r": 0.0, "pup": 0.0, "nfi": 0.0, "sus": 0.0,
    "dnr": 0.0, "na": 0.0, "cov": 0.0,
    "doubtful": 0.25, "questionable": 0.92, "probable": 0.98,
}

# How each position responds to a team throwing more than usual
PASS_RATE_ELASTICITY = {"QB": 0.55, "WR": 0.55, "TE": 0.40, "RB": -0.30}


@dataclass
class WeeklyParams:
    matchup_strength: float = 0.6     # 0 = ignore opponent, 1 = full last-season difference
    script_strength: float = 0.5      # spread-driven pass/run tilt
    volume_strength: float = 0.4      # implied team total vs average
    team_change_strength: float = 1.0 # weight on your manual pass-rate overrides
    home_field: float = 0.02
    use_weather: bool = True
    injury_mode: str = "Week 1 only"  # "Week 1 only" | "All weeks" | "Ignore"
    vacancy_share: float = 0.6        # share of an injured teammate's volume that flows down
    vacancy_cap: float = 1.40


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def component_shares(df: pd.DataFrame, scoring: dict) -> pd.DataFrame:
    """Split each player's projected points into passing/receiving ('air') and
    rushing ('ground') — used to apply weather and game-script correctly."""
    s = scoring
    rec_pts = df["receptions"] * s["pts_per_rec"]
    if s.get("te_bonus", 0) > 0:
        rec_pts = rec_pts + np.where(df["position"] == "TE", df["receptions"] * s["te_bonus"], 0)
    air = (df["pass_yds"] / s["pass_yds_per_pt"] + df["pass_td"] * s["pass_td_pts"]
           + rec_pts + df["rec_yds"] / s["rec_yds_per_pt"] + df["rec_td"] * s["rec_td_pts"])
    ground = df["rush_yds"] / s["rush_yds_per_pt"] + df["rush_td"] * s["rush_td_pts"]
    air = air.clip(lower=0)
    ground = ground.clip(lower=0)
    total = (air + ground).replace(0, np.nan)
    out = pd.DataFrame({
        "air_share": (air / total).fillna(0.5).clip(0, 1),
    })
    out["ground_share"] = 1 - out["air_share"]
    return out


def _blend(air_share: pd.Series, air_mult: pd.Series, ground_mult: pd.Series) -> pd.Series:
    return air_share * air_mult + (1 - air_share) * ground_mult


def weather_multipliers(temp_f, wind_mph, precip_pct) -> tuple[pd.Series, pd.Series]:
    """Passing and rushing multipliers for one set of conditions."""
    wind = pd.to_numeric(wind_mph, errors="coerce")
    temp = pd.to_numeric(temp_f, errors="coerce")
    precip = pd.to_numeric(precip_pct, errors="coerce")

    penalty = np.zeros(len(wind))
    penalty += np.where(wind > 12, np.minimum(0.015 * (wind - 12), 0.12), 0)
    penalty += np.where(precip >= 60, 0.03, 0)
    penalty += np.where(temp <= 25, 0.03, 0)
    penalty = np.nan_to_num(penalty)

    air = pd.Series(1 - penalty, index=wind.index)
    ground = pd.Series(1 + 0.4 * penalty, index=wind.index)   # some volume shifts to the run
    return air, ground


def weather_notes(w: pd.DataFrame) -> pd.Series:
    """Short human-readable conditions per row (vectorized, NA-safe)."""
    wind = pd.to_numeric(w.get("wind_mph"), errors="coerce")
    temp = pd.to_numeric(w.get("temp_f"), errors="coerce")
    precip = pd.to_numeric(w.get("precip_pct"), errors="coerce")
    indoor = w["indoor"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    is_bye = w["is_bye"].astype("boolean").fillna(False).to_numpy(dtype=bool)

    parts = pd.Series("", index=w.index)
    parts = parts.str.cat(np.where(wind >= 13, wind.round(0).astype("Int64").astype(str) + " mph wind, ", ""))
    parts = parts.str.cat(np.where(temp <= 32, temp.round(0).astype("Int64").astype(str) + "°F, ", ""))
    parts = parts.str.cat(np.where(precip >= 60, precip.round(0).astype("Int64").astype(str) + "% precip, ", ""))
    parts = parts.str.rstrip(", ")

    note = np.where(parts != "", parts, np.where(wind.notna(), "Clear", "No forecast yet"))
    note = np.where(indoor, "Indoors", note)
    return pd.Series(np.where(is_bye, "—", note), index=w.index)


# ----------------------------------------------------------------------------
# Main builder
# ----------------------------------------------------------------------------
def build_weekly(
    df: pd.DataFrame,
    games: pd.DataFrame,
    scoring: dict,
    params: WeeklyParams,
    ratings: pd.DataFrame | None = None,
    weather: pd.DataFrame | None = None,
    pass_rate_outlook: pd.DataFrame | None = None,
    top_n: int = 300,
) -> pd.DataFrame:
    """One row per player per week (byes included, with 0 points)."""
    cols = ["player", "player_id", "team", "position", "proj_pts", "ppg", "overall_rank"]
    cols = [c for c in cols if c in df.columns]
    base = df.nsmallest(min(top_n, len(df)), "overall_rank")[cols].copy()
    base = pd.concat([base, component_shares(df.loc[base.index], scoring)], axis=1)
    if "injury_status" in df.columns:
        base["injury_status"] = df.loc[base.index, "injury_status"]
    else:
        base["injury_status"] = np.nan
    base = base.reset_index(drop=True)

    # --- own injury status ---------------------------------------------------
    status = base["injury_status"].astype("string").str.lower().fillna("")
    base["own_injury_mult"] = status.map(INJURY_MULT).fillna(1.0)
    if params.injury_mode == "Ignore":
        base["own_injury_mult"] = 1.0

    # --- vacancy: volume freed up by injured teammates at the same position ---
    base["vacancy_mult"] = 1.0
    out_mask = base["own_injury_mult"] < 0.5
    if params.injury_mode != "Ignore" and out_mask.any():
        for (team, pos), grp in base.groupby(["team", "position"]):
            hurt = grp[grp["own_injury_mult"] < 0.5]
            healthy = grp[grp["own_injury_mult"] >= 0.5]
            if hurt.empty or healthy.empty:
                continue
            vacated = float((hurt["ppg"] * (1 - hurt["own_injury_mult"])).sum()) * params.vacancy_share
            weights = healthy["ppg"].clip(lower=0.1)
            share = vacated * weights / weights.sum()
            mult = (1 + share / healthy["ppg"].clip(lower=1.0)).clip(upper=params.vacancy_cap)
            base.loc[healthy.index, "vacancy_mult"] = mult

    # --- expand to the schedule ---------------------------------------------
    weeks = sorted(games["week"].unique())
    grid = pd.MultiIndex.from_product(
        [sorted(games["team"].unique()), weeks], names=["team", "week"]
    ).to_frame(index=False)
    sched = grid.merge(games, on=["team", "week"], how="left")
    sched["is_bye"] = sched["opponent"].isna()

    w = base.merge(sched, on="team", how="inner")

    # --- matchup -------------------------------------------------------------
    w["matchup_mult"] = 1.0
    if ratings is not None and not ratings.empty:
        r = ratings[["def_team", "position", "fpts_allowed_pg"]]
        pos_avg = r.groupby("position")["fpts_allowed_pg"].mean().rename("pos_avg")
        r = r.merge(pos_avg, on="position")
        r["ratio"] = r["fpts_allowed_pg"] / r["pos_avg"]
        w = w.merge(
            r[["def_team", "position", "ratio", "fpts_allowed_pg"]],
            left_on=["opponent", "position"], right_on=["def_team", "position"], how="left",
        ).drop(columns=["def_team"])
        w["matchup_mult"] = (
            1 + params.matchup_strength * (w["ratio"].fillna(1.0) - 1)
        ).clip(0.75, 1.30)

    # --- game script (spread) ------------------------------------------------
    w["script_mult"] = 1.0
    if "team_spread" in w.columns and w["team_spread"].notna().any():
        underdog = (-w["team_spread"].fillna(0)).clip(-10, 10) / 10.0
        air = 1 + 0.06 * params.script_strength * 2 * underdog
        ground = 1 - 0.05 * params.script_strength * 2 * underdog
        w["script_mult"] = _blend(w["air_share"], air, ground).where(w["team_spread"].notna(), 1.0)

    # --- implied team total --------------------------------------------------
    w["volume_mult"] = 1.0
    if "implied_total" in w.columns and w["implied_total"].notna().any():
        lg = float(w["implied_total"].mean())
        w["volume_mult"] = (
            1 + params.volume_strength * (w["implied_total"] / lg - 1)
        ).fillna(1.0).clip(0.80, 1.20)

    # --- your pass-rate outlook for each team --------------------------------
    w["team_mult"] = 1.0
    if pass_rate_outlook is not None and not pass_rate_outlook.empty:
        o = pass_rate_outlook.copy()
        o["delta"] = (o["projected_pass_rate"] - o["pass_rate"]) / o["pass_rate"].replace(0, np.nan)
        w = w.merge(o[["team", "delta"]], on="team", how="left")
        elast = w["position"].map(PASS_RATE_ELASTICITY).fillna(0)
        w["team_mult"] = (
            1 + params.team_change_strength * elast * w["delta"].fillna(0)
        ).clip(0.80, 1.20)

    # --- home field ----------------------------------------------------------
    w["home_mult"] = np.where(w["is_home"].fillna(False), 1 + params.home_field, 1 - params.home_field)

    # --- weather -------------------------------------------------------------
    w["weather_mult"] = 1.0
    for c in ["temp_f", "wind_mph", "precip_pct"]:
        w[c] = np.nan
    if params.use_weather and weather is not None and not weather.empty:
        w = w.drop(columns=["temp_f", "wind_mph", "precip_pct"]).merge(
            weather, on="game_id", how="left")
        outdoor = ~w["indoor"].fillna(False)
        air, ground = weather_multipliers(w["temp_f"], w["wind_mph"], w["precip_pct"])
        blended = _blend(w["air_share"], air, ground)
        w["weather_mult"] = np.where(outdoor & w["wind_mph"].notna(), blended, 1.0)
    w["weather_note"] = weather_notes(w)

    # --- injuries by week ----------------------------------------------------
    if params.injury_mode == "All weeks":
        apply_inj = pd.Series(True, index=w.index)
    elif params.injury_mode == "Week 1 only":
        apply_inj = w["week"] == min(weeks)
    else:
        apply_inj = pd.Series(False, index=w.index)
    w["injury_mult"] = np.where(apply_inj, w["own_injury_mult"], 1.0)
    w["vacancy_mult_w"] = np.where(apply_inj, w["vacancy_mult"], 1.0)

    # --- put it together -----------------------------------------------------
    factor_cols = ["matchup_mult", "script_mult", "volume_mult", "team_mult",
                   "home_mult", "weather_mult", "injury_mult", "vacancy_mult_w"]
    w["total_mult"] = w[factor_cols].prod(axis=1)
    w.loc[w["is_bye"], "total_mult"] = 0.0
    w["proj_pts_week"] = (w["ppg"] * w["total_mult"]).round(2)
    w["delta_vs_avg"] = (w["proj_pts_week"] - w["ppg"]).round(2)

    w["opponent"] = np.where(w["is_bye"], "BYE",
                             np.where(w["is_home"].fillna(False), "vs " + w["opponent"].astype(str),
                                      "@ " + w["opponent"].astype(str)))
    return w.rename(columns={"vacancy_mult_w": "vacancy_week"}).sort_values(
        ["week", "proj_pts_week"], ascending=[True, False]).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Views
# ----------------------------------------------------------------------------
def season_grid(weekly_df: pd.DataFrame, players: list[str]) -> pd.DataFrame:
    """Players × weeks matrix of weekly points (byes are 0)."""
    sub = weekly_df[weekly_df["player"].isin(players)]
    return sub.pivot_table(index="player", columns="week", values="proj_pts_week", aggfunc="first")


def stretch_summary(weekly_df: pd.DataFrame, weeks: list[int], min_rank: int = 200) -> pd.DataFrame:
    """Average weekly points over a stretch (e.g. fantasy playoffs), vs season average."""
    sub = weekly_df[(weekly_df["week"].isin(weeks)) & (weekly_df["overall_rank"] <= min_rank)]
    out = (
        sub.groupby(["player", "position", "team"])
        .agg(stretch_ppg=("proj_pts_week", "mean"),
             season_ppg=("ppg", "first"),
             byes=("is_bye", "sum"))
        .reset_index()
    )
    out["edge"] = (out["stretch_ppg"] - out["season_ppg"]).round(2)
    out["stretch_ppg"] = out["stretch_ppg"].round(2)
    return out.sort_values("stretch_ppg", ascending=False)


def factor_breakdown(weekly_df: pd.DataFrame, player: str, week: int) -> pd.DataFrame:
    """Why this week's number differs from the player's season average."""
    row = weekly_df[(weekly_df["player"] == player) & (weekly_df["week"] == week)]
    if row.empty:
        return pd.DataFrame()
    r = row.iloc[0]
    labels = {
        "matchup_mult": f"Matchup ({r['opponent']})",
        "script_mult": "Game script (spread)",
        "volume_mult": "Implied team total",
        "team_mult": "Team pass-rate outlook",
        "home_mult": "Home / away",
        "weather_mult": f"Weather ({r['weather_note']})",
        "injury_mult": "Own injury status",
        "vacancy_week": "Injured teammates",
    }
    rows = [{"factor": lbl, "multiplier": round(float(r[k]), 3),
             "points": round(float(r["ppg"]) * (float(r[k]) - 1), 2)}
            for k, lbl in labels.items()]
    return pd.DataFrame(rows).sort_values("points", key=abs, ascending=False)
