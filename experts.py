"""
Expert consensus (FantasyPros ECR, mirrored by ffverse).

Deliberately *not* the backbone of the projections. Consensus and ADP are close
to the same signal — drafters read ECR — so building projections out of it would
make "value vs ADP" compare the market against itself, and would ignore your
scoring settings entirely. It is treated here as a separate, visibly weighted
input, with the model's own rank always kept alongside.

The parts with no equivalent in the stats model are the interesting ones:
  sd / best / worst — how much the experts disagree, i.e. how uncertain a player is
  rank_delta       — consensus movement, which is news arriving
  coverage of rookies and injury returns, where there are no stats to model
"""

import numpy as np
import pandas as pd
import streamlit as st

from data_sources import POSITIONS, name_key

# ecr_type codes in the ffverse file
REDRAFT_OVERALL = "ro"
REDRAFT_SUPERFLEX = "rsf"
DYNASTY_ROOKIE = "drk"


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def load_draft_ecr(superflex: bool = False) -> pd.DataFrame | None:
    """Preseason expert consensus. Uses the superflex board when relevant."""
    import nflreadpy as nfl

    try:
        d = nfl.load_ff_rankings("draft").to_pandas()
    except Exception:
        return None
    if d is None or d.empty or "ecr_type" not in d.columns:
        return None

    want = REDRAFT_SUPERFLEX if superflex else REDRAFT_OVERALL
    sub = d[(d["ecr_type"] == want) & (d["pos"].isin(POSITIONS))].copy()
    if sub.empty:                                    # superflex page missing → fall back
        sub = d[(d["ecr_type"] == REDRAFT_OVERALL) & (d["pos"].isin(POSITIONS))].copy()
    if sub.empty:
        return None

    sub["name_key_"] = name_key(sub["player"])
    out = sub.rename(columns={
        "pos": "position", "ecr": "ecr", "sd": "ecr_sd",
        "best": "ecr_best", "worst": "ecr_worst", "rank_delta": "ecr_delta",
    })
    cols = ["name_key_", "position", "ecr", "ecr_sd", "ecr_best", "ecr_worst", "ecr_delta"]
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    out["ecr_scrape"] = out.get("scrape_date")
    return out[cols + ["ecr_scrape"]].drop_duplicates(["name_key_", "position"])


@st.cache_data(ttl=60 * 60 * 3, show_spinner=False)
def load_weekly_ecr() -> pd.DataFrame | None:
    """In-season weekly consensus, including FantasyPros' own projected points.

    The file is a snapshot of the current week — it carries a scrape date, not a
    week number, and only refreshes during the season.
    """
    import nflreadpy as nfl

    try:
        w = nfl.load_ff_rankings("week").to_pandas()
    except Exception:
        return None
    if w is None or w.empty:
        return None

    name_c = "player_name" if "player_name" in w.columns else "player"
    sub = w[w["pos"].isin(POSITIONS)].copy()
    if sub.empty:
        return None
    sub["name_key_"] = name_key(sub[name_c])
    out = sub.rename(columns={
        "pos": "position", "ecr": "week_ecr", "pos_rank": "week_pos_rank",
        "r2p_pts": "expert_pts", "player_ecr_delta": "week_ecr_delta",
        "start_sit_grade": "start_sit", "player_opponent": "expert_opponent",
        "sd": "week_ecr_sd",
    })
    keep = ["name_key_", "position", "week_ecr", "week_pos_rank", "expert_pts",
            "week_ecr_delta", "start_sit", "expert_opponent", "week_ecr_sd"]
    for c in keep:
        if c not in out.columns:
            out[c] = np.nan
    out["expert_pts"] = pd.to_numeric(out["expert_pts"], errors="coerce")
    out["week_scrape"] = out.get("scrape_date")
    return out[keep + ["week_scrape"]].drop_duplicates(["name_key_", "position"])


# ----------------------------------------------------------------------------
# Merging and blending
# ----------------------------------------------------------------------------
def attach(df: pd.DataFrame, ecr: pd.DataFrame | None) -> pd.DataFrame:
    if ecr is None or ecr.empty:
        return df
    if "name_key_" not in df.columns:
        df = df.copy()
        df["name_key_"] = name_key(df["player"])
    drop = [c for c in ecr.columns if c in df.columns and c not in ("name_key_", "position")]
    return df.merge(ecr.drop(columns=drop), on=["name_key_", "position"], how="left")


def uncertainty(df: pd.DataFrame) -> pd.Series:
    """How far apart the experts are, relative to what is normal at that rank.

    Raw spread grows with rank, so comparing a player's `sd` against the whole
    board would just re-rank the board. Comparing against `ecr` inverts it — every
    elite player looks uncertain because a 1.5 spread is large next to an ECR of 3.
    Instead each player is measured against the typical spread of their ECR
    neighbours, which isolates the players the experts genuinely argue about.
    """
    if "ecr_sd" not in df.columns or "ecr" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    sd = pd.to_numeric(df["ecr_sd"], errors="coerce")
    ecr = pd.to_numeric(df["ecr"], errors="coerce")
    ok = sd.notna() & ecr.notna()
    if ok.sum() < 10:
        return pd.Series(np.nan, index=df.index)

    order = ecr[ok].sort_values().index
    typical = sd[order].rolling(25, center=True, min_periods=5).median()
    rel = (sd[order] / typical.replace(0, np.nan)).reindex(df.index)
    return rel.rank(pct=True) * 100


def spread_label(pctl: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(pctl.isna(), "",
                 np.where(pctl >= 75, "Wide open",
                          np.where(pctl <= 25, "Consensus", "Split"))),
        index=pctl.index,
    )


def blend_board(df: pd.DataFrame, weight: float, rookie_weight: float = 0.7) -> pd.DataFrame:
    """Blend expert rank into the board order, at the rank level.

    Rookies lean on the experts by default whatever the slider says: a
    draft-capital prior is an average over a pick range, while the experts have
    watched this particular rookie's camp and know his depth chart.
    """
    df = df.copy()
    if "model_rank" not in df.columns:
        df["model_rank"] = df["vor"].rank(ascending=False, method="first")
    if "ecr" not in df.columns or df["ecr"].isna().all() or weight <= 0:
        df["blend_score"] = df["model_rank"]
        return df

    df["expert_rank"] = df["ecr"].rank(method="first")
    w = pd.Series(float(weight), index=df.index)
    if "is_rookie" in df.columns:
        rk = df["is_rookie"].fillna(False).astype(bool)
        w[rk] = max(float(weight), float(rookie_weight))
    w = w.where(df["expert_rank"].notna(), 0.0)          # unranked players keep the model's view

    expert = df["expert_rank"].fillna(df["model_rank"])
    df["expert_weight"] = w.round(2)
    df["blend_score"] = (1 - w) * df["model_rank"] + w * expert
    return df


def disagreements(df: pd.DataFrame, min_gap: int = 12, top_n: int = 15,
                  max_rank: int = 180) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Where the model and the experts part company, in both directions.

    This is the point of carrying both: not to average them into a number, but
    to produce a short list of players worth actually investigating.
    """
    if "ecr" not in df.columns or df["ecr"].isna().all():
        return pd.DataFrame(), pd.DataFrame()
    d = df[df["ecr"].notna()].copy()
    d["expert_rank"] = d["ecr"].rank(method="first")
    if "model_rank" not in d.columns:
        d["model_rank"] = d["vor"].rank(ascending=False, method="first")
    d["gap"] = d["expert_rank"] - d["model_rank"]        # + = your model likes them more
    # Only players one side or the other actually wants: a 200-place gap between
    # rank 450 and rank 250 is not a disagreement anyone will ever act on.
    d = d[(d["model_rank"] <= max_rank) | (d["expert_rank"] <= max_rank)]

    cols = [c for c in ["player", "position", "team", "model_rank", "expert_rank", "gap",
                        "proj_pts", "ecr", "ecr_sd", "ecr_delta", "adp", "is_rookie"]
            if c in d.columns]
    higher = d[d["gap"] >= min_gap].nlargest(top_n, "gap")[cols]
    lower = d[d["gap"] <= -min_gap].nsmallest(top_n, "gap")[cols]
    return higher, lower


def movers(df: pd.DataFrame, top_n: int = 12, max_rank: int = 180) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Consensus movement — the closest thing here to a breakout / fade signal."""
    if "ecr_delta" not in df.columns or df["ecr_delta"].isna().all():
        return pd.DataFrame(), pd.DataFrame()
    d = df[df["ecr_delta"].notna() & df["ecr"].notna()].copy()
    d = d[pd.to_numeric(d["ecr"], errors="coerce") <= max_rank]  # ecr is itself a rank
    cols = [c for c in ["player", "position", "team", "ecr", "ecr_delta", "overall_rank",
                        "proj_pts", "is_rookie"] if c in d.columns]
    return d.nlargest(top_n, "ecr_delta")[cols], d.nsmallest(top_n, "ecr_delta")[cols]
