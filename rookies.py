"""
Rookie projections.

A stats model can't see rookies — they have no NFL stats. This builds them a
prior from the one thing that is known before week 1: draft capital.

The curve is *fitted*, not assumed. For every rookie class in a training window
we total what that class actually produced and divide by the games it could have
played, giving the expected per-game production of a pick in each range —
automatically including the 15% of drafted rookies who never record a stat.
Market ADP then separates rookies inside the same range, because ADP knows the
landing spot and the depth chart ahead of them.
"""

import numpy as np
import pandas as pd
import streamlit as st

from data_sources import POSITIONS, STAT_COLUMNS, name_key

# Pro-Football-Reference team codes → nflverse abbreviations
PFR_TEAM_FIX = {"GNB": "GB", "KAN": "KC", "LAR": "LA", "LVR": "LV", "NOR": "NO",
                "NWE": "NE", "SFO": "SF", "TAM": "TB", "SDG": "LAC", "STL": "LA",
                "OAK": "LV", "RAI": "LV", "RAM": "LA"}

# Pick ranges the curve is fitted on, and the pick assumed for undrafted players
BUCKETS = [(1, 16), (17, 32), (33, 64), (65, 105), (106, 160), (161, 300)]
UDFA_PICK = 275
SHRINK_N = 8.0          # thin buckets get pulled toward the position's trend line
TOP_ANCHOR_CAP = 1.25   # ceiling on extrapolating the fit past the top bucket


def _bucket_of(pick: float) -> int:
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= pick <= hi:
            return i
    return len(BUCKETS) - 1


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def load_draft_picks() -> pd.DataFrame | None:
    import nflreadpy as nfl

    try:
        d = nfl.load_draft_picks().to_pandas()
    except Exception:
        return None
    if d is None or d.empty:
        return None
    d = d[d["position"].isin(POSITIONS)].copy()
    d["team"] = d["team"].astype(str).replace(PFR_TEAM_FIX)
    d["player"] = d["pfr_player_name"]
    return d[["season", "round", "pick", "team", "gsis_id", "player", "position"]]


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def load_roster_rookies(season: int) -> pd.DataFrame | None:
    """Current-team lookup, plus undrafted rookies the draft file won't have."""
    import nflreadpy as nfl

    try:
        r = nfl.load_rosters([season]).to_pandas()
    except Exception:
        return None
    if r is None or r.empty:
        return None
    pid = "gsis_id" if "gsis_id" in r.columns else "player_id"
    keep = [c for c in [pid, "team", "position", "years_exp", "full_name", "player_name"] if c in r.columns]
    r = r[keep].rename(columns={pid: "player_id"})
    r["player"] = r.get("full_name", r.get("player_name"))
    return r[r["position"].isin(POSITIONS)]


# ----------------------------------------------------------------------------
# Fit the curve
# ----------------------------------------------------------------------------
@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def fit_rookie_curve(weekly: pd.DataFrame, train_seasons: tuple[int, ...]) -> pd.DataFrame | None:
    """Expected per-game stat line by (position, pick bucket), from real classes."""
    picks = load_draft_picks()
    if picks is None:
        return None
    picks = picks[picks["season"].isin(train_seasons)]
    if picks.empty:
        return None

    totals = (
        weekly[weekly["season"].isin(train_seasons)]
        .groupby(["player_id", "season"])[STAT_COLUMNS].sum().reset_index()
    )
    r = picks.rename(columns={"gsis_id": "player_id"}).merge(
        totals, on=["player_id", "season"], how="left")
    r[STAT_COLUMNS] = r[STAT_COLUMNS].fillna(0.0)   # drafted but never played = zeros
    r["bucket"] = r["pick"].map(_bucket_of)

    rows = []
    for pos, pos_grp in r.groupby("position"):
        raw, ns = [], []
        for b in range(len(BUCKETS)):
            g = pos_grp[pos_grp["bucket"] == b]
            n = len(g)
            ns.append(n)
            raw.append((g[STAT_COLUMNS].sum() / (n * 17)) if n else
                       pd.Series(0.0, index=STAT_COLUMNS))
        raw = pd.DataFrame(raw)
        ns = np.array(ns, dtype=float)
        mids = np.array([np.mean(b) for b in BUCKETS])
        x = np.log(mids)

        # Thin buckets (a 3-rookie sample of top-16 RBs) shrink toward the position's
        # own draft-capital trend line, NOT toward its overall mean — shrinking to the
        # mean would drag elite picks down to day-3 production.
        fitted, anchors = {}, {}
        for c in STAT_COLUMNS:
            y = raw[c].to_numpy()
            slope, intercept = np.polyfit(x, y, 1, w=np.sqrt(np.maximum(ns, 0.5)))
            fitted[c] = np.maximum(slope * x + intercept, 0.0)
            anchors[c] = (slope, intercept)

        w = ns / (ns + SHRINK_N)
        for b in range(len(BUCKETS)):
            vals = {c: float(w[b] * raw[c].iloc[b] + (1 - w[b]) * fitted[c][b])
                    for c in STAT_COLUMNS}
            rows.append({"position": pos, "bucket": b, "n": int(ns[b]),
                         "mid_pick": float(mids[b]), **vals})

        # Anchor both ends so pick 1 and pick 4 aren't projected identically.
        # The top anchor is capped relative to the top bucket: extrapolating a log
        # fit past the data would promise every first-overall pick a career year.
        top_bucket = {c: float(w[0] * raw[c].iloc[0] + (1 - w[0]) * fitted[c][0])
                      for c in STAT_COLUMNS}
        for pick, bucket_id in ((1.0, -1), (float(BUCKETS[-1][1]), len(BUCKETS))):
            vals = {c: float(max(anchors[c][0] * np.log(pick) + anchors[c][1], 0.0))
                    for c in STAT_COLUMNS}
            if pick == 1.0:
                vals = {c: min(v, TOP_ANCHOR_CAP * top_bucket[c]) for c, v in vals.items()}
            rows.append({"position": pos, "bucket": bucket_id, "n": 0,
                         "mid_pick": pick, **vals})

    curve = pd.DataFrame(rows).sort_values(["position", "mid_pick"])
    # Production must not increase with a later pick — clamp inversions
    for pos in curve["position"].unique():
        m = curve["position"] == pos
        for c in STAT_COLUMNS:
            curve.loc[m, c] = np.minimum.accumulate(curve.loc[m, c].to_numpy())
    return curve.reset_index(drop=True)


def _interp_rate(curve_pos: pd.DataFrame, pick: float, col: str) -> float:
    x = np.log(curve_pos["mid_pick"].to_numpy())
    y = curve_pos[col].to_numpy()
    return float(np.interp(np.log(max(pick, 1.0)), x, y))


# ----------------------------------------------------------------------------
# Project this year's class
# ----------------------------------------------------------------------------
def project_rookies(curve: pd.DataFrame, season: int, projected_games: int,
                    exclude_ids: set | None = None,
                    rosters: pd.DataFrame | None = None) -> pd.DataFrame | None:
    """One row per rookie, shaped like build_projections output so everything
    downstream (scoring, VOR, tiers, mock draft, weekly) just works."""
    picks = load_draft_picks()
    if picks is None or curve is None:
        return None
    cls = picks[picks["season"] == season].copy()
    cls = cls.rename(columns={"gsis_id": "player_id"})

    # Undrafted rookies that made a roster
    if rosters is not None and "years_exp" in rosters.columns:
        udfa = rosters[(rosters["years_exp"].fillna(1) == 0)
                       & ~rosters["player_id"].isin(cls["player_id"])].copy()
        if len(udfa):
            udfa["pick"] = UDFA_PICK
            udfa["round"] = 8
            udfa["season"] = season
            cls = pd.concat([cls, udfa[["season", "round", "pick", "team", "player_id",
                                        "player", "position"]]], ignore_index=True)

    cls = cls[cls["position"].isin(POSITIONS) & cls["player"].notna()].copy()
    # ~11% of draft rows have no gsis_id, so the same rookie can arrive twice
    # (once drafted, once off the roster file). Keep the drafted row.
    cls["_key"] = name_key(cls["player"]) + "_" + cls["position"].astype(str)
    cls = cls.sort_values("pick").drop_duplicates("_key", keep="first").drop(columns=["_key"])
    if exclude_ids:
        cls = cls[~cls["player_id"].isin(exclude_ids)]   # already has NFL stats — not a rookie
    if cls.empty:
        return None

    # Current team beats drafted team (post-draft trades, cuts)
    if rosters is not None:
        cur = rosters.dropna(subset=["player_id"]).drop_duplicates("player_id")[["player_id", "team"]]
        cls = cls.merge(cur.rename(columns={"team": "roster_team"}), on="player_id", how="left")
        cls["team"] = cls["roster_team"].fillna(cls["team"])
        cls = cls.drop(columns=["roster_team"])

    out = cls[["player_id", "player", "position", "team", "pick", "round"]].copy()
    for col in STAT_COLUMNS:
        vals = []
        for _, row in out.iterrows():
            cp = curve[curve["position"] == row["position"]]
            vals.append(_interp_rate(cp, row["pick"], col) if len(cp) else 0.0)
        out[f"{col}_pg"] = vals
        out[col] = np.round(np.array(vals) * projected_games, 1)

    out["games"] = projected_games
    out["seasons_used"] = 0
    out["total_games"] = 0
    out["last_season_games"] = 0
    out["is_rookie"] = True
    out["draft_pick"] = out["pick"]
    return out.drop(columns=["pick"]).reset_index(drop=True)


def adjust_with_adp(df: pd.DataFrame, score_fn, weight: float = 0.5,
                    lo: float = 0.5, hi: float = 2.0) -> pd.DataFrame:
    """Scale each rookie's line toward the production its market ADP implies.

    Draft capital says what the pick is worth on average; ADP says what *this*
    rookie walked into. Veterans anchor the ADP→points mapping."""
    if "is_rookie" not in df.columns or "adp" not in df.columns:
        return df
    rookies = df["is_rookie"].fillna(False)
    have_adp = df["adp"].notna()
    if not (rookies & have_adp).any():
        return df

    pts = score_fn(df)
    vets = df[~rookies & have_adp].assign(_pts=pts[~rookies & have_adp]).sort_values("adp")
    if len(vets) < 10:
        return df
    x = vets["adp"].to_numpy()
    y = vets["_pts"].rolling(7, center=True, min_periods=1).median().to_numpy()

    target = np.interp(df.loc[rookies & have_adp, "adp"].to_numpy(), x, y)
    own = pts[rookies & have_adp].to_numpy()
    ratio = np.divide(target, own, out=np.ones_like(target), where=own > 1.0)
    factor = np.clip(ratio ** float(weight), lo, hi)

    idx = df.index[rookies & have_adp]
    for col in STAT_COLUMNS:
        if col in df.columns:
            df.loc[idx, col] = (df.loc[idx, col].to_numpy() * factor).round(1)
        if f"{col}_pg" in df.columns:
            df.loc[idx, f"{col}_pg"] = df.loc[idx, f"{col}_pg"].to_numpy() * factor
    df.loc[idx, "adp_scale"] = factor.round(2)
    return df


def curve_table(curve: pd.DataFrame, score_fn, projected_games: int = 17) -> pd.DataFrame:
    """Readable version of the fitted curve for the UI."""
    t = curve[curve["n"] > 0].copy()
    t["ppg"] = score_fn(t.assign(**{c: t[c] for c in STAT_COLUMNS})).round(2)
    t["picks"] = t["bucket"].map(lambda b: f"{BUCKETS[b][0]}–{BUCKETS[b][1]}")
    return t[["position", "picks", "n", "ppg"]].rename(
        columns={"position": "Pos", "picks": "Draft picks", "n": "Rookies in sample",
                 "ppg": "Expected PPG"})
