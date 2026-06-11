"""
Fantasy Football Projection Tool — live nflverse data
Run with:  streamlit run app.py

Data source: nflverse (https://github.com/nflverse/nflverse-data), a free,
public NFL data repository. Downloaded via the nflreadpy package — no API key.
"""

import datetime
import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Fantasy Football Projections",
    page_icon="🏈",
    layout="wide",
)

POSITIONS = ["QB", "RB", "WR", "TE"]
POSITION_COLORS = {"QB": "#E4572E", "RB": "#17BEBB", "WR": "#FFC914", "TE": "#76B041"}

STAT_COLUMNS = [
    "pass_yds", "pass_td", "interceptions",
    "rush_yds", "rush_td",
    "receptions", "rec_yds", "rec_td",
    "fumbles",
]

CURRENT_YEAR = datetime.date.today().year
# Last fully completed NFL season (season N runs into January of N+1)
LAST_COMPLETE_SEASON = CURRENT_YEAR - 1 if datetime.date.today().month >= 3 else CURRENT_YEAR - 2


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first column name that exists (nflverse schemas vary by vintage)."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ----------------------------------------------------------------------------
# nflverse data loading + aggregation
# ----------------------------------------------------------------------------
@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def load_nflverse_seasons(seasons: tuple[int, ...]) -> pd.DataFrame:
    """Download weekly player stats from nflverse and aggregate to player-seasons."""
    import nflreadpy as nfl

    weekly = nfl.load_player_stats(list(seasons)).to_pandas()

    # Regular season only
    if "season_type" in weekly.columns:
        weekly = weekly[weekly["season_type"] == "REG"]

    name_col = pick_col(weekly, ["player_display_name", "player_name"])
    team_col = pick_col(weekly, ["team", "recent_team"])
    int_col = pick_col(weekly, ["passing_interceptions", "interceptions"])
    pos_col = pick_col(weekly, ["position"])

    rename = {
        name_col: "player",
        "passing_yards": "pass_yds",
        "passing_tds": "pass_td",
        int_col: "interceptions",
        "rushing_yards": "rush_yds",
        "rushing_tds": "rush_td",
        "receiving_yards": "rec_yds",
        "receiving_tds": "rec_td",
    }
    weekly = weekly.rename(columns={k: v for k, v in rename.items() if k})
    weekly["team"] = weekly[team_col] if team_col else "—"

    # Fumbles lost across all phases
    fumble_parts = [c for c in
                    ["sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost"]
                    if c in weekly.columns]
    weekly["fumbles"] = weekly[fumble_parts].sum(axis=1) if fumble_parts else 0.0

    for col in STAT_COLUMNS:
        if col not in weekly.columns:
            weekly[col] = 0.0
        weekly[col] = pd.to_numeric(weekly[col], errors="coerce").fillna(0.0)

    # Position: from stats if present, otherwise merge from rosters
    if pos_col:
        weekly["position"] = weekly[pos_col]
    else:
        rosters = nfl.load_rosters(list(seasons)).to_pandas()
        r_id = pick_col(rosters, ["gsis_id", "player_id"])
        rosters = rosters[[r_id, "position"]].drop_duplicates(r_id)
        weekly = weekly.merge(
            rosters.rename(columns={r_id: "player_id"}), on="player_id", how="left"
        )

    weekly["position"] = weekly["position"].astype(str).str.upper().str.strip()
    weekly = weekly[weekly["position"].isin(POSITIONS)]

    agg = (
        weekly.groupby(["player_id", "player", "position", "season"])
        .agg(
            games=("week", "nunique"),
            team=("team", "last"),
            **{c: (c, "sum") for c in STAT_COLUMNS},
        )
        .reset_index()
    )
    return agg


def build_projections(
    season_stats: pd.DataFrame,
    seasons: list[int],
    projected_games: int,
    min_games: int,
) -> pd.DataFrame:
    """Recency-weighted per-game rates × projected games."""
    latest = max(seasons)
    # Exponential recency weights: latest season counts most
    weights = {s: 0.5 ** (latest - s) for s in seasons}

    d = season_stats[season_stats["games"] > 0].copy()
    d["weight"] = d["season"].map(weights) * d["games"]  # weight by games played too

    for col in STAT_COLUMNS:
        d[f"{col}_pg"] = d[col] / d["games"]

    def agg_player(g: pd.DataFrame) -> pd.Series:
        w = g["weight"].to_numpy()
        out = {
            "team": g.sort_values("season")["team"].iloc[-1],
            "position": g["position"].iloc[-1],
            "seasons_used": len(g),
            "last_season_games": int(g.loc[g["season"] == g["season"].max(), "games"].iloc[0]),
            "total_games": int(g["games"].sum()),
        }
        for col in STAT_COLUMNS:
            out[f"{col}_pg"] = float(np.average(g[f"{col}_pg"], weights=w))
        return pd.Series(out)

    proj = d.groupby(["player_id", "player"]).apply(agg_player, include_groups=False).reset_index()

    # Keep players active in the most recent season with enough sample
    active_ids = set(season_stats.loc[season_stats["season"] == latest, "player_id"])
    proj = proj[proj["player_id"].isin(active_ids) & (proj["total_games"] >= min_games)]

    proj["games"] = projected_games
    for col in STAT_COLUMNS:
        proj[col] = (proj[f"{col}_pg"] * projected_games).round(1)
    return proj


# ----------------------------------------------------------------------------
# Sidebar — data, league + scoring settings
# ----------------------------------------------------------------------------
st.sidebar.title("🏈 Settings")

st.sidebar.subheader("Data source")
source = st.sidebar.radio(
    "Source",
    ["nflverse (live)", "Sample data (offline)"],
    help="nflverse downloads real NFL stats (free, no API key). "
         "Sample data works offline with made-up demo numbers.",
)

if source == "nflverse (live)":
    season_range = st.sidebar.slider(
        "Seasons to base projections on",
        LAST_COMPLETE_SEASON - 5, LAST_COMPLETE_SEASON,
        (LAST_COMPLETE_SEASON - 2, LAST_COMPLETE_SEASON),
        help="Recent seasons are weighted more heavily.",
    )
    projected_games = st.sidebar.slider(
        "Projected games next season", 10, 17, 16,
        help="17 assumes no missed games; 15–16 bakes in typical injury risk.",
    )
    min_games = st.sidebar.slider(
        "Minimum games played (sample size)", 1, 20, 6,
        help="Filters out players with too few games to project reliably.",
    )

teams = st.sidebar.number_input("Teams in league", 4, 20, 12)

st.sidebar.subheader("Starting roster")
col1, col2 = st.sidebar.columns(2)
qb_slots = col1.number_input("QB", 0, 3, 1)
rb_slots = col2.number_input("RB", 0, 5, 2)
wr_slots = col1.number_input("WR", 0, 5, 2)
te_slots = col2.number_input("TE", 0, 3, 1)
flex_slots = st.sidebar.number_input("FLEX (RB/WR/TE)", 0, 4, 1)

st.sidebar.subheader("Scoring")
scoring_preset = st.sidebar.radio(
    "Preset", ["PPR", "Half PPR", "Standard", "Custom"], horizontal=True
)
preset_ppr = {"PPR": 1.0, "Half PPR": 0.5, "Standard": 0.0}.get(scoring_preset, 1.0)

with st.sidebar.expander("Scoring details", expanded=(scoring_preset == "Custom")):
    pts_per_rec = st.number_input("Points per reception", 0.0, 2.0, preset_ppr, 0.25)
    pass_yds_per_pt = st.number_input("Passing yards per point", 10, 50, 25, 5)
    pass_td_pts = st.number_input("Passing TD", 1.0, 8.0, 4.0, 0.5)
    int_pts = st.number_input("Interception", -5.0, 0.0, -2.0, 0.5)
    rush_yds_per_pt = st.number_input("Rushing yards per point", 5, 25, 10, 5)
    rush_td_pts = st.number_input("Rushing TD", 1.0, 8.0, 6.0, 0.5)
    rec_yds_per_pt = st.number_input("Receiving yards per point", 5, 25, 10, 5)
    rec_td_pts = st.number_input("Receiving TD", 1.0, 8.0, 6.0, 0.5)
    fumble_pts = st.number_input("Fumble lost", -5.0, 0.0, -2.0, 0.5)

# ----------------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------------
st.title("Fantasy Football Projection Tool")

if source == "nflverse (live)":
    seasons = list(range(season_range[0], season_range[1] + 1))
    try:
        with st.spinner(f"Downloading {seasons[0]}–{seasons[-1]} stats from nflverse…"):
            season_stats = load_nflverse_seasons(tuple(seasons))
        df = build_projections(season_stats, seasons, projected_games, min_games)
        st.success(
            f"Projections for {len(df)} players, built from real "
            f"{seasons[0]}–{seasons[-1]} stats ({len(season_stats)} player-seasons). "
            "Recent seasons weighted more heavily."
        )
    except Exception as e:
        st.error(
            f"Could not download nflverse data ({e}). "
            "Check your internet connection, or switch to sample data in the sidebar."
        )
        st.stop()
else:
    df = pd.read_csv("sample_projections.csv")
    df["player_id"] = df["player"]
    st.info("Using bundled **sample data** (demo numbers only).")

df["position"] = df["position"].astype(str).str.upper().str.strip()
df = df[df["position"].isin(POSITIONS)].copy()

# ----------------------------------------------------------------------------
# Scoring, VOR, tiers
# ----------------------------------------------------------------------------
df["proj_pts"] = (
    df["pass_yds"] / pass_yds_per_pt
    + df["pass_td"] * pass_td_pts
    + df["interceptions"] * int_pts
    + df["rush_yds"] / rush_yds_per_pt
    + df["rush_td"] * rush_td_pts
    + df["receptions"] * pts_per_rec
    + df["rec_yds"] / rec_yds_per_pt
    + df["rec_td"] * rec_td_pts
    + df["fumbles"] * fumble_pts
).round(1)
df["ppg"] = (df["proj_pts"] / df["games"].replace(0, np.nan)).round(2)

flex_share = {"RB": 0.45, "WR": 0.45, "TE": 0.10}
starters = {
    "QB": teams * qb_slots,
    "RB": teams * rb_slots + flex_slots * teams * flex_share["RB"],
    "WR": teams * wr_slots + flex_slots * teams * flex_share["WR"],
    "TE": teams * te_slots + flex_slots * teams * flex_share["TE"],
}

replacement_pts = {}
for pos, n in starters.items():
    pos_pts = df.loc[df["position"] == pos, "proj_pts"].sort_values(ascending=False)
    idx = min(max(int(round(n)), 1), len(pos_pts)) - 1
    replacement_pts[pos] = float(pos_pts.iloc[idx]) if len(pos_pts) else 0.0

df["vor"] = (df["proj_pts"] - df["position"].map(replacement_pts)).round(1)
df = df.sort_values("vor", ascending=False).reset_index(drop=True)
df["overall_rank"] = df.index + 1
df["pos_rank"] = df.groupby("position")["proj_pts"].rank(ascending=False, method="first").astype(int)


def assign_tiers(group: pd.DataFrame) -> pd.Series:
    """Gap-based tiers within a position: a new tier starts at unusually large point drops."""
    g = group.sort_values("proj_pts", ascending=False)
    pts = g["proj_pts"].to_numpy()
    if len(pts) < 3:
        return pd.Series(1, index=g.index)
    gaps = -np.diff(pts)
    threshold = max(gaps.mean() + gaps.std(), 1e-9)
    tiers = np.concatenate([[1], 1 + np.cumsum(gaps > threshold)])
    return pd.Series(np.minimum(tiers, 8), index=g.index)


df["tier"] = df.groupby("position", group_keys=False).apply(assign_tiers).astype(int)

# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab_rank, tab_compare, tab_pos, tab_cheat = st.tabs(
    ["📋 Rankings", "⚖️ Compare Players", "📊 Position Analysis", "📥 Cheat Sheet"]
)

with tab_rank:
    c1, c2, c3 = st.columns([2, 2, 3])
    pos_filter = c1.multiselect("Positions", POSITIONS, default=POSITIONS)
    top_n = c2.slider("Show top N", 10, max(len(df), 10), min(150, len(df)))
    search = c3.text_input("Search player")

    view = df[df["position"].isin(pos_filter)]
    if search:
        view = view[view["player"].str.contains(search, case=False, na=False)]
    view = view.head(top_n)

    show_cols = [
        "overall_rank", "player", "team", "position", "pos_rank", "tier",
        "proj_pts", "ppg", "vor",
    ]
    st.dataframe(
        view[show_cols].rename(columns={
            "overall_rank": "Rank", "player": "Player", "team": "Team",
            "position": "Pos", "pos_rank": "Pos Rank", "tier": "Tier",
            "proj_pts": "Proj Pts", "ppg": "PPG", "vor": "VOR",
        }),
        width="stretch",
        hide_index=True,
        height=560,
    )
    st.caption(
        "Replacement levels — " + " · ".join(
            f"{p}: {replacement_pts[p]:.0f} pts (≈{starters[p]:.0f} starters)" for p in POSITIONS
        )
    )

with tab_compare:
    picks = st.multiselect(
        "Pick 2–6 players to compare",
        options=df["player"].tolist(),
        default=df["player"].head(3).tolist(),
        max_selections=6,
    )
    if len(picks) >= 2:
        comp = df[df["player"].isin(picks)]
        fig = go.Figure()
        fig.add_bar(
            x=comp["player"], y=comp["proj_pts"],
            marker_color=[POSITION_COLORS[p] for p in comp["position"]],
            text=comp["proj_pts"], textposition="outside",
        )
        fig.update_layout(title="Projected season points", yaxis_title="Points", height=400)
        st.plotly_chart(fig, width="stretch")

        detail_cols = ["player", "team", "position", "games", "proj_pts", "ppg", "vor", "tier"] + STAT_COLUMNS
        st.dataframe(comp[detail_cols].round(1), width="stretch", hide_index=True)
    else:
        st.info("Select at least two players to compare.")

with tab_pos:
    pos = st.selectbox("Position", POSITIONS)
    pos_df = df[df["position"] == pos].sort_values("proj_pts", ascending=False).head(36)

    fig = px.bar(
        pos_df, x="player", y="proj_pts", color="tier",
        color_continuous_scale="Viridis",
        title=f"{pos} projected points by tier",
        labels={"proj_pts": "Projected points", "player": ""},
    )
    fig.add_hline(
        y=replacement_pts[pos], line_dash="dash", line_color="red",
        annotation_text="Replacement level",
    )
    fig.update_layout(height=460, xaxis_tickangle=-45)
    st.plotly_chart(fig, width="stretch")

    if len(pos_df):
        drop = pos_df["proj_pts"].iloc[0] - replacement_pts[pos]
        st.metric(
            f"Scarcity at {pos}",
            f"{drop:.0f} pts",
            help="Gap between the top player and replacement level — bigger gap means "
                 "the position is more worth reaching for early.",
        )

with tab_cheat:
    st.markdown("Download a draft-day cheat sheet ranked by value over replacement.")
    sheet = df[[
        "overall_rank", "player", "team", "position", "pos_rank", "tier", "proj_pts", "ppg", "vor"
    ]]
    buf = io.StringIO()
    sheet.to_csv(buf, index=False)
    st.download_button(
        "Download cheat sheet (CSV)",
        buf.getvalue(),
        file_name="draft_cheat_sheet.csv",
        mime="text/csv",
    )
    st.dataframe(sheet.head(50), width="stretch", hide_index=True)

st.caption(
    "Projections are recency-weighted per-game averages from real nflverse stats — a solid "
    "baseline, but they don't know about trades, rookies' situations, or coaching changes. "
    "Adjust with your own judgment."
)
