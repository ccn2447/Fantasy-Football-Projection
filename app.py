"""
Fantasy Football Projection Tool — live data, superflex-aware, strategy-generating
Run with:  streamlit run app.py

Data sources (all free, no API keys):
- nflverse: real NFL stats, schedules, injury reports
- Sleeper API: current injury status
- FantasyFootballCalculator: market ADP
"""

import datetime
import io
import urllib.parse

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import data_sources as ds
import projections as pj
from data_sources import POSITIONS

st.set_page_config(page_title="Fantasy Football Projections", page_icon="🏈", layout="wide")

POSITION_COLORS = {"QB": "#E4572E", "RB": "#17BEBB", "WR": "#FFC914", "TE": "#76B041"}
DISPLAY_STATS = ["pass_yds", "pass_td", "interceptions", "rush_yds", "rush_td",
                 "receptions", "rec_yds", "rec_td", "fumbles"]

TODAY = datetime.date.today()
LAST_COMPLETE_SEASON = TODAY.year - 1 if TODAY.month >= 3 else TODAY.year - 2
UPCOMING_SEASON = LAST_COMPLETE_SEASON + 1

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.title("🏈 Settings")

st.sidebar.subheader("Data source")
source = st.sidebar.radio("Source", ["nflverse (live)", "Sample data (offline)"])

if source == "nflverse (live)":
    season_range = st.sidebar.slider(
        "Seasons to base projections on",
        LAST_COMPLETE_SEASON - 5, LAST_COMPLETE_SEASON,
        (LAST_COMPLETE_SEASON - 2, LAST_COMPLETE_SEASON),
    )
    projected_games = st.sidebar.slider("Projected games next season", 10, 17, 16)
    min_games = st.sidebar.slider("Minimum games played (sample size)", 1, 20, 6)
    td_reg = st.sidebar.slider(
        "TD regression", 0.0, 0.6, 0.3, 0.05,
        help="TD rates are the noisiest stat year-to-year. This pulls each player's TD rate "
             "toward the position average — 0.3 means 30% of the way. Improves accuracy; "
             "set to 0 for raw historical rates.",
    )
    enrich = st.sidebar.multiselect(
        "Extra data (each adds a download)",
        ["Injury status & history", "Strength of schedule", "Market ADP"],
        default=["Injury status & history", "Strength of schedule", "Market ADP"],
    )
else:
    enrich = []

teams = st.sidebar.number_input("Teams in league", 4, 20, 12)

st.sidebar.subheader("Starting roster")
c1, c2 = st.sidebar.columns(2)
qb_slots = c1.number_input("QB", 0, 3, 1)
rb_slots = c2.number_input("RB", 0, 5, 2)
wr_slots = c1.number_input("WR", 0, 5, 2)
te_slots = c2.number_input("TE", 0, 3, 1)
flex_slots = c1.number_input("FLEX (RB/WR/TE)", 0, 4, 1)
superflex_slots = c2.number_input(
    "SUPERFLEX (QB/RB/WR/TE)", 0, 2, 0,
    help="Superflex slots are almost always filled with QBs — this dramatically raises QB value.",
)

st.sidebar.subheader("Scoring")
scoring_preset = st.sidebar.radio("Preset", ["PPR", "Half PPR", "Standard", "Custom"], horizontal=True)
preset_ppr = {"PPR": 1.0, "Half PPR": 0.5, "Standard": 0.0}.get(scoring_preset, 1.0)

with st.sidebar.expander("Scoring details", expanded=(scoring_preset == "Custom")):
    pts_per_rec = st.number_input("Points per reception", 0.0, 2.0, preset_ppr, 0.25)
    te_bonus = st.number_input("TE reception bonus (TE premium)", 0.0, 1.0, 0.0, 0.25)
    pass_yds_per_pt = st.number_input("Passing yards per point", 10, 50, 25, 5)
    pass_td_pts = st.number_input("Passing TD", 1.0, 8.0, 4.0, 0.5)
    int_pts = st.number_input("Interception", -5.0, 0.0, -2.0, 0.5)
    rush_yds_per_pt = st.number_input("Rushing yards per point", 5, 25, 10, 5)
    rush_td_pts = st.number_input("Rushing TD", 1.0, 8.0, 6.0, 0.5)
    rec_yds_per_pt = st.number_input("Receiving yards per point", 5, 25, 10, 5)
    rec_td_pts = st.number_input("Receiving TD", 1.0, 8.0, 6.0, 0.5)
    fumble_pts = st.number_input("Fumble lost", -5.0, 0.0, -2.0, 0.5)


def score(d: pd.DataFrame) -> pd.Series:
    rec_pts = d["receptions"] * pts_per_rec
    if te_bonus > 0 and "position" in d.columns:
        rec_pts = rec_pts + np.where(d["position"] == "TE", d["receptions"] * te_bonus, 0)
    return (
        d["pass_yds"] / pass_yds_per_pt + d["pass_td"] * pass_td_pts
        + d["interceptions"] * int_pts
        + d["rush_yds"] / rush_yds_per_pt + d["rush_td"] * rush_td_pts
        + rec_pts + d["rec_yds"] / rec_yds_per_pt + d["rec_td"] * rec_td_pts
        + d["fumbles"] * fumble_pts
    )


# ----------------------------------------------------------------------------
# Load + enrich data
# ----------------------------------------------------------------------------
st.title("Fantasy Football Projection Tool")

tendencies = sos = None
if source == "nflverse (live)":
    seasons = list(range(season_range[0], season_range[1] + 1))
    try:
        with st.spinner(f"Downloading {seasons[0]}–{seasons[-1]} stats from nflverse…"):
            weekly = ds.load_nflverse_weekly(tuple(seasons))
        season_stats = ds.aggregate_seasons(weekly)
        df = pj.build_projections(season_stats, seasons, projected_games, min_games, td_reg)
    except Exception as e:
        st.error(f"Could not download nflverse data ({e}). Check your connection or switch to sample data.")
        st.stop()

    sources_ok, sources_down = [f"nflverse {seasons[0]}–{seasons[-1]}"], []

    # Team tendencies (pass/run) — free, computed from already-downloaded data
    tendencies = ds.team_tendencies(weekly, max(seasons))
    df = df.merge(tendencies[["team", "pass_rate"]], on="team", how="left")

    # Injuries
    if "Injury status & history" in enrich:
        with st.spinner("Fetching injury data…"):
            sleeper = ds.load_sleeper_injuries()
            inj_hist = ds.load_injury_history(max(seasons))
        if sleeper is not None:
            df["name_key_"] = ds.name_key(df["player"])
            df = df.merge(sleeper, on=["name_key_", "position"], how="left")
            sources_ok.append("Sleeper injuries")
        else:
            sources_down.append("Sleeper injury status")
        if inj_hist is not None:
            df = df.merge(inj_hist, on="player_id", how="left")
            sources_ok.append(f"{max(seasons)} injury reports")
        else:
            sources_down.append("nflverse injury history")

    # Strength of schedule
    if "Strength of schedule" in enrich:
        with st.spinner("Computing strength of schedule…"):
            ratings = ds.defense_ratings(weekly, max(seasons), score)
            schedule = ds.load_schedule_opponents(UPCOMING_SEASON)
        if ratings is not None and schedule is not None:
            sos = ds.compute_sos(schedule, ratings)
            df = df.merge(sos, on=["team", "position"], how="left")
            sources_ok.append(f"{UPCOMING_SEASON} schedule SOS")
        else:
            sources_down.append(f"{UPCOMING_SEASON} schedule (may not be released yet)")

    # Market ADP
    if "Market ADP" in enrich:
        with st.spinner("Fetching market ADP…"):
            adp_format = "Superflex" if superflex_slots > 0 else scoring_preset
            adp = ds.load_adp(adp_format, int(teams), UPCOMING_SEASON)
        if adp is not None:
            if "name_key_" not in df.columns:
                df["name_key_"] = ds.name_key(df["player"])
            df = df.merge(adp, on=["name_key_", "position"], how="left")
            sources_ok.append(f"FFC ADP ({adp_format})")
        else:
            sources_down.append("FantasyFootballCalculator ADP")

    msg = f"Loaded: {', '.join(sources_ok)}."
    if sources_down:
        st.warning(msg + f" Unavailable right now: {', '.join(sources_down)} — those columns are hidden.")
    else:
        st.success(msg)
else:
    df = pd.read_csv("sample_projections.csv")
    df["player_id"] = df["player"]
    st.info("Using bundled **sample data** (demo numbers only — live extras like injuries/SOS/ADP need the nflverse source).")

df["position"] = df["position"].astype(str).str.upper().str.strip()
df = df[df["position"].isin(POSITIONS)].copy()

# ----------------------------------------------------------------------------
# Scoring, VOR, tiers
# ----------------------------------------------------------------------------
df["proj_pts"] = score(df).round(1)
df["ppg"] = (df["proj_pts"] / df["games"].replace(0, np.nan)).round(2)

starters, repl = pj.replacement_levels(
    df, teams, qb_slots, rb_slots, wr_slots, te_slots, flex_slots, superflex_slots
)
df["vor"] = (df["proj_pts"] - df["position"].map(repl)).round(1)
df = df.sort_values("vor", ascending=False).reset_index(drop=True)
df["overall_rank"] = df.index + 1
df["pos_rank"] = df.groupby("position")["proj_pts"].rank(ascending=False, method="first").astype(int)
df["tier"] = df.groupby("position", group_keys=False).apply(pj.assign_tiers).astype(int)

has_adp = "adp" in df.columns and df["adp"].notna().any()
has_injury = "injury" in df.columns
has_sos = "sos_pctl" in df.columns and df["sos_pctl"].notna().any()
has_pass_rate = "pass_rate" in df.columns and df["pass_rate"].notna().any()

if has_adp:
    df["adp_value"] = (df["adp"] - df["overall_rank"]).round(0)

# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab_rank, tab_strategy, tab_news, tab_compare, tab_pos, tab_teams, tab_cheat = st.tabs([
    "📋 Rankings", "🎯 Draft Strategy", "🏥 News & Injuries",
    "⚖️ Compare", "📊 Positions", "🏟️ Teams", "📥 Cheat Sheet",
])

RENAME = {
    "overall_rank": "Rank", "player": "Player", "team": "Team", "position": "Pos",
    "pos_rank": "Pos Rank", "tier": "Tier", "proj_pts": "Proj Pts", "ppg": "PPG",
    "vor": "VOR", "adp": "ADP", "adp_value": "Value vs ADP", "injury": "Injury",
    "sos_pctl": "SOS", "pass_rate": "Team Pass %", "weeks_out": "Wks Out (LY)",
}

# --- Rankings ---
with tab_rank:
    c1, c2, c3 = st.columns([2, 2, 3])
    pos_filter = c1.multiselect("Positions", POSITIONS, default=POSITIONS)
    top_n = c2.slider("Show top N", 10, max(len(df), 10), min(150, len(df)))
    search = c3.text_input("Search player")

    view = df[df["position"].isin(pos_filter)]
    if search:
        view = view[view["player"].str.contains(search, case=False, na=False)]
    view = view.head(top_n)

    cols = ["overall_rank", "player", "team", "position", "pos_rank", "tier", "proj_pts", "ppg", "vor"]
    if has_adp:
        cols += ["adp", "adp_value"]
    if has_injury:
        cols += ["injury"]
    if has_sos:
        cols += ["sos_pctl"]

    st.dataframe(
        view[cols].rename(columns=RENAME), width="stretch", hide_index=True, height=560,
        column_config={
            "SOS": st.column_config.NumberColumn(
                help="Strength of schedule percentile, 100 = easiest. Based on fantasy points "
                     "each opponent's defense allowed to this position last season."),
            "Value vs ADP": st.column_config.NumberColumn(
                help="Market ADP minus your projection rank. Positive = the market lets this "
                     "player fall past where your numbers rank them."),
        },
    )
    st.caption(
        "Replacement levels — " + " · ".join(
            f"{p}: {repl[p]:.0f} pts (≈{starters[p]:.0f} starters)" for p in POSITIONS)
        + (f" · Superflex: {superflex_slots} slot(s) routed ~80% to QB demand" if superflex_slots else "")
    )

# --- Draft Strategy ---
with tab_strategy:
    strategy_md = pj.generate_strategy(
        df, int(teams), starters, repl, int(superflex_slots),
        pts_per_rec, te_bonus, pass_td_pts, has_adp,
    )
    st.markdown(strategy_md)

# --- News & Injuries ---
with tab_news:
    if not has_injury and source == "nflverse (live)":
        st.info("Enable “Injury status & history” in the sidebar to load this tab.")
    elif not has_injury:
        st.info("Injury data requires the live nflverse source.")
    else:
        st.markdown("Current injury status from Sleeper, last-season injury report history from "
                    "nflverse, and a news search link per player.")
        inj_view = df[df["injury"].notna() | (df.get("weeks_out", pd.Series(dtype=float)).fillna(0) > 0)].copy()
        only_top = st.checkbox("Only players in my draft range", value=True,
                               help=f"Top {int(teams) * 15} overall")
        if only_top:
            inj_view = inj_view[inj_view["overall_rank"] <= int(teams) * 15]

        inj_view["news_link"] = inj_view["player"].map(
            lambda p: "https://news.google.com/search?q=" + urllib.parse.quote(f"{p} NFL injury")
        )
        cols = ["overall_rank", "player", "team", "position", "injury"]
        if "injury_notes" in inj_view.columns:
            cols.append("injury_notes")
        if "weeks_out" in inj_view.columns:
            cols.append("weeks_out")
        if "news_updated" in inj_view.columns:
            cols.append("news_updated")
        cols.append("news_link")

        st.dataframe(
            inj_view[cols].rename(columns={**RENAME, "injury_notes": "Notes",
                                           "news_updated": "Last News", "news_link": "Search News"}),
            width="stretch", hide_index=True, height=500,
            column_config={"Search News": st.column_config.LinkColumn(display_text="🔎 News")},
        )
        st.caption("“Wks Out (LY)” = weeks listed Out/Doubtful/IR on last season's official injury "
                   "reports. Always verify status close to your draft — camp news moves fast.")

# --- Compare ---
with tab_compare:
    picks = st.multiselect("Pick 2–6 players to compare", options=df["player"].tolist(),
                           default=df["player"].head(3).tolist(), max_selections=6)
    if len(picks) >= 2:
        comp = df[df["player"].isin(picks)]
        fig = go.Figure()
        fig.add_bar(x=comp["player"], y=comp["proj_pts"],
                    marker_color=[POSITION_COLORS[p] for p in comp["position"]],
                    text=comp["proj_pts"], textposition="outside")
        fig.update_layout(title="Projected season points", yaxis_title="Points", height=400)
        st.plotly_chart(fig, width="stretch")

        cols = ["player", "team", "position", "games", "proj_pts", "ppg", "vor", "tier"]
        for c in ["adp", "injury", "sos_pctl", "pass_rate"]:
            if c in comp.columns:
                cols.append(c)
        cols += DISPLAY_STATS
        st.dataframe(comp[cols].rename(columns=RENAME).round(1), width="stretch", hide_index=True)
    else:
        st.info("Select at least two players to compare.")

# --- Positions ---
with tab_pos:
    pos = st.selectbox("Position", POSITIONS)
    pos_df = df[df["position"] == pos].sort_values("proj_pts", ascending=False).head(36)
    fig = px.bar(pos_df, x="player", y="proj_pts", color="tier",
                 color_continuous_scale="Viridis",
                 title=f"{pos} projected points by tier",
                 labels={"proj_pts": "Projected points", "player": ""})
    fig.add_hline(y=repl[pos], line_dash="dash", line_color="red", annotation_text="Replacement level")
    fig.update_layout(height=460, xaxis_tickangle=-45)
    st.plotly_chart(fig, width="stretch")
    if len(pos_df):
        st.metric(f"Scarcity at {pos}", f"{pos_df['proj_pts'].iloc[0] - repl[pos]:.0f} pts",
                  help="Gap between the top player and replacement level.")

# --- Teams ---
with tab_teams:
    if tendencies is None:
        st.info("Team tendencies require the live nflverse source.")
    else:
        st.markdown(f"**Offensive philosophy, {LAST_COMPLETE_SEASON} season** — pass rate as a share "
                    "of pass attempts + carries. Pass-heavy teams support more fantasy-relevant "
                    "receivers; run-heavy teams concentrate value in their backfield.")
        fig = px.bar(tendencies, x="team", y="pass_rate", title="Team pass rate (%)",
                     labels={"pass_rate": "Pass %", "team": ""})
        fig.add_hline(y=tendencies["pass_rate"].mean(), line_dash="dash",
                      annotation_text="League average")
        fig.update_layout(height=420)
        st.plotly_chart(fig, width="stretch")
        if has_sos and sos is not None:
            st.markdown(f"**Strength of schedule, {UPCOMING_SEASON}** — average opponent generosity "
                        "to each position (percentile; 100 = easiest schedule).")
            sos_wide = sos.pivot(index="team", columns="position", values="sos_pctl")[POSITIONS]
            st.dataframe(sos_wide.style.background_gradient(cmap="RdYlGn", axis=None),
                         width="stretch", height=450)
        st.caption("A reminder these are last-season tendencies — new coordinators and QB changes "
                   "can shift a team's identity. Treat as context, not gospel.")

# --- Cheat Sheet ---
with tab_cheat:
    st.markdown("Everything in one download: rankings with all enrichments, plus your "
                "settings-specific draft strategy as a separate notes file.")
    cols = ["overall_rank", "player", "team", "position", "pos_rank", "tier", "proj_pts", "ppg", "vor"]
    for c in ["adp", "adp_value", "injury", "weeks_out", "sos_pctl", "pass_rate"]:
        if c in df.columns:
            cols.append(c)
    sheet = df[cols].rename(columns=RENAME)

    buf = io.StringIO()
    sheet.to_csv(buf, index=False)
    d1, d2 = st.columns(2)
    d1.download_button("📥 Cheat sheet (CSV)", buf.getvalue(),
                       file_name="draft_cheat_sheet.csv", mime="text/csv")
    strategy_md = pj.generate_strategy(df, int(teams), starters, repl, int(superflex_slots),
                                       pts_per_rec, te_bonus, pass_td_pts, has_adp)
    d2.download_button("📥 Draft strategy notes (Markdown)",
                       f"# Draft Strategy\n\n{strategy_md}",
                       file_name="draft_strategy.md", mime="text/markdown")
    st.dataframe(sheet.head(50), width="stretch", hide_index=True)

st.caption(
    "Projections are recency-weighted per-game averages from real stats, with TD regression. They "
    "can't see trades, rookies, or scheme changes — use the Teams and News tabs as context and "
    "adjust with your own judgment."
)
