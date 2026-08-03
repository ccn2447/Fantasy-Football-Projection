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
import mock_draft as md
import projections as pj
import weekly as wk
from data_sources import POSITIONS

st.set_page_config(page_title="Fantasy Football Projections", page_icon="🏈", layout="wide")

POSITION_COLORS = {"QB": "#E4572E", "RB": "#17BEBB", "WR": "#FFC914", "TE": "#76B041"}
DISPLAY_STATS = ["pass_yds", "pass_td", "interceptions", "rush_yds", "rush_td",
                 "receptions", "rec_yds", "rec_td", "fumbles"]

TODAY = datetime.date.today()
LAST_COMPLETE_SEASON = TODAY.year - 1 if TODAY.month >= 3 else TODAY.year - 2
UPCOMING_SEASON = LAST_COMPLETE_SEASON + 1
CURRENT_SEASON = UPCOMING_SEASON          # the season whose schedule is live

# Is the season underway? Detected from the schedule, so nothing needs editing
# when week 1 kicks off.
SEASON = ds.season_state(CURRENT_SEASON)

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.title("🏈 Settings")

st.sidebar.subheader("Data source")
source = st.sidebar.radio("Source", ["nflverse (live)", "Sample data (offline)"])

if SEASON["started"] and not SEASON["complete"]:
    in_season = st.sidebar.toggle(
        f"In-season mode — {CURRENT_SEASON} week {SEASON['current_week']}", value=True,
        help="Adds this season's games to the projection blend, mixes current-year defenses into "
             "the matchup ratings, and points the Weekly tab at the games you still have left. "
             "Turn it off to see the preseason board again.",
    )
else:
    in_season = False

if source == "nflverse (live)":
    newest_season = CURRENT_SEASON if in_season else LAST_COMPLETE_SEASON
    season_range = st.sidebar.slider(
        "Seasons to base projections on",
        newest_season - 5, newest_season,
        (newest_season - 2, newest_season),
    )
    default_games = min(SEASON["weeks_remaining"], 17) if in_season else 16
    projected_games = st.sidebar.slider(
        "Games remaining to project" if in_season else "Projected games next season",
        1 if in_season else 10, 17, default_games,
        help="In-season this defaults to the weeks left on the schedule, so season totals "
             "become rest-of-season totals." if in_season else None,
    )
    min_games = st.sidebar.slider("Minimum games played (sample size)", 1, 20, 6)
    td_reg = st.sidebar.slider(
        "TD regression", 0.0, 0.6, 0.3, 0.05,
        help="TD rates are the noisiest stat year-to-year. This pulls each player's TD rate "
             "toward the position average — 0.3 means 30% of the way. Improves accuracy; "
             "set to 0 for raw historical rates.",
    )
    enrich = st.sidebar.multiselect(
        "Extra data (each adds a download)",
        ["Injury status & history", "Strength of schedule", "Market ADP",
         "Week-by-week schedule & weather"],
        default=["Injury status & history", "Strength of schedule", "Market ADP",
                 "Week-by-week schedule & weather"],
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


SCORING = dict(
    pts_per_rec=pts_per_rec, te_bonus=te_bonus, pass_yds_per_pt=pass_yds_per_pt,
    pass_td_pts=pass_td_pts, int_pts=int_pts, rush_yds_per_pt=rush_yds_per_pt,
    rush_td_pts=rush_td_pts, rec_yds_per_pt=rec_yds_per_pt, rec_td_pts=rec_td_pts,
    fumble_pts=fumble_pts,
)


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

tendencies = sos = ratings = games = game_weather = None
if source == "nflverse (live)":
    seasons = list(range(season_range[0], season_range[1] + 1))
    try:
        with st.spinner(f"Downloading {seasons[0]}–{seasons[-1]} stats from nflverse…"):
            weekly = ds.load_nflverse_weekly(tuple(seasons))
        # A requested season may not be published yet — keep only what came back
        seasons = sorted(int(x) for x in weekly["season"].unique()) or seasons
        season_stats = ds.aggregate_seasons(weekly)
        active_seasons = seasons[-2:] if in_season else [max(seasons)]
        df = pj.build_projections(season_stats, seasons, projected_games, min_games, td_reg,
                                  active_seasons=active_seasons)
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

    # Defense ratings — shared by strength of schedule and the weekly model
    ratings_blend = 0.0
    if {"Strength of schedule", "Week-by-week schedule & weather"} & set(enrich):
        if in_season and CURRENT_SEASON in seasons and len(seasons) > 1:
            ratings, ratings_blend = ds.blended_defense_ratings(
                weekly, sorted(seasons)[-2], CURRENT_SEASON, score, SEASON["weeks_played"])
        else:
            ratings = ds.defense_ratings(weekly, max(seasons), score)

    # Strength of schedule
    if "Strength of schedule" in enrich:
        with st.spinner("Computing strength of schedule…"):
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

    # Week-by-week schedule + weather
    if "Week-by-week schedule & weather" in enrich:
        with st.spinner(f"Loading the {UPCOMING_SEASON} week-by-week schedule…"):
            games = ds.load_schedule_games(UPCOMING_SEASON)
        if games is not None:
            sources_ok.append(f"{UPCOMING_SEASON} schedule ({games['week'].max()} weeks)")
            game_weather = ds.load_game_weather(games)
            if game_weather is not None:
                sources_ok.append(f"Open-Meteo forecast ({len(game_weather)} games)")
        else:
            sources_down.append(f"{UPCOMING_SEASON} week-by-week schedule")

    if in_season:
        sources_ok.append(
            f"in-season mode (week {SEASON['current_week']}, "
            f"{SEASON['weeks_remaining']} weeks left"
            + (f", defenses {ratings_blend:.0%} current-year" if ratings_blend else "") + ")"
        )

    msg = f"Loaded: {', '.join(sources_ok)}."
    if sources_down:
        st.warning(msg + f" Unavailable right now: {', '.join(sources_down)} — those columns are hidden.")
    else:
        st.success(msg)
else:
    try:
        df = pd.read_csv("sample_projections.csv")
    except FileNotFoundError:
        st.error("`sample_projections.csv` isn't in this folder. Switch the data source to "
                 "**nflverse (live)** in the sidebar, or drop a sample CSV next to `app.py`.")
        st.stop()
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
(tab_rank, tab_strategy, tab_mock, tab_week, tab_news, tab_compare,
 tab_pos, tab_teams, tab_cheat) = st.tabs([
    "📋 Rankings", "🎯 Draft Strategy", "🕹️ Mock Draft", "📅 Weekly",
    "🏥 News & Injuries", "⚖️ Compare", "📊 Positions", "🏟️ Teams", "📥 Cheat Sheet",
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
    if in_season:
        st.caption(f"In-season: these are rest-of-season totals over {projected_games} remaining "
                   "games. The Weekly tab breaks them out week by week with byes and matchups.")
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

# --- Mock Draft ---
with tab_mock:
    st.markdown(
        "Simulate your draft from your real draft slot. The other managers pick off a blend of "
        "market ADP and your own projections, with noise and roster-need logic — so a superflex "
        "or TE-premium league produces a superflex or TE-premium draft automatically."
    )

    c1, c2, c3, c4 = st.columns(4)
    my_slot = c1.number_input("Your draft slot", 1, int(teams), min(4, int(teams)), key="md_slot")
    rounds = c2.number_input("Rounds", 4, 25, 15, key="md_rounds")
    adp_weight = c3.slider(
        "Board: your projections ↔ market ADP", 0.0, 1.0, 0.6 if has_adp else 0.0, 0.1,
        disabled=not has_adp, key="md_adp_weight",
        help="0 = the room drafts entirely off your numbers. 1 = entirely off market ADP. "
             "Somewhere in the middle is the most realistic.",
    )
    sigma = c4.slider("Manager unpredictability", 0.0, 15.0, 6.0, 0.5, key="md_sigma",
                      help="Standard deviation of how far a manager reaches or waits, in board ranks.")

    c5, c6, c7 = st.columns([3, 3, 2])
    strategy = c5.selectbox("Your auto-pick strategy", md.STRATEGIES, key="md_strategy")
    mode = c6.radio("Mode", ["Simulate the whole draft", "Draft interactively"], horizontal=True,
                    key="md_mode")
    seed = c7.number_input("Seed", 0, 9999, 7, key="md_seed", help="Same seed = same draft. Change it for a new room.")

    if in_season:
        st.info(f"The season is underway (week {SEASON['current_week']}), so ADP and this "
                "simulator describe a draft board rather than your league. Still useful for "
                "dynasty startups and late-drafting leagues.")

    cfg = md.DraftConfig(
        teams=int(teams), rounds=int(rounds), my_slot=int(my_slot), adp_weight=float(adp_weight),
        sigma=float(sigma), my_strategy=strategy, qb_slots=int(qb_slots), rb_slots=int(rb_slots),
        wr_slots=int(wr_slots), te_slots=int(te_slots), flex_slots=int(flex_slots),
        superflex_slots=int(superflex_slots), seed=int(seed), starters=starters,
    )
    pool = md.build_pool(df, cfg)
    st.caption("Your picks: " + ", ".join(str(p) for p in md.my_pick_numbers(cfg)))

    PICK_COLS = ["pick", "round", "manager", "player", "position", "nfl_team", "proj_pts", "vor"]

    def show_my_team(picks_df: pd.DataFrame, grades: pd.DataFrame) -> None:
        mine = picks_df[picks_df["manager"] == "YOU"]
        if mine.empty:
            return
        row = grades[grades["manager"] == "YOU"].iloc[0]
        m1, m2, m3 = st.columns(3)
        m1.metric("Projected starting lineup", f"{row['starter_pts']:.0f} pts",
                  f"{row['starter_pts'] - grades['starter_pts'].mean():+.0f} vs league average")
        m2.metric("Draft grade", f"#{int(row['draft_rank'])} of {int(teams)}")
        m3.metric("Bench points", f"{row['bench_pts']:.0f}")

        lineup = md.starting_lineup(mine, cfg)
        left, right = st.columns([3, 2])
        left.dataframe(
            lineup[["lineup_slot", "pick", "player", "position", "nfl_team", "proj_pts", "vor"]]
            .rename(columns={"lineup_slot": "Slot", "pick": "Pick", "player": "Player",
                             "position": "Pos", "nfl_team": "Team", "proj_pts": "Proj Pts",
                             "vor": "VOR"}),
            width="stretch", hide_index=True, height=430,
        )
        counts = mine["position"].value_counts().reindex(POSITIONS).fillna(0).reset_index()
        counts.columns = ["position", "n"]
        fig = px.bar(counts, x="position", y="n", color="position",
                     color_discrete_map=POSITION_COLORS, title="Roster composition")
        fig.update_layout(height=430, showlegend=False, yaxis_title="Players")
        right.plotly_chart(fig, width="stretch")

    if mode == "Simulate the whole draft":
        state = md.simulate_full(cfg, pool)
        picks = md.results_frame(state, pool, cfg)
        grades = md.grade_teams(picks, cfg)
        show_my_team(picks, grades)

        st.markdown("**League draft grades** — projected points from each team's best legal lineup.")
        st.dataframe(
            grades[["draft_rank", "manager", "starter_pts", "bench_pts", "total_vor"]]
            .rename(columns={"draft_rank": "#", "manager": "Manager", "starter_pts": "Starters",
                             "bench_pts": "Bench", "total_vor": "Total VOR"}),
            width="stretch", hide_index=True, height=min(420, 40 + 35 * int(teams)),
        )

        with st.expander("Full draft board"):
            st.dataframe(md.board_grid(picks, cfg), width="stretch", height=520)
        with st.expander("Every pick"):
            st.dataframe(picks[PICK_COLS + (["adp", "reach"] if has_adp else [])],
                         width="stretch", hide_index=True, height=520)

        with st.expander("Who will actually be there at your picks? (Monte Carlo)", expanded=False):
            s1, s2 = st.columns([1, 3])
            n_sims = s1.slider("Drafts to simulate", 10, 200, 50, 10, key="md_nsims")
            if s1.button("Run simulations", type="primary", key="md_run"):
                with st.spinner(f"Simulating {n_sims} drafts…"):
                    st.session_state["md_avail"] = md.availability(cfg, pool, n_sims=int(n_sims), top_n=48)
            av = st.session_state.get("md_avail")
            if av is not None and len(av):
                pcols = [c for c in av.columns if c.startswith("P") and c[1:].isdigit()]
                heat = av.set_index(av["player"] + " (" + av["position"] + ")")[pcols]
                fig = px.imshow(heat, color_continuous_scale="RdYlGn", aspect="auto", zmin=0, zmax=1,
                                labels={"x": "Your pick #", "y": "", "color": "P(available)"},
                                text_auto=".0%")
                fig.update_layout(height=max(400, 18 * len(heat)),
                                  title="Probability the player is still on the board")
                fig.update_xaxes(side="top", tickvals=pcols, ticktext=[c[1:] for c in pcols])
                st.plotly_chart(fig, width="stretch")
                st.caption("Green = likely available. Anyone at ~50% is a genuine decision point: "
                           "take them now or plan a fallback.")
            else:
                s2.info("Run the simulations to see availability probabilities at each of your picks.")

    else:
        cfg_key = (cfg.teams, cfg.rounds, cfg.my_slot, cfg.adp_weight, cfg.sigma,
                   cfg.my_strategy, cfg.seed, cfg.superflex_slots, len(pool))
        if st.session_state.get("md_key") != cfg_key:
            st.session_state["md_key"] = cfg_key
            st.session_state["md_state"] = md.new_state(cfg, len(pool))
            md.run_to(st.session_state["md_state"], pool, cfg)
        state = st.session_state["md_state"]

        done = state["pick_no"] > cfg.n_picks
        picks = md.results_frame(state, pool, cfg)

        if not done:
            rnd = int(np.ceil(state["pick_no"] / cfg.teams))
            st.subheader(f"On the clock — pick {state['pick_no']} (round {rnd})")
            avail = pool[~state["taken"]].sort_values("vor", ascending=False)
            b1, b2, b3, b4 = st.columns([4, 1.2, 1.2, 1.2])
            options = avail.head(60)
            labels = [f"{r['player']} · {r['position']} · {r['proj_pts']:.0f} pts (VOR {r['vor']:.0f})"
                      for _, r in options.iterrows()]
            chosen = b1.selectbox("Best available", labels, key="md_choice")
            if b2.button("Draft", type="primary", key="md_draft"):
                idx = int(options.index[labels.index(chosen)])
                md.step(state, pool, cfg, forced_idx=idx)
                md.run_to(state, pool, cfg)
                st.rerun()
            if b3.button("Auto-pick", key="md_auto"):
                md.step(state, pool, cfg)
                md.run_to(state, pool, cfg)
                st.rerun()
            if b4.button("Reset", key="md_reset"):
                st.session_state.pop("md_key", None)
                st.rerun()
        else:
            st.success("Draft complete.")
            if st.button("Start a new draft", key="md_new"):
                st.session_state.pop("md_key", None)
                st.rerun()

        if len(picks):
            grades = md.grade_teams(picks, cfg)
            show_my_team(picks, grades)
            recent = picks.tail(int(teams) * 2)[PICK_COLS[:5]]
            with st.expander("Recent picks", expanded=True):
                st.dataframe(recent, width="stretch", hide_index=True, height=280)

# --- Weekly projections ---
with tab_week:
    if games is None:
        st.info("Enable **Week-by-week schedule & weather** in the sidebar (live nflverse source) "
                "to build weekly projections.")
    else:
        st.markdown(
            f"Per-game baselines moved week to week by matchup, game script, injuries and weather. "
            f"Every factor is a visible multiplier on the player's season per-game average — "
            f"switch any of them off below."
        )

        with st.expander("Model settings", expanded=False):
            f1, f2, f3 = st.columns(3)
            matchup_strength = f1.slider(
                "Opponent matchup", 0.0, 1.0, 0.6 if ratings is not None else 0.0, 0.1,
                disabled=ratings is None, key="wk_matchup",
                help="How much of last season's difference in points allowed to carry forward. "
                     "1.0 takes it at face value, which overrates it — defenses change.")
            script_strength = f2.slider("Game script (spread)", 0.0, 1.0, 0.5, 0.1, key="wk_script",
                                        help="Underdogs throw more; favorites run more.")
            volume_strength = f3.slider("Implied team total", 0.0, 1.0, 0.4, 0.1, key="wk_volume",
                                        help="Vegas total and spread → expected points for the offense.")
            g1, g2, g3 = st.columns(3)
            injury_mode = g1.radio("Current injuries apply to", ["Week 1 only", "All weeks", "Ignore"],
                                   key="wk_injmode",
                                   help="Sleeper reports today's status. Carrying a Questionable tag "
                                        "through week 17 is usually wrong.")
            use_weather = g2.checkbox("Use kickoff weather", value=True, key="wk_weather",
                                      help="Open-Meteo forecasts reach about two weeks out; games "
                                           "beyond that get no weather adjustment.")
            home_field = g3.slider("Home-field edge", 0.0, 0.05, 0.02, 0.005, key="wk_hfa")

        outlook = None
        if tendencies is not None:
            with st.expander("Team pass-rate outlook — the one thing the stats can't see"):
                st.caption(
                    "Projections inherit last season's offensive philosophy. If a team changed "
                    "coordinator or quarterback, edit its projected pass rate here and every player "
                    "on that offense shifts (receivers up, backs down, or the reverse)."
                )
                editable = tendencies[["team", "pass_rate"]].copy()
                editable["projected_pass_rate"] = editable["pass_rate"]
                outlook = st.data_editor(
                    editable, hide_index=True, width="stretch", height=320, key="wk_outlook",
                    disabled=["team", "pass_rate"],
                    column_config={
                        "team": "Team",
                        "pass_rate": st.column_config.NumberColumn(
                            f"{LAST_COMPLETE_SEASON} pass %", disabled=True, format="%.1f"),
                        "projected_pass_rate": st.column_config.NumberColumn(
                            f"{UPCOMING_SEASON} pass % (edit)", min_value=35.0, max_value=75.0,
                            step=0.5, format="%.1f"),
                    },
                )

        params = wk.WeeklyParams(
            matchup_strength=matchup_strength, script_strength=script_strength,
            volume_strength=volume_strength, home_field=home_field,
            use_weather=use_weather, injury_mode=injury_mode,
        )
        weekly_df = wk.build_weekly(
            df, games, SCORING, params, ratings=ratings,
            weather=game_weather if use_weather else None,
            pass_rate_outlook=outlook, top_n=300,
        )

        all_weeks = sorted(int(w) for w in weekly_df["week"].unique())
        weeks_left = [w for w in all_weeks if w >= SEASON["current_week"]] if in_season else all_weeks

        r1, r2 = st.columns([1, 3])
        only_left = r1.checkbox("Only weeks I have left", value=in_season, disabled=not in_season,
                                key="wk_remaining")
        weeks_shown = weeks_left if (only_left and in_season) else all_weeks
        if in_season:
            r2.caption(f"Week {SEASON['current_week']} of {SEASON['total_weeks']} — "
                       f"{SEASON['weeks_remaining']} to play. Projections blend "
                       f"{CURRENT_SEASON} games in as they happen.")

        w1, w2, w3 = st.columns([1, 2, 3])
        week_pick = w1.selectbox("Week", weeks_shown, key="wk_week")
        pos_pick = w2.multiselect("Positions", POSITIONS, default=POSITIONS, key="wk_positions")
        roster = w3.multiselect("Only my players (optional)", df["player"].tolist(), key="wk_roster")

        wv = weekly_df[(weekly_df["week"] == week_pick) & (weekly_df["position"].isin(pos_pick))]
        if roster:
            wv = wv[wv["player"].isin(roster)]
        wv = wv.sort_values("proj_pts_week", ascending=False).head(80)

        WCOLS = {
            "player": "Player", "position": "Pos", "team": "Team", "opponent": "Opp",
            "proj_pts_week": "Week Pts", "ppg": "Season PPG", "delta_vs_avg": "Δ vs avg",
            "matchup_mult": "Matchup", "script_mult": "Script", "volume_mult": "Total",
            "injury_mult": "Injury", "vacancy_week": "Vacated", "weather_mult": "Weather",
            "weather_note": "Conditions",
        }
        st.dataframe(
            wv[list(WCOLS)].rename(columns=WCOLS).round(2), width="stretch", hide_index=True,
            height=520,
            column_config={
                "Matchup": st.column_config.NumberColumn(
                    help="Opponent's fantasy points allowed to this position vs league average."),
                "Vacated": st.column_config.NumberColumn(
                    help="Volume freed up by injured teammates at the same position."),
            },
        )
        bye_teams = sorted(weekly_df.loc[(weekly_df["week"] == week_pick) & weekly_df["is_bye"], "team"].unique())
        st.caption(f"Week {week_pick} byes: " + (", ".join(bye_teams) if bye_teams else "none"))

        st.divider()
        d1, d2 = st.columns([2, 3])
        focus = d1.selectbox("Player detail", weekly_df["player"].drop_duplicates().tolist(), key="wk_focus")
        focus_week = d2.select_slider("Break down which week?", weeks_shown, value=week_pick,
                                      key="wk_bdweek")

        pdata = weekly_df[weekly_df["player"] == focus].sort_values("week")
        fig = go.Figure()
        fig.add_bar(x=pdata["week"], y=pdata["proj_pts_week"],
                    marker_color=np.where(pdata["is_bye"], "#bbbbbb", POSITION_COLORS.get(
                        pdata["position"].iloc[0], "#888")),
                    text=pdata["opponent"], hovertext=pdata["weather_note"])
        fig.add_hline(y=float(pdata["ppg"].iloc[0]), line_dash="dash",
                      annotation_text="Season average")
        fig.update_layout(title=f"{focus} — projected points by week", height=380,
                          xaxis_title="Week", yaxis_title="Points")
        st.plotly_chart(fig, width="stretch")

        bd = wk.factor_breakdown(weekly_df, focus, focus_week)
        if len(bd):
            b1, b2 = st.columns([2, 3])
            b1.dataframe(bd.rename(columns={"factor": "Factor", "multiplier": "×",
                                            "points": "Points"}),
                         width="stretch", hide_index=True)
            fig2 = px.bar(bd, x="points", y="factor", orientation="h",
                          color=np.where(bd["points"] >= 0, "up", "down"),
                          color_discrete_map={"up": "#17BEBB", "down": "#E4572E"},
                          labels={"points": "Points vs season average", "factor": ""})
            fig2.update_layout(height=320, showlegend=False, title=f"Week {focus_week} drivers")
            b2.plotly_chart(fig2, width="stretch")

        st.divider()
        ros_label = "Rest of season" if in_season else "Full season"
        st.markdown(f"**{ros_label} totals** — every weekly adjustment added up over "
                    f"weeks {min(weeks_shown)}–{max(weeks_shown)}.")
        ros = wk.stretch_summary(weekly_df, weeks_shown, min_rank=int(teams) * 15)
        ros_pos = st.multiselect("Positions", POSITIONS, default=POSITIONS, key="ros_positions")
        st.dataframe(
            ros[ros["position"].isin(ros_pos)].head(120).rename(
                columns={"player": "Player", "position": "Pos", "team": "Team",
                         "stretch_total": f"{ros_label} Pts", "stretch_ppg": "PPG",
                         "season_ppg": "Baseline PPG", "edge": "Edge", "games": "Games",
                         "byes": "Byes"}),
            width="stretch", hide_index=True, height=420,
        )
        st.caption("“Edge” is how much the schedule, injuries and weather move a player off their "
                   "baseline per-game average. This is the in-season equivalent of the Rankings "
                   "tab — it already knows about byes and matchups.")

        st.divider()
        st.markdown("**Fantasy playoff stretch** — who gets easier weeks when it matters.")
        p1, p2 = st.columns([2, 3])
        playoff_default = [w for w in weeks_shown if w >= max(weeks_shown) - 3]
        playoff_weeks = p1.multiselect("Playoff weeks", weeks_shown, key="wk_playoff",
                                       default=playoff_default)
        if playoff_weeks:
            stretch = wk.stretch_summary(weekly_df, playoff_weeks, min_rank=int(teams) * 12)
            p2.dataframe(
                stretch.head(20)[["player", "position", "team", "stretch_ppg", "season_ppg",
                                  "edge", "byes"]]
                .rename(columns={"player": "Player", "position": "Pos", "team": "Team",
                                 "stretch_ppg": "Stretch PPG", "season_ppg": "Season PPG",
                                 "edge": "Edge", "byes": "Byes"}),
                width="stretch", hide_index=True, height=400,
            )

        buf_w = io.StringIO()
        weekly_df[["week", "player", "position", "team", "opponent", "ppg", "proj_pts_week",
                   "matchup_mult", "script_mult", "volume_mult", "injury_mult", "vacancy_week",
                   "weather_mult", "weather_note", "is_bye"]].to_csv(buf_w, index=False)
        st.download_button("📥 Weekly projections (CSV)", buf_w.getvalue(), key="wk_csv",
                           file_name=f"weekly_projections_{UPCOMING_SEASON}.csv", mime="text/csv")
        st.caption(
            "Weather only adjusts games inside the forecast window and outdoors; everything else "
            "shows as “No forecast yet”. Matchup ratings come from last season's defenses, so treat "
            "week 14 as a rough sketch, not a lineup decision."
        )

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
            sos_wide = sos_wide.sort_index(ascending=False)  # A at top after y-axis render
            fig_sos = px.imshow(
                sos_wide,
                color_continuous_scale="RdYlGn",
                aspect="auto",
                labels={"x": "Position", "y": "", "color": "SOS pctl"},
                text_auto=".0f",
            )
            fig_sos.update_layout(height=700, coloraxis_colorbar_title="Easier →")
            st.plotly_chart(fig_sos, width="stretch")
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
