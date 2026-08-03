"""
Projection model (recency-weighted rates + TD regression) and the
settings-aware draft strategy generator.
"""

import numpy as np
import pandas as pd

from data_sources import POSITIONS, STAT_COLUMNS

PROJ_STATS = [
    "pass_yds", "pass_td", "interceptions",
    "rush_yds", "rush_td",
    "receptions", "rec_yds", "rec_td",
    "fumbles",
]
TD_STATS = ["pass_td", "rush_td", "rec_td"]


def build_projections(
    season_stats: pd.DataFrame,
    seasons: list[int],
    projected_games: int,
    min_games: int,
    td_regression: float = 0.3,
    active_seasons: list[int] | None = None,
) -> pd.DataFrame:
    """Recency-weighted per-game rates × projected games, with TD rates
    regressed toward the position mean (TDs are the noisiest stat year-to-year).

    `active_seasons` decides who counts as an active player. Defaults to the
    latest season; in-season you want the last two, so a player who has been hurt
    since week 1 doesn't vanish from the board."""
    latest = max(seasons)
    weights = {s: 0.5 ** (latest - s) for s in seasons}

    d = season_stats[(season_stats["games"] > 0)
                     & season_stats["season"].isin(seasons)].copy()
    d["weight"] = d["season"].map(weights) * d["games"]

    for col in PROJ_STATS:
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
        for col in PROJ_STATS:
            out[f"{col}_pg"] = float(np.average(g[f"{col}_pg"], weights=w))
        return pd.Series(out)

    proj = d.groupby(["player_id", "player"]).apply(agg_player, include_groups=False).reset_index()

    active = active_seasons or [latest]
    active_ids = set(season_stats.loc[season_stats["season"].isin(active), "player_id"])
    proj = proj[proj["player_id"].isin(active_ids) & (proj["total_games"] >= min_games)].copy()

    # TD regression toward position mean
    if td_regression > 0:
        for col in TD_STATS:
            pos_mean = proj.groupby("position")[f"{col}_pg"].transform("mean")
            proj[f"{col}_pg"] = (1 - td_regression) * proj[f"{col}_pg"] + td_regression * pos_mean

    proj["games"] = projected_games
    for col in PROJ_STATS:
        proj[col] = (proj[f"{col}_pg"] * projected_games).round(1)
    return proj


def replacement_levels(
    df: pd.DataFrame, teams: int,
    qb_slots: int, rb_slots: int, wr_slots: int, te_slots: int,
    flex_slots: int, superflex_slots: int,
) -> tuple[dict, dict]:
    """Starters demanded per position. Flex splits 45/45/10 RB/WR/TE;
    superflex goes overwhelmingly to QBs (80/10/8/2)."""
    flex_share = {"RB": 0.45, "WR": 0.45, "TE": 0.10}
    sf_share = {"QB": 0.80, "RB": 0.10, "WR": 0.08, "TE": 0.02}

    starters = {
        "QB": teams * (qb_slots + superflex_slots * sf_share["QB"]),
        "RB": teams * (rb_slots + flex_slots * flex_share["RB"] + superflex_slots * sf_share["RB"]),
        "WR": teams * (wr_slots + flex_slots * flex_share["WR"] + superflex_slots * sf_share["WR"]),
        "TE": teams * (te_slots + flex_slots * flex_share["TE"] + superflex_slots * sf_share["TE"]),
    }
    repl = {}
    for pos, n in starters.items():
        pos_pts = df.loc[df["position"] == pos, "proj_pts"].sort_values(ascending=False)
        idx = min(max(int(round(n)), 1), max(len(pos_pts), 1)) - 1
        repl[pos] = float(pos_pts.iloc[idx]) if len(pos_pts) else 0.0
    return starters, repl


def assign_tiers(group: pd.DataFrame) -> pd.Series:
    g = group.sort_values("proj_pts", ascending=False)
    pts = g["proj_pts"].to_numpy()
    if len(pts) < 3:
        return pd.Series(1, index=g.index)
    gaps = -np.diff(pts)
    threshold = max(gaps.mean() + gaps.std(), 1e-9)
    tiers = np.concatenate([[1], 1 + np.cumsum(gaps > threshold)])
    return pd.Series(np.minimum(tiers, 8), index=g.index)


# ----------------------------------------------------------------------------
# Draft strategy generator — adapts to league settings + computed data
# ----------------------------------------------------------------------------
def generate_strategy(
    df: pd.DataFrame, teams: int, starters: dict, repl: dict,
    superflex_slots: int, pts_per_rec: float, te_bonus: float,
    pass_td_pts: float, has_adp: bool,
) -> str:
    lines = []
    league_desc = []
    if superflex_slots > 0:
        league_desc.append(f"**Superflex** ({superflex_slots} extra QB-eligible slot{'s' if superflex_slots > 1 else ''})")
    league_desc.append({1.0: "**Full PPR**", 0.5: "**Half PPR**", 0.0: "**Standard**"}.get(pts_per_rec, f"**{pts_per_rec} PPR**"))
    if te_bonus > 0:
        league_desc.append(f"**TE premium** (+{te_bonus}/rec)")
    lines.append(f"### Your league: {teams} teams · " + " · ".join(league_desc))
    lines.append("")

    # Position priority by scarcity (avg VOR of top-5 at each position)
    scarcity = {
        p: df.loc[df["position"] == p].nlargest(5, "vor")["vor"].mean()
        for p in POSITIONS
    }
    order = sorted(scarcity, key=scarcity.get, reverse=True)
    lines.append("**Position priority** (by elite-tier value over replacement): " +
                 " → ".join(f"{p} ({scarcity[p]:.0f})" for p in order))
    lines.append("")

    # Superflex strategy
    if superflex_slots > 0:
        n_qb = int(round(starters["QB"]))
        qb_repl = repl["QB"]
        qbs = df[df["position"] == "QB"].nlargest(n_qb, "proj_pts")
        lines.append(f"**Superflex playbook.** Roughly **{n_qb} QBs** will be started league-wide each "
                     f"week, so replacement level is QB{n_qb} (~{qb_repl:.0f} pts). In superflex drafts, "
                     "startable QBs typically vanish in the first 3–4 rounds. Plan:")
        lines.append(f"- Leave the first two rounds with **at least one QB**; aim to have **two** by round 4–5.")
        if len(qbs) >= 12:
            t2_end = qbs.iloc[11]["player"]
            lines.append(f"- The QB cliff in your projections comes after **{t2_end}** "
                         f"(QB12, {qbs.iloc[11]['proj_pts']:.0f} pts).")
        lines.append("- A 3rd QB late is a real asset here — bye-week cover plus trade leverage.")
    else:
        qb_gap = df[df["position"] == "QB"].nlargest(3, "vor")["vor"].mean()
        lines.append(f"**Single-QB league.** QB replacement level is high, so elite QBs carry only "
                     f"~{qb_gap:.0f} pts of edge. Waiting on QB until the middle rounds is usually right "
                     "unless a top-3 QB falls well past their tier.")
    lines.append("")

    # Scoring-driven advice
    if pts_per_rec >= 1.0:
        top_rec = df[df["position"].isin(["RB", "WR"])].nlargest(5, "receptions")
        names = ", ".join(top_rec["player"].head(3))
        lines.append(f"**Full PPR tilt.** Each reception is a full point, so target-hogs gain ground on "
                     f"TD-dependent players. Highest projected reception volume: {names}. "
                     "Pass-catching RBs get a meaningful floor boost; one-dimensional grinders lose value.")
    elif pts_per_rec == 0:
        lines.append("**Standard scoring.** Yardage and TDs are everything — favor goal-line backs and "
                     "deep threats over possession receivers; volume receivers lose their PPR cushion.")
    if pass_td_pts >= 6:
        lines.append(f"**{pass_td_pts:.0f}-pt passing TDs** push QBs up roughly a full tier — treat QB "
                     "like a premium position even outside superflex.")
    if te_bonus > 0:
        top_te = df[df["position"] == "TE"].nlargest(3, "proj_pts")["player"].tolist()
        lines.append(f"**TE premium** makes elite TEs ({', '.join(top_te)}) legitimate round 1–2 picks; "
                     "the mid-TE dead zone gets worse, so go elite or punt.")
    lines.append("")

    # Tier cliffs
    lines.append("**Tier cliffs** — picks by which each position's top two tiers are likely gone "
                 "(based on overall value rank):")
    for p in POSITIONS:
        pos_df = df[df["position"] == p]
        t2 = pos_df[pos_df["tier"] <= 2]
        if len(t2):
            last = t2.nlargest(1, "overall_rank").iloc[0]
            rnd = int(np.ceil(last["overall_rank"] / teams))
            lines.append(f"- {p}: tier 2 ends with **{last['player']}** "
                         f"(~pick {int(last['overall_rank'])}, round {rnd})")
    lines.append("")

    # Value vs ADP
    if has_adp and "adp" in df.columns and df["adp"].notna().any():
        val = df[df["adp"].notna()].copy()
        val["value"] = val["adp"] - val["overall_rank"]
        steals = val[val["value"] >= teams].nlargest(6, "value")
        reaches = val[val["value"] <= -teams].nsmallest(6, "value")
        if len(steals):
            lines.append("**Values vs market ADP** (projections rank them ≥1 round earlier than drafters take them):")
            for _, r in steals.iterrows():
                lines.append(f"- {r['player']} ({r['position']}) — your rank {int(r['overall_rank'])}, "
                             f"market ADP {r['adp']:.0f}")
        if len(reaches):
            lines.append("")
            lines.append("**Market reaches** (drafted ~1+ round ahead of your projections — let someone else pay up):")
            for _, r in reaches.iterrows():
                lines.append(f"- {r['player']} ({r['position']}) — market ADP {r['adp']:.0f}, "
                             f"your rank {int(r['overall_rank'])}")
        lines.append("")

    # Injury-discount targets
    if "injury_weeks" in df.columns:
        risky = df[(df["weeks_out"].fillna(0) >= 3) & (df["overall_rank"] <= teams * 8)]
        if len(risky):
            names = ", ".join(f"{r['player']} ({int(r['weeks_out'])} wks out)" for _, r in risky.head(5).iterrows())
            lines.append(f"**Injury history flags in your draft range:** {names}. "
                         "Discount a round, or pair with their handcuff/backup.")
            lines.append("")

    lines.append("**General principles for this build:** draft the steepest cliff first, take the best "
                 "VOR on the board when tiers are deep, and bank late-round picks on upside (ambiguous "
                 "backfields, year-2 receivers) rather than safe floors — replacement level is free on waivers.")
    return "\n".join(lines)
