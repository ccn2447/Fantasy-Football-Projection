"""
Mock draft simulator.

The AI teams draft off a blended board (market ADP + your VOR projections) with
Gaussian noise, plus roster-need logic. Because the VOR side of the board is
already superflex- and scoring-aware, a superflex league automatically produces
a superflex-looking draft — QBs fly off the board early without any special case.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data_sources import POSITIONS

POS_CODE = {p: i for i, p in enumerate(POSITIONS)}

STRATEGIES = [
    "Balanced (VOR + roster need)",
    "Best player available (VOR)",
    "Follow ADP (market)",
    "Zero RB",
    "Hero RB",
    "Robust RB",
    "Late-round QB",
]


@dataclass
class DraftConfig:
    teams: int = 12
    rounds: int = 15
    my_slot: int = 1                    # 1-based draft position
    snake: bool = True
    adp_weight: float = 0.6             # 0 = pure your-projections, 1 = pure market ADP
    sigma: float = 6.0                  # AI unpredictability, in board-rank points
    my_strategy: str = STRATEGIES[0]
    qb_slots: int = 1
    rb_slots: int = 2
    wr_slots: int = 2
    te_slots: int = 1
    flex_slots: int = 1
    superflex_slots: int = 0
    seed: int = 7
    starters: dict = field(default_factory=dict)   # from projections.replacement_levels

    # ---- derived helpers -------------------------------------------------
    @property
    def n_picks(self) -> int:
        return self.teams * self.rounds

    def starter_need(self) -> dict:
        """Roughly how many of each position a team wants as starters."""
        return {
            "QB": self.qb_slots + self.superflex_slots,
            "RB": self.rb_slots + (1 if self.flex_slots else 0),
            "WR": self.wr_slots + (1 if self.flex_slots else 0),
            "TE": self.te_slots,
        }

    def roster_max(self) -> dict:
        """Hard caps so the AI doesn't hoard one position."""
        need = self.starter_need()
        return {
            "QB": min(need["QB"] + 1, 4),
            "RB": need["RB"] + 3,
            "WR": need["WR"] + 3,
            "TE": min(need["TE"] + 1, 3),
        }


# ----------------------------------------------------------------------------
# Draft order
# ----------------------------------------------------------------------------
def pick_slots(cfg: DraftConfig) -> np.ndarray:
    """0-based team slot making each pick, in pick order."""
    order = []
    for rnd in range(cfg.rounds):
        rnd_order = list(range(cfg.teams))
        if cfg.snake and rnd % 2 == 1:
            rnd_order = rnd_order[::-1]
        order += rnd_order
    return np.array(order)


def my_pick_numbers(cfg: DraftConfig) -> list[int]:
    slots = pick_slots(cfg)
    return [int(i + 1) for i, s in enumerate(slots) if s == cfg.my_slot - 1]


# ----------------------------------------------------------------------------
# Player pool / board
# ----------------------------------------------------------------------------
def build_pool(df: pd.DataFrame, cfg: DraftConfig, pad: int = 80) -> pd.DataFrame:
    keep = ["player", "position", "team", "proj_pts", "vor", "overall_rank"]
    keep = [c for c in keep if c in df.columns]
    pool = df.nsmallest(min(cfg.n_picks + pad, len(df)), "overall_rank")[keep].copy()

    if "adp" in df.columns and df["adp"].notna().any():
        pool["adp"] = df.loc[pool.index, "adp"]
    else:
        pool["adp"] = np.nan
    pool = pool.reset_index(drop=True)

    proj_rank = pool["proj_pts"].rank(ascending=False, method="first")
    vor_rank = pool["vor"].rank(ascending=False, method="first")
    # Undrafted-by-the-market players sit a couple of rounds behind their projection rank
    market_rank = pool["adp"].rank(method="first")
    market_rank = market_rank.fillna(vor_rank + cfg.teams * 2)

    w = float(np.clip(cfg.adp_weight, 0, 1))
    pool["board"] = w * market_rank + (1 - w) * vor_rank
    pool["proj_rank"] = proj_rank
    pool["pos_code"] = pool["position"].map(POS_CODE).astype(int)
    return pool


# ----------------------------------------------------------------------------
# Pick logic
# ----------------------------------------------------------------------------
def _need_penalty(counts: dict, cfg: DraftConfig, picks_left: int) -> np.ndarray:
    """Per-position adjustment (in board-rank points) for one drafting team.
    Negative = more attractive, large positive = effectively off the board."""
    need = cfg.starter_need()
    caps = cfg.roster_max()
    pen = np.zeros(len(POSITIONS))
    unfilled = sum(max(need[p] - counts.get(p, 0), 0) for p in POSITIONS)

    for p in POSITIONS:
        have = counts.get(p, 0)
        if have >= caps[p]:
            pen[POS_CODE[p]] = 1e6
        elif have >= need[p]:
            pen[POS_CODE[p]] = 10.0          # depth pick — mild deprioritization
        elif have == 0 and picks_left <= unfilled + 1:
            pen[POS_CODE[p]] = -25.0         # must-fill before the draft ends
    return pen


def _strategy_bonus(strategy: str, rnd: int, counts: dict, cfg: DraftConfig) -> np.ndarray:
    """Your own auto-pick tilt, applied on top of need penalties."""
    b = np.zeros(len(POSITIONS))
    rb, wr, te, qb = (POS_CODE[p] for p in ("RB", "WR", "TE", "QB"))

    if strategy == "Best player available (VOR)":
        b[:] = 0
    elif strategy == "Zero RB":
        if rnd <= 5:
            b[rb] += 45
            b[wr] -= 8
            b[te] -= 4
        elif rnd >= 7:
            b[rb] -= 12
    elif strategy == "Hero RB":
        if rnd == 1:
            b[rb] -= 20
        elif 2 <= rnd <= 6 and counts.get("RB", 0) >= 1:
            b[rb] += 35
            b[wr] -= 6
    elif strategy == "Robust RB":
        if rnd <= 3:
            b[rb] -= 18
        elif rnd >= 8 and counts.get("RB", 0) < 4:
            b[rb] -= 6
    elif strategy == "Late-round QB":
        if rnd <= 7 and counts.get("QB", 0) >= (1 if cfg.superflex_slots else 0):
            b[qb] += 60
        elif rnd >= 8:
            b[qb] -= 10
    return b


def _choose(pool_board: np.ndarray, pos_code: np.ndarray, taken: np.ndarray,
            counts: dict, cfg: DraftConfig, pick_no: int, rng: np.random.Generator,
            is_me: bool) -> int:
    avail = np.flatnonzero(~taken)
    if avail.size == 0:
        return -1
    picks_left = int(np.ceil((cfg.n_picks - pick_no + 1) / cfg.teams))
    rnd = int(np.ceil(pick_no / cfg.teams))

    adj = _need_penalty(counts, cfg, picks_left)
    if is_me:
        if cfg.my_strategy == "Follow ADP (market)":
            adj = np.zeros(len(POSITIONS))
        adj = adj + _strategy_bonus(cfg.my_strategy, rnd, counts, cfg)
        noise = 0.0        # your auto-picks are deterministic
    else:
        # Managers get less disciplined as the draft wears on
        scale = cfg.sigma * (0.6 + 0.9 * pick_no / cfg.n_picks)
        noise = rng.normal(0, scale, size=avail.size)

    key = pool_board[avail] + adj[pos_code[avail]] + noise
    return int(avail[int(np.argmin(key))])


# ----------------------------------------------------------------------------
# Draft state (stepable, so the UI can pause on your picks)
# ----------------------------------------------------------------------------
def new_state(cfg: DraftConfig, n_players: int) -> dict:
    return {
        "pick_no": 1,
        "slots": pick_slots(cfg),
        "taken": np.zeros(n_players, dtype=bool),
        "counts": [{} for _ in range(cfg.teams)],
        "log": [],          # list of (pick_no, slot, pool_idx)
    }


def step(state: dict, pool: pd.DataFrame, cfg: DraftConfig, forced_idx: int | None = None) -> bool:
    """Make one pick. Returns False when the draft is over."""
    pick_no = state["pick_no"]
    if pick_no > cfg.n_picks:
        return False
    slot = int(state["slots"][pick_no - 1])
    is_me = slot == cfg.my_slot - 1
    rng = np.random.default_rng([cfg.seed, pick_no])

    if forced_idx is not None and not state["taken"][forced_idx]:
        idx = int(forced_idx)
    else:
        idx = _choose(pool["board"].to_numpy(), pool["pos_code"].to_numpy(),
                      state["taken"], state["counts"][slot], cfg, pick_no, rng, is_me)
    if idx < 0:
        return False

    state["taken"][idx] = True
    pos = pool.at[idx, "position"]
    state["counts"][slot][pos] = state["counts"][slot].get(pos, 0) + 1
    state["log"].append((pick_no, slot, idx))
    state["pick_no"] = pick_no + 1
    return True


def run_to(state: dict, pool: pd.DataFrame, cfg: DraftConfig,
           stop_before_my_pick: bool = True) -> dict:
    """Advance until it's your turn (or the draft ends)."""
    while state["pick_no"] <= cfg.n_picks:
        slot = int(state["slots"][state["pick_no"] - 1])
        if stop_before_my_pick and slot == cfg.my_slot - 1:
            break
        if not step(state, pool, cfg):
            break
    return state


def simulate_full(cfg: DraftConfig, pool: pd.DataFrame) -> dict:
    state = new_state(cfg, len(pool))
    while step(state, pool, cfg):
        pass
    return state


def results_frame(state: dict, pool: pd.DataFrame, cfg: DraftConfig) -> pd.DataFrame:
    rows = []
    for pick_no, slot, idx in state["log"]:
        p = pool.iloc[idx]
        rows.append({
            "pick": pick_no,
            "round": int(np.ceil(pick_no / cfg.teams)),
            "slot": slot + 1,
            "manager": "YOU" if slot == cfg.my_slot - 1 else f"Team {slot + 1}",
            "player": p["player"], "position": p["position"], "nfl_team": p.get("team"),
            "proj_pts": p["proj_pts"], "vor": p["vor"], "adp": p.get("adp"),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["reach"] = (out["adp"] - out["pick"]).round(0)   # + = fell past market ADP
    return out


def board_grid(picks: pd.DataFrame, cfg: DraftConfig) -> pd.DataFrame:
    """Round × draft-slot grid of the full draft."""
    if picks.empty:
        return pd.DataFrame()
    g = picks.copy()
    g["cell"] = g["player"] + " (" + g["position"] + ")"
    grid = g.pivot(index="round", columns="slot", values="cell")
    grid.columns = [f"{'★ ' if c == cfg.my_slot else ''}Slot {c}" for c in grid.columns]
    return grid


# ----------------------------------------------------------------------------
# Roster evaluation
# ----------------------------------------------------------------------------
def starting_lineup(roster: pd.DataFrame, cfg: DraftConfig) -> pd.DataFrame:
    """Greedy best legal starting lineup, in points order."""
    remaining = roster.sort_values("proj_pts", ascending=False).copy()
    rows = []
    fixed = [("QB", cfg.qb_slots), ("RB", cfg.rb_slots), ("WR", cfg.wr_slots), ("TE", cfg.te_slots)]
    for pos, n in fixed:
        pick = remaining[remaining["position"] == pos].head(n)
        for _, r in pick.iterrows():
            rows.append({**r.to_dict(), "lineup_slot": pos})
        remaining = remaining.drop(pick.index)
    for _ in range(int(cfg.flex_slots)):
        pick = remaining[remaining["position"].isin(["RB", "WR", "TE"])].head(1)
        for _, r in pick.iterrows():
            rows.append({**r.to_dict(), "lineup_slot": "FLEX"})
        remaining = remaining.drop(pick.index)
    for _ in range(int(cfg.superflex_slots)):
        pick = remaining.head(1)
        for _, r in pick.iterrows():
            rows.append({**r.to_dict(), "lineup_slot": "SUPERFLEX"})
        remaining = remaining.drop(pick.index)
    for _, r in remaining.iterrows():
        rows.append({**r.to_dict(), "lineup_slot": "BENCH"})
    return pd.DataFrame(rows)


def grade_teams(picks: pd.DataFrame, cfg: DraftConfig) -> pd.DataFrame:
    """Rank every team by projected starting-lineup points."""
    rows = []
    for slot, grp in picks.groupby("slot"):
        lineup = starting_lineup(grp, cfg)
        starters = lineup[lineup["lineup_slot"] != "BENCH"]
        rows.append({
            "slot": slot,
            "manager": grp["manager"].iloc[0],
            "starter_pts": round(float(starters["proj_pts"].sum()), 1),
            "bench_pts": round(float(lineup[lineup["lineup_slot"] == "BENCH"]["proj_pts"].sum()), 1),
            "total_vor": round(float(grp["vor"].sum()), 1),
        })
    out = pd.DataFrame(rows).sort_values("starter_pts", ascending=False).reset_index(drop=True)
    out["draft_rank"] = out.index + 1
    return out


# ----------------------------------------------------------------------------
# Monte Carlo: who is actually likely to be there at each of your picks?
# ----------------------------------------------------------------------------
def availability(cfg: DraftConfig, pool: pd.DataFrame, n_sims: int = 40,
                 top_n: int = 60) -> pd.DataFrame:
    """P(player is still on the board) at each of your picks, across n_sims drafts."""
    my_picks = my_pick_numbers(cfg)
    taken_at = np.full((n_sims, len(pool)), cfg.n_picks + 1, dtype=np.int32)

    for s in range(n_sims):
        sim_cfg = DraftConfig(**{**cfg.__dict__, "seed": cfg.seed + 1000 * (s + 1)})
        state = simulate_full(sim_cfg, pool)
        for pick_no, _slot, idx in state["log"]:
            taken_at[s, idx] = pick_no

    rows = []
    order = pool["board"].to_numpy().argsort()[:top_n]
    for idx in order:
        row = {"player": pool.at[idx, "player"], "position": pool.at[idx, "position"],
               "proj_pts": pool.at[idx, "proj_pts"], "adp": pool.at[idx, "adp"],
               "avg_pick": float(np.mean(np.minimum(taken_at[:, idx], cfg.n_picks + 1)))}
        for p in my_picks:
            row[f"P{p}"] = float(np.mean(taken_at[:, idx] >= p))
        rows.append(row)
    return pd.DataFrame(rows)
