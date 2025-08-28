from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from cpp_core import DraftState, TeamState, Simulator, AgentConfig, SimConfig


def default_position_map() -> Dict[str, int]:
    # Keep stable small ints for C++ core
    positions = [
        "QB",
        "RB",
        "WR",
        "TE",
        "DL",
        "LB",
        "DB",
        "IDP",
        "FLEX",
        "B",
    ]
    return {p: i for i, p in enumerate(positions)}


def build_pos_index(series: pd.Series, pos_map: Dict[str, int]) -> np.ndarray:
    def to_idx(pos: str) -> int:
        # For multi-eligible positions use the first listed token
        key = pos.split(",")[0].strip()
        return pos_map.get(key, -1)

    return series.astype(str).map(to_idx).to_numpy(dtype=np.int32)


def build_team_states(
    team_budgets: Sequence[int],
    team_needs_by_pos: Sequence[Sequence[int]],
) -> List[TeamState]:
    teams: List[TeamState] = []
    for ridx, (budget, needs) in enumerate(zip(team_budgets, team_needs_by_pos)):
        t = TeamState()
        t.roster_id = int(ridx)
        t.budget = int(budget)
        t.need_by_pos = np.asarray(needs, dtype=np.int32)
        teams.append(t)
    return teams


def build_draft_state(
    draft_df: pd.DataFrame,
    pos_map: Dict[str, int],
    team_budgets: Sequence[int],
    team_needs_by_pos: Sequence[Sequence[int]],
) -> DraftState:
    ds = DraftState()
    ds.sleeper_ids = [int(x) for x in draft_df["sleeper_id"].to_numpy()]  # type: ignore[arg-type]
    ds.drafted_by = draft_df["Drafted"].to_numpy(dtype=np.int32)
    ds.draft_price = draft_df["Draft$"].to_numpy(dtype=np.int32)
    ds.pos_idx = build_pos_index(draft_df["Position"], pos_map)
    ds.teams = build_team_states(team_budgets, team_needs_by_pos)
    ds.n_positions = int(max(pos_map.values()) + 1)
    # Extra indices for FLEX and BENCH if present in pos_map
    ds.flex_pos_idx = int(pos_map.get("FLEX", -1))
    ds.bench_pos_idx = int(pos_map.get("B", -1))
    return ds


def build_agent_configs(n_teams: int, seed: int | None = None) -> List[AgentConfig]:
    """Create varied agent configs.

    Randomize mixture weights and behavior so agents don't behave identically.
    The random seed can be provided for reproducibility.
    """
    rng = np.random.default_rng(seed)
    cfgs: List[AgentConfig] = []
    for _ in range(n_teams):
        c = AgentConfig()
        c.alpha_points = 1.0
        # Sample mixture over site/hist/vorp and keep some mass for points
        vec = rng.dirichlet([2.0, 2.0, 1.0])  # site, hist, vorp
        scale = float(rng.uniform(0.6, 0.9))  # leave 10–40% for points
        c.w_site = float(scale * vec[0])
        c.w_hist = float(scale * vec[1])
        c.w_vorp = float(scale * vec[2])
        # Behavior
        c.aggressiveness = float(rng.uniform(0.8, 1.05))
        c.noise_sigma = float(rng.uniform(0.1, 0.25))
        c.patience = float(rng.uniform(0.3, 0.7))
        c.nomination_mode = int(rng.integers(0, 3))  # 0=random,1=drain,2=target
        cfgs.append(c)
    return cfgs


def compute_vorp_prices(
    draft_df: pd.DataFrame,
    teams: int,
    roster_size: int,
    my_budget: int,
    league_budget: int,
) -> np.ndarray:
    """
    Compute VORP-based dollar prices using the legacy EDPV approach:
      edpv = remaining_excess_dollars / remaining_value
      price_i = max(1, VORP_i * edpv) for undrafted players with VORP>0
    Drafted players keep their realized Draft$ if available.
    """
    df = draft_df.copy()
    # Players rostered: non-zero Drafted means rostered in current UI semantics
    players_rostered = int((df["Drafted"].astype(int) != 0).sum())
    n_left = teams * roster_size - players_rostered
    remaining_excess_dollars = int(my_budget + league_budget - n_left)
    value_mask = df["Drafted"].astype(int) == 0
    positive_vorp_mask = df["VORP"].astype(float) > 0
    remaining_value = float(df.loc[value_mask & positive_vorp_mask, "VORP"].sum())
    if remaining_value <= 0:
        # Fallback to ones to avoid divide-by-zero
        return np.maximum(
            1.0, df.get("Draft$", pd.Series(1, index=df.index)).to_numpy(dtype=float)
        )
    edpv = remaining_excess_dollars / remaining_value
    vorp_prices = np.ones(len(df), dtype=float)
    # Use realized draft prices when known and > 0
    if "Draft$" in df.columns:
        realized = df["Draft$"].to_numpy(dtype=float)
        drafted_mask = df["Drafted"].astype(int) != 0
        vorp_prices[drafted_mask] = np.maximum(1.0, realized[drafted_mask])
    # Undrafted with positive VORP get VORP*edpv
    undrafted_pos = (df["Drafted"].astype(int) == 0) & (df["VORP"].astype(float) > 0)
    vorp_prices[undrafted_pos.to_numpy()] = np.maximum(
        1.0, (df.loc[undrafted_pos, "VORP"].to_numpy(dtype=float) * edpv)
    )
    return vorp_prices


def run_simulation(
    projection_table,
    player_ids: Sequence[int],
    pos_idx: np.ndarray,
    site_prices: np.ndarray,
    hist_curve: np.ndarray,
    vorp_prices: np.ndarray,
    draft_state: DraftState,
    n_sims: int = 100,
    seed: int = 0,
    agent_cfgs: List[AgentConfig] | None = None,
    my_roster_id: int | None = None,
    take: bool | None = None,
    focal_sid: int | None = None,
    focal_price: int | None = None,
):
    sim = Simulator()
    sim.set_projection_table(projection_table)
    sim.set_players([int(x) for x in player_ids], pos_idx.astype(np.int32))
    sim.set_market_curves(site_prices.astype(float), hist_curve.astype(float))
    sim.set_vorp_prices(vorp_prices.astype(float))
    cfgs = (
        agent_cfgs
        if agent_cfgs is not None
        else build_agent_configs(len(draft_state.teams), seed=seed)
    )
    sc = SimConfig()
    sc.n_sims = int(n_sims)
    sc.seed = int(seed)
    if (
        take is not None
        and focal_sid is not None
        and focal_price is not None
        and my_roster_id is not None
    ):
        sc.condition_take = bool(take)
        sc.focal_sid = int(focal_sid)
        sc.focal_price = int(focal_price)
        sc.my_roster_id = int(my_roster_id)
    return sim.run(draft_state, cfgs, sc)


def calibrate_from_pick(
    agent_cfgs: List[AgentConfig],
    roster_id: int,
    player_index: int,
    observed_price: float,
    site_prices: np.ndarray,
    hist_curve: np.ndarray,
    vorp_prices: np.ndarray,
    lr: float = 0.2,
) -> None:
    """
    Update the target agent's mixture weights toward the anchor that best explains
    the observed price. Lightweight EMA toward site/hist/vorp with learning rate lr.
    """
    if roster_id < 0 or roster_id >= len(agent_cfgs):
        return
    cfg = agent_cfgs[roster_id]
    # Gather anchors
    site_val = (
        float(site_prices[player_index])
        if player_index < len(site_prices)
        else observed_price
    )
    hist_val = (
        float(hist_curve[player_index])
        if player_index < len(hist_curve)
        else observed_price
    )
    vorp_val = (
        float(vorp_prices[player_index])
        if player_index < len(vorp_prices)
        else observed_price
    )
    anchors = np.array([site_val, hist_val, vorp_val], dtype=float)
    # Choose closest anchor
    diffs = np.abs(anchors - observed_price)
    best = int(np.argmin(diffs))
    # Current weights (ensure non-negative)
    w_site, w_hist, w_vorp = (
        max(cfg.w_site, 0.0),
        max(cfg.w_hist, 0.0),
        max(cfg.w_vorp, 0.0),
    )
    w_pts = max(0.0, 1.0 - (w_site + w_hist + w_vorp))
    w = np.array([w_site, w_hist, w_vorp, w_pts], dtype=float)
    # Move probability mass toward the winning anchor by lr
    move = min(max(lr, 0.0), 1.0)
    inc = move * (1.0 - w[best])
    dec = move
    # Reduce others proportionally from their mass
    total_other = w.sum() - w[best]
    if total_other > 1e-8:
        w[best] += inc
        for i in range(4):
            if i == best:
                continue
            w[i] -= dec * (w[i] / total_other)
    # Renormalize to sum<=1.0 and map back to config
    total = w.sum()
    if total > 1.0:
        w /= total
    cfg.w_site, cfg.w_hist, cfg.w_vorp = float(w[0]), float(w[1]), float(w[2])
    # Slightly adjust aggressiveness toward observed/anchor ratio
    denom = max(anchors[best], 1.0)
    ratio = observed_price / denom
    cfg.aggressiveness = float(
        max(0.5, min(2.0, 0.9 * cfg.aggressiveness + 0.1 * ratio))
    )
