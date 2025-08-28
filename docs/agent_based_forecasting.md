## Agent-Based Draft Forecasting: Detailed Implementation Plan

This document specifies the architecture, algorithms, and interfaces for a fast agent-based Monte Carlo forecaster for auction drafts. Agents and the auction engine will be implemented in C++ and exposed to Python via nanobind for live use during drafts. We will index all players and internal arrays by `sleeper_id` to ensure robust joins between data sources and the simulation core [[memory:7124524]].

### Goals and scope

- Implement a performant auction simulation engine in C++ that can run thousands of rollouts in seconds.
- Model owner behavior via agents with private valuations, needs-awareness, and budget-aware bidding.
- Condition forecasts on the live draft state (who is drafted, prices, budgets, roster needs).
- Provide APIs to: (a) simulate full drafts, (b) marginalize outcomes for a focal player price decision, and (c) update agent beliefs online from observed picks.
- Keep modules small (<200 LOC each) and interfaces simple. Prefer clear heuristics first; refine once baseline is correct and fast.

---

### High-level design

The forecaster executes many simulations of the remaining draft, each with:
1) a draw of player performance projections and value maps, and 2) a sequence of auctions governed by agent policies. Outputs include distributions of final rosters, spend, and prices. We also expose conditional queries used by opportunity-cost charts and optimal team introspection in the GUI.

- Core concepts:
  - Players: immutable metadata and current projected utility (PPW/season points) per simulation draw.
  - Draft state: budgets, roster needs, already-drafted players at known prices, nomination order cursor.
  - Agent: valuation function + bidding and nomination policy.
  - Auction engine: simulates ascending-price English auctions per nomination until rosters are filled.

We will leverage the existing `cpp_core::ProjectionTable` to sample performance per player and store agent valuation matrices for vectorized computation.

---

### C++ modules and files

Add the following small, focused headers/implementations under `package/cpp_core/`:

- include/cpp_core/agent.hpp (≈150 LOC)
  - `struct AgentConfig` (mixture weights, noise scales, aggressiveness, team preference priors)
  - `struct AgentState` (budget, roster needs by position, private valuation vector view)
  - `class Agent` with methods:
    - `void revalue(const Eigen::VectorXd& base_points, const Eigen::VectorXi& positions, const Eigen::VectorXd& site_prices, const Eigen::VectorXd& hist_curve, std::uint64_t seed)`
    - `double cap_for_player(std::int64_t sleeper_id, const DraftConstraints&) const`
    - `bool will_bid(std::int64_t sleeper_id, int current_price) const`
    - `int nominate(const DraftStateView&) const` (returns sleeper_id)

- include/cpp_core/market.hpp (≈150 LOC)
  - Transforms points to dollars given market constraints (budget left vs VORP supply; optional LOWESS from Python replaced by simple parametric curve and position scarcity multipliers). Provides:
    - `Eigen::VectorXd points_to_dollars(const Eigen::VectorXd& points, const MarketState&)`
    - Position scarcity reweighting based on roster vacancies and supply.

- include/cpp_core/draft_state.hpp (≈180 LOC)
  - `struct TeamState { int roster_id; int budget; Eigen::ArrayXi need_by_pos; }`
  - `struct DraftState { std::vector<std::int64_t> sleeper_ids; Eigen::ArrayXi drafted_by; Eigen::ArrayXi draft_price; std::vector<TeamState> teams; ... }`
  - Lightweight, POD-like views for bridging Python memory (avoid pandas in the core).

- include/cpp_core/auction.hpp (≈180 LOC)
  - Ascending auction runner:
    - `struct AuctionConfig { int min_increment; int start_price; }`
    - `struct AuctionResult { int winner_roster_id; int price; }`
    - `AuctionResult run_auction(const std::int64_t sid, const std::vector<Agent>& agents, DraftState& state, const AuctionConfig&)`

- include/cpp_core/simulator.hpp (≈180 LOC)
  - `struct SimConfig { int n_sims; std::uint64_t seed; bool condition_take; std::int64_t focal_sid; int focal_price; }`
  - `class Simulator` with:
    - `void set_players(const std::vector<std::int64_t>& ids, const Eigen::VectorXi& pos_idx)`
    - `void set_market_curves(const Eigen::VectorXd& site_prices, const Eigen::VectorXd& hist_curve)`
    - `void set_projection_table(const ProjectionTable* tbl)`
    - `SimSummary run(const DraftState& initial, const std::vector<AgentConfig>& agent_cfgs, const SimConfig&)`
  - `struct SimSummary { Eigen::ArrayXi final_owner; Eigen::ArrayXi final_price; Eigen::VectorXd price_mean; Eigen::VectorXd price_p10; Eigen::VectorXd price_p90; }`

- src/agent.cpp, market.cpp, draft_state.cpp, auction.cpp, simulator.cpp (each under ≈180 LOC, clear functions, minimal shared state).

- src/test/*.cpp
  - Unit tests for bidding logic, scarcity weighting, and auction outcomes.

Expose the key classes/functions in `src/bindings.cpp` under a new `cpp_core::forecast` namespace binding block.

---

### Data representations and indices

- Player identity: use `std::int64_t sleeper_id` everywhere [[memory:7124524]].
- Position encoding: map string positions to small ints on the Python side and pass `Eigen::VectorXi positions`; C++ holds a fixed mapping enum for speed.
- Drafted status: `drafted_by[i] ∈ {-1: undrafted, 0..N-1: roster_id}` with `draft_price[i] ≥ 0`.
- Team needs: fixed-length array of required counts per position (including FLEX/IDP groups expanded to position-level constraints for the core). FLEX and IDP will be represented as position capacity constraints, not as special positions, to keep the core linear.

---

### Valuation model (private agent values)

Reasoning: We need coherent, fast-to-compute valuations that reflect both skill (projected points) and market anchors (site prices, historical price curves), while adapting to roster scarcity and budgets. We will start with a calibrated linear mixture with noise and multiplicative scarcity adjustment. This is robust, transparent, and fast. Later, we can upgrade to a learned mapping or LOWESS emulation if needed.

For each agent and player i:

1) Base points draw: `p_i` from `ProjectionTable.sample_matrix` for the current simulation.
2) Points→baseline dollars: `d_points_i = alpha * p_i + beta * rank_term_i` (rank term approximates diminishing returns).
3) External anchors: `d_anchor_i = w_site * site_i + w_hist * hist_i`.
4) Scarcity multiplier: `m_pos = f(needed[pos], supply[pos])`, monotonic in needs/supply.
5) Idiosyncratic preferences: lognormal multiplicative noise `ε_i ~ LogNormal(0, σ)` and target/avoid nudges from the UI if available.

Private value: `v_i = m_pos * ( (1 - w_ext) * d_points_i + w_ext * d_anchor_i ) * ε_i`, capped by remaining budget and constrained by max price policy (e.g., leave $1 for each empty slot).

Parameters per agent (in `AgentConfig`): weights `alpha, beta, w_site, w_hist`, `σ`, aggressiveness, patience, nomination style.

---

### Bidding policy

Reasoning: True optimal bidding is a stochastic game. For speed and stability we use a cap-based policy that raises until the current price exceeds the agent’s cap for that player under current needs and remaining budget.

- Define `cap_i(agent)` computed from `v_i` with guardrails:
  - Enforce reserve: `cap ≤ budget - (slots_left - 1)`.
  - Enforce positional reserves (at least $1 per remaining starter/bench slot).
  - Inflate cap for must-fill scarce roles late in draft.

- During an auction:
  - All agents whose `cap ≥ current_price + increment` are “active”.
  - If multiple are active, next higher bid comes from one of the active agents (e.g., the one with highest slack `cap - current_price`, or sampled proportional to slack to create realistic unpredictability).
  - Terminate when only one active remains; record winner and price.

This yields realistic price dispersion and overpays when two agents highly value the same scarce role.

---

### Nomination policy

Reasoning: Owners often nominate players they don’t intend to buy, to drain budgets, or nominate must-get targets opportunistically. We implement three simple modes and sample per agent:

- Drainers: nominate top-ranked at crowded positions they don’t need.
- Targeters: nominate one of top-K wanted players at slightly below their cap.
- Randomized: weighted by market popularity (site_rank) with exclusions for already filled roles.

We start with a weighted mixture; parameters live in `AgentConfig`.

---

### Simulation loop (per rollout)

Pseudocode sketch:

```cpp
for (round = start_round; !done; ++round) {
  int nom_roster = current_nomination_roster(state);
  std::int64_t sid = agents[nom_roster].nominate(view_of(state));
  AuctionResult res = run_auction(sid, agents, state, auction_cfg);
  apply_result_to_state(res, sid, state);
  if (all_rosters_full(state)) break;
}
```

Conditioning (opportunity cost queries): For a focal player and assumed price, force assign to “my roster” at that price before running the loop; compare outcomes to the leave case over many simulations.

---

### Python ↔ C++ interface (nanobind)

Expose minimal, array-based APIs. Example signatures:

```cpp
// bindings
class AgentConfig { /* POD fields, no methods */ };
class TeamInit { public: int roster_id; int budget; std::vector<int> need_by_pos; };

class Simulator {
 public:
  void set_projection_table(ProjectionTable* tbl);
  void set_players(std::vector<std::int64_t> ids, std::vector<int> pos_idx);
  void set_market_curves(std::vector<double> site_prices, std::vector<double> hist_curve);
  SimSummary run(
    DraftState initial,
    std::vector<AgentConfig> agent_cfgs,
    SimConfig cfg);
};
```

Python usage from `mainwindow.py`:

```python
from cpp_core import Simulator, AgentConfig

sim = Simulator()
sim.set_projection_table(self.projection_table)
sim.set_players(player_ids, position_indices)
sim.set_market_curves(site_prices, hist_curve)

summary = sim.run(initial_state, agent_cfgs, SimConfig(
    n_sims=2000, seed=seed, condition_take=True,
    focal_sid=sid, focal_price=price
))

# summary.price_mean maps by sleeper_id index
```

Draft state translation in Python:
- Build `DraftState` by extracting from `self.draft_df` the columns: `sleeper_id`, `Drafted`→`drafted_by`, `Draft$`→`draft_price`, per-roster budgets and needs from `self.league_roster` minus current counts.
- We will add a thin adapter in `package/services/projection_adapter.py` or a new `services/forecast_adapter.py` for clean conversion.

---

### Online calibration from live picks

When an observed pick (sid, price, roster_id) arrives, update agent mixture weights to better fit revealed preferences:
- Simple rule: shift weights so that the winning agent’s cap would have exceeded price by a small margin and others’ caps would be below price.
- Maintain an exponential moving average per agent of `(observed_price / anchor_price)` by position to adapt to room-specific inflation.
- Optionally store per-opponent spend patterns and update aggressiveness.

These updates are cheap and can be applied between polling intervals.

---

### Performance considerations

- Use Eigen vectors/matrices for valuation and price computations; avoid per-player heap allocations.
- Reuse RNG streams via seed mixing (`mix_seed`) similar to `ProjectionTable`.
- Batch sample projections with `ProjectionTable.sample_matrix` once per rollout (or once globally per outer seed and reuse across agents).
- Keep auction loops tight: track active agents in a small vector, end early when only one remains.
- Avoid string ops in the core; Python prepares integer position indices.

Target: 1–2k simulations in <2 seconds on a typical laptop for ~250 players remaining and 10 teams.

---

### Testing strategy

- C++ unit tests (src/test):
  - Valuation monotonicity: higher points → higher cap when needs/budget fixed.
  - Budget guards: cannot overspend relative to slots remaining.
  - Auction single-bidder and two-bidder edge cases.
  - Scarcity: increasing needs raises caps for that position.

- Python integration tests (pytest):
  - End-to-end `Simulator.run` on a tiny toy draft (3 players, 2 teams) matches expected outcomes.
  - Opportunity-cost conditioning returns higher OC when focal price is inflated.

Use hypothesis to fuzz tiny drafts and assert invariants (budgets non-negative, each player sold at most once, roster sizes correct).

---

### Milestones

1) Skeleton types + bindings: `DraftState`, `AgentConfig`, `Simulator` (no logic). Build and import from Python.
2) Points→dollars market mapping and basic valuation (no scarcity, no noise). Unit tests pass.
3) Auction engine and cap-based bidding. Simulate a single auction deterministically; unit tests.
4) Full draft loop with random nominations. End-to-end tiny draft test in pytest.
5) Scarcity and budget guards; position capacity constraints; mixture anchors.
6) Conditioning API for OC curves; integrate with GUI hook in `mainwindow.py`.
7) Online calibration from observed picks; expose update entry point.
8) Performance pass and parameter tuning defaults; document configs.

---

### Future enhancements (post-MVP)

- Learnable valuation mapping per position via lightweight ridge regression fit online.
- Better nomination policies (game-theory-inspired bluff/decoy frequency control).
- Team-stacking or schedule correlation preferences.
- Weekly projections mode (`week > 0`) for in-season FAAB forecasting.

---

### Deliverables

- New C++ headers/sources listed above with nanobind bindings.
- Python adapter to build `DraftState` and call `Simulator`.
- Unit and integration tests with CI target runtime < 60s.
- Minimal user-facing controls to configure agent aggressiveness from the GUI (future PR).


---

### Current status (MVP implemented)

- C++ core:
  - `ProjectionTable` sampling integrated in `Simulator` to draw base points per sim (n_sims) and compute mean/p10/p90 price summaries.
  - `Agent` valuation now blends mapped points→$ (linear fit to site), site prices, historical curve, and VORP-derived prices via weights `w_site`, `w_hist`, `w_vorp`. Top-K scaling matches the site top end; aggressiveness+noise applied.
  - `Simulator` exposes `set_vorp_prices`. Percentiles computed across sims; seed mixing per (agent, sim).

- Python adapters/utilities:
  - `forecast_adapter.compute_vorp_prices`: VORP-dollar mapping mirroring legacy EDPV (remaining excess $ / remaining VORP).
  - `forecast_adapter.calibrate_from_pick`: lightweight online update of `AgentConfig` weights and aggressiveness toward the anchor explaining the observed price.
  - Replay script (`tools/replay_last_year.py`): replays 2024 Sleeper draft picks; updates team budgets/needs; tolerates unknown picks by using $1 placeholders; prints hypothetical team, prices, and PPW. Adds scarcity- and endgame-aware buy rule for experimentation.

- GUI smoke test: single-sim invocation from `mainwindow.py` to validate integration, now using `ProjectionTable` and VORP/site mixtures.

---

### Decision-time OC (take vs pass at price X) – what’s needed

Goal: When it’s my turn and a player is up at price X, run two conditional simulations many times:
1) TAKE: force assign player to my roster at $X, update budget/needs, then simulate remainder.
2) PASS: forbid me from buying the player at $X (or at all), simulate remainder (letting others win), then compare distributions of my final outcomes (e.g., PPW, value, dollars, roster composition).

Required updates:
- `Simulator`: implement conditioning flags in `SimConfig` (present but unused). Add parameters for `my_roster_id`, `(focal_sid, focal_price)`, and condition mode:
  - TAKE: set `drafted_by[focal_idx] = my_roster_id`, `draft_price = X`, decrement my team budget/needs before loop.
  - PASS: mark me ineligible for this player (e.g., cap=0 for me on focal player), allow others to bid/win.
- Auction loop: current MVP does greedy assignment by private value. For OC fidelity we need a minimal ascending auction:
  - Compute each agent cap (value with guards); step price by increment; last active wins.
  - This engine should be used for focal player (and ideally all nominations) to reflect competitive dynamics.
- Return metrics: per-condition summary of my final team utility (e.g., starter PPW sum), dollars, and deltas. Optionally return the estimated OC curve root (price where TAKE≈PASS).
- GUI: wire OC plots to call `Simulator` twice per price grid and render TAKE−PASS difference; reuse existing OC panel hooks.

---

### Near-term tasks

1) Implement conditional TAKE/PASS path inside `Simulator.run` using `SimConfig` and `my_roster_id`.
2) Add a minimal auction for at least the focal player; optionally for all nominations when n_sims is moderate.
3) Expose my-roster metrics in `SimSummary` (starter PPW sum, dollars spent/remaining, roster fill by position).
4) Hook OC computation in `mainwindow.py` replacing the Python-side pool logic with the C++ conditional simulator for speed.
5) Improve nomination policy and per-position scarcity in C++; remove Python-only heuristics.

---

### Lessons learned so far

- Percentiles: computing p10/p90 across sims avoids zeros; single-sim should mirror mean.
- Market scale: mapping points→$ with a site-calibrated linear fit plus top-K scaling is necessary for realistic top-end prices (~$100 in this league).
- Mixtures: adding `w_vorp` alongside site/hist improves robustness and better tracks room tendencies after calibration.
- Replay: unknown picks are common late; treating them as $1 placeholders stabilizes budget/needs evolution and avoids index errors.
- Buy rules: adding scarcity and endgame logic reduces leftover budget and better fills positional needs; these belong in the C++ agent caps/auction engine for fidelity and performance.


