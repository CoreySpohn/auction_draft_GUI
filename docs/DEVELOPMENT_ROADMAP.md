## Auction Draft GUI — Refactor Roadmap

This document proposes a phased plan to refactor the current monolithic `package/mainwindow.py` into a clean and modular architecture. The goals are simpler maintenance and better performance/UX responsiveness, while minimizing testing and CI overhead because this is a temporary solution.

### Core Principles
- Keep modules small (<200 lines when practical) and single‑purpose.
- Separate concerns: GUI, domain logic, data access, optimization, and plotting should live in separate layers.
- Prefer pure functions and immutable data for core calculations.
- Add type hints and docstrings (Google-style); keep lightweight sanity checks and defer automated tests/CI for now.
- Avoid blocking the GUI thread; offload IO/CPU tasks to workers.

### Current Pain Points (from `package/mainwindow.py`)
- Mixed responsibilities: scraping, modeling, optimization, plotting, state management, and UI are tightly coupled.
- Long file (~2.8k lines) with many implicit dependencies and magic constants (e.g., `Drafted` values 0/1/2).
- Blocking operations in UI thread (Selenium, file IO, multiprocessing pools) risk UI freezes.
- Business logic encoded in ad-hoc DataFrame manipulations without typed models or validation.
- Plotting logic intertwined with pricing/optimization logic, making both hard to test.

### Target Architecture

Layered, modular structure with clear boundaries:

```
package/
  app/                    # GUI-only: Qt widgets, signals/slots, presenters
    main_window.py        # Thin layer orchestrating services via signals
    widgets/              # Reusable UI components

  domain/                 # Core entities and simple logic (no IO/GUI)
    models.py             # Player, LeagueRoster, ScoringRules, DraftRow, Enums
    drafting.py           # DraftState operations (pure, dataframe-agnostic if possible)

  data/                   # Data access, caching, adapters (IO only)
    repositories.py       # DraftStateRepository (CSV), SitePriceRepository
    sources/
      fantasypros.py      # NFL projections fetch/parse
      fftoday_idp.py      # IDP projections fetch/parse
      hashtag_bball.py    # NBA rankings/values fetch/parse
      espn_values.py      # ESPN auction values fetch/parse
    cache.py              # Local cache mgmt, path helpers

  analysis/               # Math/modeling (pure/numpy), no GUI
    vorp.py               # VORP calculations (NFL/NBA variants)
    pricing.py            # Price models (LOWESS, Lasso, site blends)
    opportunity_cost.py   # OC curve computation

  optimization/           # OR-Tools CP-SAT models
    team_optimizer.py     # find_optimal_team_sat() + helpers

  plotting/               # Matplotlib-only views; pure w.r.t. computation
    price_plot.py         # Price vs. rank
    oc_plot.py            # Opp. cost plots
    palettes.py           # Color utilities

  services/               # Orchestration combining data + analysis + optimization
    draft_state_manager.py  # State transitions, events, budgets
    projection_adapter.py   # Existing adapter; align to new interfaces

  config/
    settings.py           # App settings (pydantic or simple dataclass)
```

Notes:
- Keep DataFrame usage at the edges (data adapters) and convert to typed records for core logic where feasible. If DF is retained, centralize schema in one place.
- Replace magic codes with Enums (e.g., `Drafted.NONE/OTHER/MINE`).
- Provide a thin `app/main_window.py` to wire signals to service methods; no business logic inside UI.

### Phased Plan

#### Phase 0 — Hygiene and Guardrails
- Introduce minimal linting (`ruff`) and basic type hints where helpful; skip CI and formal test scaffolding for now.
- Define central constants and Enums (draft status, sports, positions).

Deliverables:
- `pyproject.toml` with minimal linting config.
- `package/domain/models.py` with Enums and typed records.

#### Phase 1 — Extract Optimization and Analysis
- Move `find_optimal_team_sat()` and helpers to `optimization/team_optimizer.py`.
- Move VORP pricing, OC calculations, and price modeling (LOWESS/Lasso blend) to `analysis/` as pure functions.

Deliverables:
- `package/optimization/team_optimizer.py`.
- `package/analysis/{vorp.py,pricing.py,opportunity_cost.py}`.

#### Phase 2 — Data Sources and Repositories
- Extract all scraping/parsing into `data/sources/*`. Replace Selenium usage where feasible with requests+parsing; if Selenium remains, run in a worker and add retries/timeouts.
- Implement repositories for draft state and site prices (CSV/Parquet). Centralize `.cache` paths and naming in `data/cache.py`.
- Provide a stable interface for `services` to retrieve/update data.

Deliverables:
- `package/data/sources/*.py` with clear function contracts.
- `package/data/repositories.py` and `package/data/cache.py`.

#### Phase 3 — Services and State Management
- Implement `DraftStateManager` to encapsulate:
  - Budget math, team roster counts, and state transitions
  - Derivations like `Proj$`, `OC`, and `OnOptTeam`
  - Event hooks for UI updates
- Replace direct `DataFrame` mutations in the UI with service calls returning immutable results or diffs.

Deliverables:
- `package/services/draft_state_manager.py` refined.

#### Phase 4 — Plotting and UI Separation
- Move plotting to `plotting/*` with simple, composable functions that accept plain arrays/records and return artists.
- Refactor `MainWindow` into `app/main_window.py`:
  - Signals/slots only
  - Background tasks via `QThreadPool`/`QRunnable` or `concurrent.futures` integrated with Qt event loop
  - No business logic or blocking IO

Deliverables:
- `package/plotting/{price_plot.py,oc_plot.py,palettes.py}`.
- `package/app/main_window.py` with minimal orchestration.

#### Phase 5 — Incremental API Hardening and Docs
- Add comprehensive type hints, docstrings, and examples.
- Document service interfaces and data contracts.

### Validation Approach (temporary)
- Manual validation through the GUI after each refactor step.
- Quick spot checks on small CSV samples; log diffs before/after major operations.
- Add assertions and lightweight runtime checks in critical paths (can be toggled off later).

### Performance and Responsiveness
- Use background workers for scraping/model fitting (Lasso/LOWESS) and CP-SAT optimization.
- Cache parsed inputs and intermediate models to avoid repeat computation.
- Prefer vectorized `numpy`/`pandas` operations in analysis modules.

### Data and Config
- Centralize `.cache` path and versioning in `data/cache.py`.
- Introduce `config/settings.py` for league rules, scoring, and UI defaults. Persist user overrides.

### Risks and Mitigations
- Selenium dependence can stall UI: isolate in worker + add graceful cancellation/timeouts. Consider pure-HTTP fallbacks where possible.
- Tight coupling to current DataFrame schema: add schema validators; introduce typed layers gradually.
- Large refactor risk: proceed incrementally; keep legacy facade functions so the UI compiles at each step.

### Concrete Tasks (first milestones)
- [ ] Add Enums/constants for `Drafted`, `Sport`, positions; replace magic numbers.
- [ ] Extract `find_optimal_team_sat()` to `optimization/team_optimizer.py`.
- [ ] Extract VORP and price models to `analysis/`.
- [ ] Create `data/repositories.py` and move CSV read/write logic out of UI.
- [ ] Introduce `DraftStateManager` facade in `services/` and refactor UI to call it.
- [ ] Move plotting into `plotting/` and refactor UI to consume these functions.
- [ ] Replace multiprocessing Pools in UI with a background worker abstraction.

### Non-Goals (for now)
- Rewriting the entire UI toolkit. We keep PyQt5 and the existing `.ui` layout, only thinning the main window.
- Replacing OR-Tools. We keep CP-SAT and improve constraints/behavior.

---

This roadmap enables incremental, low-risk progress. Each phase yields a smaller `MainWindow`, clearer abstractions, and better performance, while preserving current functionality during transition and avoiding heavy test/CI investments.


