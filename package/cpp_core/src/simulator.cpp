#include "cpp_core/simulator.hpp"
#include "cpp_core/auction.hpp"
#include <algorithm>
#include <iostream>

namespace cpp_core {

static inline std::uint64_t mix_seed(std::uint64_t a, std::uint64_t b) {
  std::uint64_t z = a + 0x9e3779b97f4a7c15ULL + (b << 6) + (b >> 2);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

void Simulator::set_players(const std::vector<std::int64_t> &ids,
                            const Eigen::VectorXi &pos_idx) {
  ids_ = ids;
  pos_idx_ = pos_idx;
}

void Simulator::set_market_curves(const Eigen::VectorXd &site_prices,
                                  const Eigen::VectorXd &hist_curve) {
  site_prices_ = site_prices;
  hist_curve_ = hist_curve;
}

SimSummary Simulator::run(const DraftState &initial,
                          const std::vector<AgentConfig> &agent_cfgs,
                          const SimConfig &cfg) const {
  const int sims = std::max(1, cfg.n_sims);
  SimSummary summary;
  const int n = static_cast<int>(initial.sleeper_ids.size());
  summary.final_owner = Eigen::ArrayXi::Constant(n, -1);
  summary.final_price = Eigen::ArrayXi::Zero(n);
  summary.price_mean = Eigen::VectorXd::Zero(n);
  summary.price_p10 = Eigen::VectorXd::Zero(n);
  summary.price_p90 = Eigen::VectorXd::Zero(n);

  // Precompute base points samples for all sims
  Eigen::MatrixXd base_mat;
  if (proj_tbl_ && !ids_.empty()) {
    base_mat = proj_tbl_->sample_matrix(ids_, static_cast<std::size_t>(sims), cfg.seed);
  } else {
    base_mat = Eigen::MatrixXd::Zero(n, sims);
    Eigen::VectorXd base = site_prices_.size() == n ? site_prices_ : Eigen::VectorXd::Ones(n);
    for (int k = 0; k < sims; ++k) base_mat.col(k) = base;
  }

  // Collect simulated prices per player per sim
  Eigen::MatrixXd price_samples(n, sims);

  for (int k = 0; k < sims; ++k) {
    // Create a mutable draft state for this rollout
    DraftState state = initial;

    // Note: assume Python adapter has already reconciled existing drafted players
    // into team budgets/needs if any picks exist. We avoid re-applying here to
    // prevent double-counting when callers pass pre-adjusted team states.

    // Optional TAKE conditioning (applied on top of existing drafted state)
    if (cfg.condition_take && cfg.my_roster_id >= 0) {
      // Map focal_sid into player index
      int focal_idx = -1;
      for (int i = 0; i < n; ++i) {
        if (ids_.empty()) break;
        if (ids_[static_cast<std::size_t>(i)] == cfg.focal_sid) { focal_idx = i; break; }
      }
      if (focal_idx >= 0 && state.drafted_by[focal_idx] < 0) {
        state.drafted_by[focal_idx] = cfg.my_roster_id;
        state.draft_price[focal_idx] = std::max(1, cfg.focal_price);
        // Update my team budget and needs
        if (cfg.my_roster_id < static_cast<int>(state.teams.size())) {
          TeamState &my = state.teams[static_cast<std::size_t>(cfg.my_roster_id)];
          my.budget -= state.draft_price[focal_idx];
          const int p = (focal_idx < pos_idx_.size()) ? pos_idx_[focal_idx] : -1;
          if (p >= 0 && p < my.need_by_pos.size()) {
            my.need_by_pos[p] = std::max(0, my.need_by_pos[p] - 1);
          }
        }
      }
    }

    // Build agents for this sim using the updated state
    std::vector<Agent> agents;
    agents.reserve(agent_cfgs.size());
    for (std::size_t i = 0; i < agent_cfgs.size(); ++i) {
      AgentState st;
      st.roster_id = static_cast<int>(i);
      st.budget = state.teams[i].budget;
      st.need_by_pos = state.teams[i].need_by_pos;
      st.private_value = Eigen::VectorXd::Ones(n);
      Agent ag(agent_cfgs[i], st);
      const std::uint64_t sseed = mix_seed(cfg.seed, (static_cast<std::uint64_t>(i) << 32) ^ static_cast<std::uint64_t>(k));
      ag.revalue(base_mat.col(k), state.pos_idx, site_prices_, hist_curve_, vorp_prices_, sseed);
      agents.push_back(ag);
    }

    // Debug: print private values summary + each agent's top-5 valuations on first sim
    if (k == 0 && !agents.empty()) {
      const int nn = static_cast<int>(agents[0].state().private_value.size());
      // Overall scale for agent 0
      {
        const auto &pv = agents[0].state().private_value;
        if (nn > 0) {
          double mean_pv = pv.mean();
          double max_pv = pv.maxCoeff();
          const int kk = std::max(1, std::min(nn, 20));
          std::vector<double> pv_copy(pv.data(), pv.data() + nn);
          std::vector<double> site_copy;
          if (site_prices_.size() == nn) {
            site_copy.assign(site_prices_.data(), site_prices_.data() + nn);
          }
          std::nth_element(pv_copy.begin(), pv_copy.end() - kk, pv_copy.end());
          double topk_pv_mean = 0.0;
          for (int i = nn - kk; i < nn; ++i) topk_pv_mean += pv_copy[i];
          topk_pv_mean /= static_cast<double>(kk);
          double topk_site_mean = 0.0;
          if (!site_copy.empty()) {
            std::nth_element(site_copy.begin(), site_copy.end() - kk, site_copy.end());
            for (int i = nn - kk; i < nn; ++i) topk_site_mean += site_copy[i];
            topk_site_mean /= static_cast<double>(kk);
          }
          std::cout << "[Debug] Agent0 pv: mean=" << mean_pv
                    << " max=" << max_pv
                    << " top" << kk << "_mean=" << topk_pv_mean
                    << " site_top" << kk << "_mean=" << topk_site_mean
                    << std::endl;
        }
      }
      // Per-agent top-5 list
      const int show_top = 5;
      for (const auto &ag : agents) {
        const auto &pv = ag.state().private_value;
        std::vector<int> idx(nn);
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + std::min(nn, show_top), idx.end(),
                          [&](int a, int b){ return pv[a] > pv[b]; });
        std::cout << "[Debug] Agent" << ag.state().roster_id << " top" << show_top << ":";
        const int m = std::min(nn, show_top);
        for (int j = 0; j < m; ++j) {
          const int i = idx[j];
          const double v = pv[i];
          const double sp = (site_prices_.size() == nn) ? site_prices_[i] : 0.0;
          const int p = (i < pos_idx_.size()) ? pos_idx_[i] : -1;
          std::cout << "  (i=" << i << ",pos=" << p << ",pv=" << v << ",site=" << sp << ")";
        }
        std::cout << std::endl;
      }
    }

    // Needs-aware nomination loop: prioritize positions with remaining league-wide need
    AuctionConfig acfg; acfg.min_increment = 1; acfg.start_price = 1;
    std::vector<int> undrafted(n, 1);
    for (int i = 0; i < n; ++i) {
      if (state.drafted_by[i] >= 0) {
        undrafted[i] = 0;
        price_samples(i, k) = static_cast<double>(state.draft_price[i]);
      }
    }

    auto total_slots_left = [&]() {
      long long total = 0;
      for (const auto &t : state.teams) total += t.need_by_pos.sum();
      return total;
    };

    // Helper: choose next player index to nominate
    auto choose_next_idx = [&]() -> int {
      // Compute league-wide need per position
      Eigen::ArrayXi need_sum = Eigen::ArrayXi::Zero(state.n_positions);
      for (const auto &t : state.teams) {
        if (t.need_by_pos.size() == need_sum.size()) need_sum += t.need_by_pos;
      }
      // Allowed positions are those with positive need; if none except bench, allow any
      std::vector<char> pos_allowed(static_cast<std::size_t>(state.n_positions), 0);
      int allowed_count = 0;
      for (int p = 0; p < need_sum.size(); ++p) {
        if (need_sum[p] > 0) { pos_allowed[static_cast<std::size_t>(p)] = 1; ++allowed_count; }
      }
      bool bench_only = (allowed_count == 0) && (state.bench_pos_idx >= 0) && (need_sum[state.bench_pos_idx] > 0);
      int best_idx = -1;
      double best_score = -1.0;
      for (int i = 0; i < n; ++i) {
        if (!undrafted[i]) continue;
        const int p = (i < pos_idx_.size()) ? pos_idx_[i] : -1;
        if (!bench_only) {
          if (!(p >= 0 && p < static_cast<int>(pos_allowed.size()) && pos_allowed[static_cast<std::size_t>(p)])) continue;
        }
        // Score by site price as a stable anchor
        const double s = (site_prices_.size() == n) ? static_cast<double>(site_prices_[i]) : 1.0;
        if (s > best_score) { best_score = s; best_idx = i; }
      }
      return best_idx;
    };

    while (total_slots_left() > 0) {
      // Stop if no team has budget
      bool any_budget = false;
      for (const auto &t : state.teams) { if (t.budget > 0) { any_budget = true; break; } }
      if (!any_budget) break;

      const int i = choose_next_idx();
      if (i < 0) break; // nothing eligible to nominate

      // Run auction for nominated player i
      AuctionResult ar = run_auction(static_cast<std::size_t>(i), agents, state, acfg);
      const int winner = ar.winner_roster_id;
      const int price = std::max(1, ar.price);
      price_samples(i, k) = static_cast<double>(price);

      if (winner >= 0 && winner < static_cast<int>(state.teams.size())) {
        state.drafted_by[i] = winner;
        state.draft_price[i] = price;
        undrafted[i] = 0;
        TeamState &tw = state.teams[static_cast<std::size_t>(winner)];
        tw.budget -= price;
        const int p = (i < pos_idx_.size()) ? pos_idx_[i] : -1;
        auto dec_slot = [&](int idx) {
          if (idx >= 0 && idx < tw.need_by_pos.size()) {
            tw.need_by_pos[idx] = std::max(0, tw.need_by_pos[idx] - 1);
          }
        };
        bool decremented = false;
        if (p >= 0 && p < tw.need_by_pos.size() && tw.need_by_pos[p] > 0) {
          dec_slot(p); decremented = true;
        }
        const bool flex_ok = (p == 1 || p == 2 || p == 3);
        if (!decremented && flex_ok && state.flex_pos_idx >= 0 && state.flex_pos_idx < tw.need_by_pos.size() && tw.need_by_pos[state.flex_pos_idx] > 0) {
          dec_slot(state.flex_pos_idx); decremented = true;
        }
        if (!decremented && state.bench_pos_idx >= 0 && state.bench_pos_idx < tw.need_by_pos.size() && tw.need_by_pos[state.bench_pos_idx] > 0) {
          dec_slot(state.bench_pos_idx); decremented = true;
        }
        // Keep the winning agent's internal state in sync for subsequent auctions
        for (auto &ag : agents) {
          if (ag.state().roster_id == winner) {
            AgentState &ast = const_cast<AgentState &>(ag.state());
            ast.budget = tw.budget;
            ast.need_by_pos = tw.need_by_pos;
            break;
          }
        }
      } else {
        // Unsold: mark as considered; keep undrafted so it could be re-nominated later if needed
        price_samples(i, k) = static_cast<double>(std::max(1, acfg.start_price));
        // Prevent infinite loops by marking as processed
        undrafted[i] = 0;
      }
    }

    if (k == sims - 1) {
      for (int i = 0; i < n; ++i) {
        summary.final_owner[i] = state.drafted_by[i];
        summary.final_price[i] = state.draft_price[i];
      }
    }
  }

  // Aggregate stats
  for (int i = 0; i < n; ++i) {
    const Eigen::VectorXd row = price_samples.row(i);
    summary.price_mean[i] = row.mean();
    // quantiles
    std::vector<double> v(row.data(), row.data() + row.size());
    std::sort(v.begin(), v.end());
    const auto qidx = [&](double q) {
      int idx = static_cast<int>(std::floor(q * (v.size() - 1)));
      if (idx < 0) idx = 0; if (idx >= static_cast<int>(v.size())) idx = static_cast<int>(v.size()) - 1;
      return idx;
    };
    summary.price_p10[i] = v[qidx(0.10)];
    summary.price_p90[i] = v[qidx(0.90)];
  }

  return summary;
}

} // namespace cpp_core


