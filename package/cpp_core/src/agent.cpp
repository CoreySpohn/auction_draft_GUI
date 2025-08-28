#include "cpp_core/agent.hpp"
#include <algorithm>
#include <iostream>
#include <cmath>

namespace cpp_core {

void Agent::revalue(const Eigen::VectorXd &base_points,
                    const Eigen::VectorXi &pos_idx,
                    const Eigen::VectorXd &site_prices,
                    const Eigen::VectorXd &hist_curve,
                    const Eigen::VectorXd &vorp_prices,
                    std::uint64_t seed) {
  // (void)pos_idx;
  // Map points to dollar scale by linear fit against site_prices when available
  Eigen::VectorXd mapped_points = base_points;
  if (site_prices.size() == base_points.size()) {
    const double mx = base_points.mean();
    const double my = site_prices.mean();
    double cov = 0.0, varx = 0.0;
    for (int i = 0; i < base_points.size(); ++i) {
      const double dx = base_points[i] - mx;
      const double dy = site_prices[i] - my;
      cov += dx * dy;
      varx += dx * dx;
    }
    const double eps = 1e-9;
    const double slope = (varx > eps) ? (cov / varx) : 1.0;
    const double intercept = my - slope * mx;
    mapped_points = slope * base_points.array() + intercept;
  }

  // Weighted blend (ensure weights sum to <= 1.0)
  const double w_site = cfg_.w_site;
  const double w_hist = cfg_.w_hist;
  const double w_vorp = cfg_.w_vorp;
  const double w_pts = std::max(0.0, 1.0 - w_site - w_hist - w_vorp);

  st_.private_value = Eigen::VectorXd::Zero(base_points.size());
  if (mapped_points.size() == st_.private_value.size())
    st_.private_value += w_pts * mapped_points;
  if (site_prices.size() == st_.private_value.size())
    st_.private_value += w_site * site_prices;
  if (hist_curve.size() == st_.private_value.size())
    st_.private_value += w_hist * hist_curve;
  if (vorp_prices.size() == st_.private_value.size())
    st_.private_value += w_vorp * vorp_prices;


  // Snapshot before multiplicative effects
  Eigen::VectorXd before_mult = st_.private_value;

  // Aggressiveness: emphasize stars rather than uniform scaling (toned down)
  // Identify top-K by site (preferred) or by current private values and apply a mild boost
  {
    const int n = static_cast<int>(st_.private_value.size());
    if (n > 0) {
      const int k = std::max(1, std::min(n, 10));
      std::vector<int> idx(n);
      std::iota(idx.begin(), idx.end(), 0);
      auto key = [&](int i) {
        if (site_prices.size() == n) return site_prices[i];
        return st_.private_value[i];
      };
      std::nth_element(idx.begin(), idx.end() - k, idx.end(),
                       [&](int a, int b){ return key(a) < key(b); });
      // Compute star-weighted multipliers: best gets biggest boost
      const double ag = std::max(0.0, cfg_.aggressiveness);
      const double max_boost = 0.12 * ag; // up to +12% at ag=1.0
      // Apply boosts linearly across top-K
      for (int r = 0; r < k; ++r) {
        const int i = idx[n - 1 - r]; // descending
        const double w = (k <= 1) ? 1.0 : (1.0 - static_cast<double>(r) / static_cast<double>(k - 1));
        const double mult = 1.0 + max_boost * w;
        st_.private_value[i] *= mult;
      }
    }
  }

  std::mt19937_64 rng(seed);
  // Small multiplicative lognormal noise (relative mostly constant)
  const double sigma_mult = std::max(0.0, 0.5 * cfg_.noise_sigma);
  std::normal_distribution<double> norm_mult(0.0, sigma_mult);
  // Additive noise scaled to a fraction of budget (not to player price)
  const double std_additive = std::max(0.0, 0.05 * cfg_.noise_sigma * static_cast<double>(st_.budget));
  std::normal_distribution<double> norm_add(0.0, std_additive);
  for (int i = 0; i < st_.private_value.size(); ++i) {
    double v = st_.private_value[i];
    if (sigma_mult > 0.0) {
      const double eps = std::exp(norm_mult(rng));
      v *= eps;
    }
    if (std_additive > 0.0) {
      double delta = norm_add(rng);
      // Cap additive impact to keep cheaper players from having large relative swings
      const double cap = 0.25 * std::max(1.0, v);
      if (delta > cap) delta = cap; else if (delta < -cap) delta = -cap;
      v += delta;
    }
    st_.private_value[i] = std::max(1.0, v);
  }

  // Late-stage scarcity: apply a small uplift only when near end and a starter slot is almost the only one left
  if (pos_idx.size() == st_.private_value.size() && st_.need_by_pos.size() > 0) {
    const int npos = static_cast<int>(st_.need_by_pos.size());
    const int slots_left = st_.need_by_pos.sum();
    double min_mult = 1.0, max_mult = 1.0;
    for (int i = 0; i < pos_idx.size(); ++i) {
      const int p = pos_idx[i];
      if (p >= 0 && p < npos) {
        const int rem_pos = st_.need_by_pos[p];
        double mult = 1.0;
        // Only nudge when close to the end and position nearly filled
        if (slots_left <= 3 && rem_pos <= 1) {
          mult = 1.10; // small, bounded boost
        }
        st_.private_value[i] *= mult;
        if (mult < min_mult) min_mult = mult;
        if (mult > max_mult) max_mult = mult;
      }
    }

    // Debug weights/scarcity once for agent 0
    if (st_.roster_id == 0) {
      static bool printed = false;
      if (!printed) {
        const int n = static_cast<int>(st_.private_value.size());
        const int k = std::max(1, std::min(n, 20));
        auto topk_mean = [&](const Eigen::VectorXd &v) {
          std::vector<double> c(v.data(), v.data() + v.size());
          std::nth_element(c.begin(), c.end() - k, c.end());
          double s = 0.0; for (int i = static_cast<int>(c.size()) - k; i < static_cast<int>(c.size()); ++i) s += c[i];
          return s / static_cast<double>(k);
        };
        double topk_before = topk_mean(before_mult);
        double topk_after = topk_mean(st_.private_value);
        double site_topk = 0.0, hist_topk = 0.0, vorp_topk = 0.0;
        if (site_prices.size() == n) site_topk = topk_mean(site_prices);
        if (hist_curve.size() == n) hist_topk = topk_mean(hist_curve);
        if (vorp_prices.size() == n) vorp_topk = topk_mean(vorp_prices);
        double min_s = min_mult;
        double max_s = max_mult;
        std::cout << "[Debug] Agent0 weights: w_pts=" << w_pts
                  << " w_site=" << w_site
                  << " w_hist=" << w_hist
                  << " w_vorp=" << w_vorp
                  << " aggressiveness=" << cfg_.aggressiveness
                  << " noise_sigma=" << cfg_.noise_sigma
                  << " scarcity[min,max]=" << min_s << "," << max_s
                  << " topK_before=" << topk_before
                  << " topK_after=" << topk_after
                  << " site_topK=" << site_topk
                  << " hist_topK=" << hist_topk
                  << " vorp_topK=" << vorp_topk
                  << std::endl;
        printed = true;
      }
    }
  }
}

double Agent::cap_for_player(std::size_t player_idx,
                             const DraftConstraints &cons) const {
  if (player_idx >= static_cast<std::size_t>(st_.private_value.size()))
    return 0.0;
  const double v = st_.private_value[static_cast<int>(player_idx)];
  const int slots_left = st_.need_by_pos.sum();
  // Budget reserve: protect $1 per open slot; allocate majority to starters
  // Heuristic: spend_ratio for starters rises as starters fill; target ~90% on starters
  int starters_left = 0;
  if (cons.bench_pos_idx >= 0 && cons.bench_pos_idx < st_.need_by_pos.size()) {
    for (int i = 0; i < st_.need_by_pos.size(); ++i) {
      if (i == cons.bench_pos_idx) continue;
      starters_left += st_.need_by_pos[i];
    }
  } else {
    starters_left = slots_left;
  }
  const int bench_left = std::max(0, slots_left - starters_left);
  const double target_starter_spend_ratio = 0.90;
  // Compute a dynamic cap fraction: more permissive for starters, conservative for bench
  double role_multiplier = 1.0;
  if (cons.player_pos_idx == cons.bench_pos_idx && bench_left > 0) {
    role_multiplier = 0.3; // bench value cap at 30% of baseline v
  } else {
    role_multiplier = 1.0; // starters/flex use full baseline v
  }
  const int reserve = std::max(0, slots_left - 1) * cons.min_dollar_per_slot;
  const int max_affordable = std::max(0, st_.budget - reserve);
  const double capped_v = std::min<double>(v * role_multiplier, static_cast<double>(max_affordable));
  return std::max(0.0, capped_v);
}

bool Agent::will_bid(std::size_t player_idx, int next_price,
                     const DraftConstraints &cons) const {
  // Enforce positional capacity: if no remaining slots for player's position (and not FLEX), don't bid
  if (cons.player_pos_idx >= 0 && cons.player_pos_idx < cons.remaining_slots_by_pos.size()) {
    const int rem = cons.remaining_slots_by_pos[cons.player_pos_idx];
    if (rem <= 0) {
      // Allow FLEX pool (RB/WR/TE) if designated and available
      if (cons.flex_pos_idx >= 0 && cons.flex_pos_idx < cons.remaining_slots_by_pos.size()) {
        if (cons.remaining_slots_by_pos[cons.flex_pos_idx] <= 0) {
          return false;
        }
      } else {
        return false;
      }
    }
  }
  return cap_for_player(player_idx, cons) >= static_cast<double>(next_price);
}

int Agent::nominate(const std::vector<int> &undrafted_mask,
                    const Eigen::VectorXi & /*pos_idx*/) const {
  int best = -1;
  double best_val = -1.0;
  for (int i = 0; i < static_cast<int>(undrafted_mask.size()); ++i) {
    if (!undrafted_mask[i])
      continue;
    const double v = (i < st_.private_value.size()) ? st_.private_value[i] : 0.0;
    if (v > best_val) {
      best_val = v;
      best = i;
    }
  }
  return best; // fallback: highest private value
}

} // namespace cpp_core


