#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Dense>

namespace cpp_core {

struct AgentConfig {
  // Mixture weights and behavior parameters (simple MVP set)
  double alpha_points{1.0};
  double beta_rank{0.0};
  double w_site{0.5};
  double w_hist{0.5};
  double w_vorp{0.0};
  double noise_sigma{0.1};
  double aggressiveness{1.0};
  double patience{0.5};
  int nomination_mode{1}; // 0=random,1:drain,2:target
};

struct DraftConstraints {
  int total_budget{0};
  int min_dollar_per_slot{1};
  Eigen::ArrayXi remaining_slots_by_pos; // length = n_positions
  int player_pos_idx{-1};
  int flex_pos_idx{-1};
  int bench_pos_idx{-1};
};

struct AgentState {
  int roster_id{-1};
  int budget{0};
  Eigen::ArrayXi need_by_pos; // length = n_positions
  // Private values vector aligned to player index order
  Eigen::VectorXd private_value; // in dollars
};

class Agent {
public:
  Agent() = default;
  Agent(AgentConfig cfg, AgentState st) : cfg_(cfg), st_(std::move(st)) {}

  void revalue(const Eigen::VectorXd &base_points,
               const Eigen::VectorXi &pos_idx,
               const Eigen::VectorXd &site_prices,
               const Eigen::VectorXd &hist_curve,
               const Eigen::VectorXd &vorp_prices,
               std::uint64_t seed);

  // Maximum willingness to pay under constraints
  double cap_for_player(std::size_t player_idx,
                        const DraftConstraints &cons) const;

  // Simple decision: bid if next price ≤ cap
  bool will_bid(std::size_t player_idx, int next_price,
                const DraftConstraints &cons) const;

  // Placeholder nomination: pick highest value undrafted among drain/target rules
  int nominate(const std::vector<int> &undrafted_mask,
               const Eigen::VectorXi &pos_idx) const;

  const AgentState &state() const { return st_; }
  AgentState &mutable_state() { return st_; }

private:
  AgentConfig cfg_{};
  AgentState st_{};
};

} // namespace cpp_core


