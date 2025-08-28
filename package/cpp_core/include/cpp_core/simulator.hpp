#pragma once

#include <cstdint>
#include <vector>

#include <Eigen/Dense>

#include "cpp_core/agent.hpp"
#include "cpp_core/draft_state.hpp"
#include "cpp_core/projection.hpp"

namespace cpp_core {

struct SimConfig {
  int n_sims{100};
  std::uint64_t seed{0};
  bool condition_take{false};
  std::int64_t focal_sid{0};
  int focal_price{0};
  int my_roster_id{-1};
};

struct SimSummary {
  Eigen::ArrayXi final_owner; // per-player index owner id (last sim or mean argmax later)
  Eigen::ArrayXi final_price; // per-player index price
  Eigen::VectorXd price_mean; // mean price over sims (aligned to players)
  Eigen::VectorXd price_p10;
  Eigen::VectorXd price_p90;
};

class Simulator {
public:
  Simulator() = default;

  void set_projection_table(const ProjectionTable *tbl) { proj_tbl_ = tbl; }
  void set_players(const std::vector<std::int64_t> &ids,
                   const Eigen::VectorXi &pos_idx);
  void set_market_curves(const Eigen::VectorXd &site_prices,
                         const Eigen::VectorXd &hist_curve);
  void set_vorp_prices(const Eigen::VectorXd &vorp_prices) { vorp_prices_ = vorp_prices; }

  SimSummary run(const DraftState &initial,
                 const std::vector<AgentConfig> &agent_cfgs,
                 const SimConfig &cfg) const;

private:
  const ProjectionTable *proj_tbl_{nullptr};
  std::vector<std::int64_t> ids_;
  Eigen::VectorXi pos_idx_;
  Eigen::VectorXd site_prices_;
  Eigen::VectorXd hist_curve_;
  Eigen::VectorXd vorp_prices_;
};

} // namespace cpp_core


