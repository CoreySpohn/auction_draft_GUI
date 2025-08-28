#pragma once

#include <cstdint>
#include <vector>

#include <Eigen/Dense>

namespace cpp_core {

struct TeamState {
  int roster_id{0};
  int budget{0};
  Eigen::ArrayXi need_by_pos; // length = n_positions
};

struct DraftState {
  // Players aligned by index i
  std::vector<std::int64_t> sleeper_ids;
  Eigen::ArrayXi drafted_by;  // -1 undrafted, else roster_id
  Eigen::ArrayXi draft_price; // >= 0
  Eigen::VectorXi pos_idx;    // integer-coded positions per player

  std::vector<TeamState> teams; // size = n_teams

  int n_positions{0};
  int flex_pos_idx{-1};
  int bench_pos_idx{-1};
};

} // namespace cpp_core


