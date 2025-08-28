#pragma once

#include <Eigen/Dense>

namespace cpp_core {

struct MarketState {
  int total_budget{0};
  int n_teams{0};
  Eigen::ArrayXi remaining_slots_by_pos; // length = n_positions
};

// Minimal MVP points->dollars mapping using affine transform and cap at >=1
inline Eigen::VectorXd points_to_dollars(const Eigen::VectorXd &points,
                                         const MarketState & /*ms*/,
                                         double alpha, double beta) {
  Eigen::VectorXd dollars = alpha * points.array() + beta;
  for (int i = 0; i < dollars.size(); ++i) {
    if (dollars[i] < 1.0)
      dollars[i] = 1.0;
  }
  return dollars;
}

} // namespace cpp_core


