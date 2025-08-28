#pragma once

#include <cmath>
#include <cstdint>
#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace cpp_core {

// Simple triangular distribution parameterized by low <= mode <= high.
struct Triangular {
  double low{0.0};
  double mode{0.0};
  double high{0.0};

  Triangular() = default;
  Triangular(double low_, double mode_, double high_)
      : low(low_), mode(mode_), high(high_) {
    if (!(low <= mode && mode <= high)) {
      throw std::invalid_argument("Triangular: require low <= mode <= high");
    }
  }

  double mean() const { return (low + mode + high) / 3.0; }

  double variance() const {
    const double l = low, m = mode, h = high;
    return (l * l + m * m + h * h - l * m - l * h - m * h) / 18.0;
  }

  Eigen::VectorXd
  sample(std::size_t n, std::uint64_t seed) const {
    if (n == 0)
      return {};
    if (!(low <= mode && mode <= high)) {
      throw std::runtime_error("Triangular.sample: invalid parameters");
    }
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    const double range = high - low;
    const double c = (mode - low) / (range > 0.0 ? range : 1.0);
    // Create an Eigen vector with size n
    Eigen::VectorXd vec(n);
    for (std::size_t i = 0; i < n; ++i) {
      const double u = unif(rng);
      if (u < c) {
        vec(i) = low + std::sqrt(u * range * (mode - low));
      } else {
        vec(i) = high - std::sqrt((1.0 - u) * range * (high - mode));
      }
    }
    return vec;
  }
};

// Projection for a player's season fantasy points (single scalar distribution).
struct FantasyPointsProjection {
  Triangular dist{};
  int season{0};
  int week{0}; // 0 for season-long; >0 for weekly if needed later
};

class ProjectionTable {
public:
  ProjectionTable() = default;

  void set(std::int64_t sleeper_id, const FantasyPointsProjection &p) {
    map_[sleeper_id] = p;
  }

  bool has(const std::int64_t &sleeper_id) const {
    return map_.find(sleeper_id) != map_.end();
  }

  FantasyPointsProjection get(const std::int64_t &sleeper_id) const {
    auto it = map_.find(sleeper_id);
    if (it == map_.end()) {
      throw std::out_of_range("ProjectionTable: sleeper_id not found");
    }
    return it->second; // return by value (nanobind-friendly)
  }

  std::size_t size() const { return map_.size(); }

  std::vector<std::int64_t> ids() const {
    std::vector<std::int64_t> out;
    out.reserve(map_.size());
    for (const auto &kv : map_)
      out.push_back(kv.first);
    return out;
  }

  // Sample n values for a single player. Returns Eigen::VectorXd of length n.
  Eigen::VectorXd sample_player(const std::int64_t &sleeper_id,
                                    std::size_t n, std::uint64_t seed) const {
    const FantasyPointsProjection proj = get(sleeper_id);
    const std::uint64_t combined_seed =
        mix_seed(seed, static_cast<std::uint64_t>(sleeper_id));
    return proj.dist.sample(n, combined_seed);
  }

  // Sample for many players at once. Returns a matrix of size
  // ids.size() x n, in row-major order.
  Eigen::MatrixXd
  sample_matrix(const std::vector<std::int64_t> &sleeper_ids, std::size_t n,
                std::uint64_t seed) const {
    Eigen::MatrixXd projections(sleeper_ids.size(), n);
    for (std::int64_t i = 0; i < sleeper_ids.size(); ++i) {
      // get is an already defined method that takes a sleeper_id and returns a
      // FantasyPointsProjection
      const std::int64_t sid = sleeper_ids[i];
      const FantasyPointsProjection proj = get(sid);
      const std::uint64_t s = mix_seed(seed, static_cast<std::uint64_t>(sid));
      projections.row(i) = proj.dist.sample(n, s);
    }
    return projections;
  }

private:
  static std::uint64_t mix_seed(std::uint64_t a, std::uint64_t b) {
    // splitmix64-style mixing to decorrelate seeds
    std::uint64_t z = a + 0x9e3779b97f4a7c15ULL + (b << 6) + (b >> 2);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }

  std::unordered_map<std::int64_t, FantasyPointsProjection> map_;
};

} // namespace cpp_core
