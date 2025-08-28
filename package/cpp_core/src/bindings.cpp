// This comes from package/cpp_core/include/cpp_core/hello.hpp
// and provides the hello() function.
#include "cpp_core/hello.hpp"
#include "cpp_core/player.hpp"
#include "cpp_core/projection.hpp"
#include "cpp_core/agent.hpp"
#include "cpp_core/draft_state.hpp"
#include "cpp_core/simulator.hpp"
#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

// A simple C++ function to be exposed to Python.
int add(int a, int b) { return a + b; }

// The NB_MODULE macro defines the entry point for the Python module.
NB_MODULE(cpp_core, m) {
  m.doc() = "A high-performance C++ simulation core.";
  // Test functions that were created to test the build system.
  m.def("add", &add, "A function that adds two numbers.");
  m.def("hello", &hello, "A function that prints 'Hello, World!'");

  // Player
  nanobind::class_<cpp_core::Player>(m, "Player")
      .def(nanobind::init<>())
      .def(
          nanobind::init<std::string, std::string, std::string, std::int64_t>())
      .def_rw("name", &cpp_core::Player::name)
      .def_rw("position", &cpp_core::Player::position)
      .def_rw("team", &cpp_core::Player::team)
      .def_rw("sleeper_id", &cpp_core::Player::sleeper_id)
      .def("__repr__", [](const cpp_core::Player &p) {
        return fmt::format(
            "Player(name={}, position={}, team={}, sleeper_id={})", p.name,
            p.position, p.team, p.sleeper_id);
      });

  // PlayerTable
  nanobind::class_<cpp_core::PlayerTable>(m, "PlayerTable")
      .def(nanobind::init<>())
      .def("add_player", &cpp_core::PlayerTable::add_player)
      .def("size", &cpp_core::PlayerTable::size)
      .def("has_sleeper_id", &cpp_core::PlayerTable::has_sleeper_id)
      .def("get_by_sleeper_id", &cpp_core::PlayerTable::get_by_sleeper_id)
      .def("players", &cpp_core::PlayerTable::players);

  // Triangular distribution
  nanobind::class_<cpp_core::Triangular>(m, "Triangular")
      .def(nanobind::init<>())
      .def(nanobind::init<double, double, double>())
      .def_rw("low", &cpp_core::Triangular::low)
      .def_rw("mode", &cpp_core::Triangular::mode)
      .def_rw("high", &cpp_core::Triangular::high)
      .def("mean", &cpp_core::Triangular::mean)
      .def("variance", &cpp_core::Triangular::variance)
      .def("sample", &cpp_core::Triangular::sample,
           nanobind::arg("n"), nanobind::arg("seed"))
      .def("__repr__", [](const cpp_core::Triangular &t) {
        return fmt::format("Triangular(low={}, mode={}, high={})", t.low,
                           t.mode, t.high);
      });

  // FantasyPointsProjection
  nanobind::class_<cpp_core::FantasyPointsProjection>(m,
                                                      "FantasyPointsProjection")
      .def(nanobind::init<>())
      .def(nanobind::init<cpp_core::Triangular, int, int>())
      .def_rw("dist", &cpp_core::FantasyPointsProjection::dist)
      .def_rw("season", &cpp_core::FantasyPointsProjection::season)
      .def_rw("week", &cpp_core::FantasyPointsProjection::week)
      .def("__repr__", [](const cpp_core::FantasyPointsProjection &p) {
        return fmt::format("FantasyPointsProjection(dist=Triangular(low={}, "
                           "mode={}, high={}), season={}, week={})",
                           p.dist.low, p.dist.mode, p.dist.high, p.season,
                           p.week);
      });

  // ProjectionTable
  nanobind::class_<cpp_core::ProjectionTable>(m, "ProjectionTable")
      .def(nanobind::init<>())
      .def("set", &cpp_core::ProjectionTable::set)
      .def("has", &cpp_core::ProjectionTable::has)
      .def("get", &cpp_core::ProjectionTable::get)
      .def("size", &cpp_core::ProjectionTable::size)
      .def("ids", &cpp_core::ProjectionTable::ids)
      .def("sample_player", &cpp_core::ProjectionTable::sample_player,
           nanobind::arg("sleeper_id"), nanobind::arg("n"),
           nanobind::arg("seed"))
      // .def("sample_player_ndarray",
      //      &cpp_core::ProjectionTable::sample_player_ndarray,
      //      nanobind::arg("sleeper_id"), nanobind::arg("n"),
      //      nanobind::arg("seed"))
      .def("sample_matrix", &cpp_core::ProjectionTable::sample_matrix,
           nanobind::arg("sleeper_ids"), nanobind::arg("n"),
           nanobind::arg("seed"))
      .def("__repr__", [](const cpp_core::ProjectionTable &t) {
        return fmt::format("ProjectionTable(size={})", t.size());
      });

  // Forecasting: AgentConfig
  nanobind::class_<cpp_core::AgentConfig>(m, "AgentConfig")
      .def(nanobind::init<>())
      .def_rw("alpha_points", &cpp_core::AgentConfig::alpha_points)
      .def_rw("beta_rank", &cpp_core::AgentConfig::beta_rank)
      .def_rw("w_site", &cpp_core::AgentConfig::w_site)
      .def_rw("w_hist", &cpp_core::AgentConfig::w_hist)
      .def_rw("w_vorp", &cpp_core::AgentConfig::w_vorp)
      .def_rw("noise_sigma", &cpp_core::AgentConfig::noise_sigma)
      .def_rw("aggressiveness", &cpp_core::AgentConfig::aggressiveness)
      .def_rw("patience", &cpp_core::AgentConfig::patience)
      .def_rw("nomination_mode", &cpp_core::AgentConfig::nomination_mode);

  // Draft state types
  nanobind::class_<cpp_core::TeamState>(m, "TeamState")
      .def(nanobind::init<>())
      .def_rw("roster_id", &cpp_core::TeamState::roster_id)
      .def_rw("budget", &cpp_core::TeamState::budget)
      .def_rw("need_by_pos", &cpp_core::TeamState::need_by_pos);

  nanobind::class_<cpp_core::DraftState>(m, "DraftState")
      .def(nanobind::init<>())
      .def_rw("sleeper_ids", &cpp_core::DraftState::sleeper_ids)
      .def_rw("drafted_by", &cpp_core::DraftState::drafted_by)
      .def_rw("draft_price", &cpp_core::DraftState::draft_price)
      .def_rw("pos_idx", &cpp_core::DraftState::pos_idx)
      .def_rw("teams", &cpp_core::DraftState::teams)
      .def_rw("n_positions", &cpp_core::DraftState::n_positions)
      .def_rw("flex_pos_idx", &cpp_core::DraftState::flex_pos_idx)
      .def_rw("bench_pos_idx", &cpp_core::DraftState::bench_pos_idx);

  // Simulator config/result
  nanobind::class_<cpp_core::SimConfig>(m, "SimConfig")
      .def(nanobind::init<>())
      .def_rw("n_sims", &cpp_core::SimConfig::n_sims)
      .def_rw("seed", &cpp_core::SimConfig::seed)
      .def_rw("condition_take", &cpp_core::SimConfig::condition_take)
      .def_rw("focal_sid", &cpp_core::SimConfig::focal_sid)
      .def_rw("focal_price", &cpp_core::SimConfig::focal_price)
      .def_rw("my_roster_id", &cpp_core::SimConfig::my_roster_id);

  nanobind::class_<cpp_core::SimSummary>(m, "SimSummary")
      .def(nanobind::init<>())
      .def_rw("final_owner", &cpp_core::SimSummary::final_owner)
      .def_rw("final_price", &cpp_core::SimSummary::final_price)
      .def_rw("price_mean", &cpp_core::SimSummary::price_mean)
      .def_rw("price_p10", &cpp_core::SimSummary::price_p10)
      .def_rw("price_p90", &cpp_core::SimSummary::price_p90);

  nanobind::class_<cpp_core::Simulator>(m, "Simulator")
      .def(nanobind::init<>())
      .def("set_projection_table", &cpp_core::Simulator::set_projection_table)
      .def("set_players", &cpp_core::Simulator::set_players)
      .def("set_market_curves", &cpp_core::Simulator::set_market_curves)
      .def("set_vorp_prices", &cpp_core::Simulator::set_vorp_prices)
      .def("run", &cpp_core::Simulator::run);
}