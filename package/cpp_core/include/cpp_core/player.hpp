#pragma once

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cpp_core {

struct Player {
  // Data members
  std::string name;
  std::string position;
  std::string team;
  std::int64_t sleeper_id;

  // Default constructor
  Player() = default;
  // Constructor with parameters
  Player(std::string name_, std::string position_, std::string team_,
         std::int64_t sleeper_id_)
      : name(std::move(name_)), position(std::move(position_)),
        team(std::move(team_)), sleeper_id(std::move(sleeper_id_)) {}
};

class PlayerTable {
public:
  PlayerTable() = default;

  void add_player(const Player &p) {
    const std::size_t idx = players_.size();
    players_.push_back(p);
    if (p.sleeper_id != 0) {
      sleeper_index_[p.sleeper_id] = idx;
    }
  }

  std::size_t size() const { return players_.size(); }

  bool has_sleeper_id(const std::int64_t &sid) const {
    return sleeper_index_.find(sid) != sleeper_index_.end();
  }

  Player get_by_sleeper_id(const std::int64_t &sid) const {
    auto it = sleeper_index_.find(sid);
    if (it == sleeper_index_.end()) {
      throw std::out_of_range("Sleeper id not found");
    }
    return players_.at(it->second);
  }

  std::vector<Player> players() const { return players_; }

private:
  std::vector<Player> players_;
  std::unordered_map<std::int64_t, std::size_t> sleeper_index_;
};

} // namespace cpp_core
