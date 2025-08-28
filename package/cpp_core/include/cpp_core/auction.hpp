#pragma once

#include "cpp_core/agent.hpp"
#include "cpp_core/draft_state.hpp"

namespace cpp_core {

struct AuctionConfig {
  int min_increment{1};
  int start_price{1};
};

struct AuctionResult {
  int winner_roster_id{-1};
  int price{0};
};

// Minimal ascending auction using cap-based bidding
AuctionResult run_auction(std::size_t player_idx, const std::vector<Agent> &agents,
                          DraftState &state, const AuctionConfig &cfg);

} // namespace cpp_core


