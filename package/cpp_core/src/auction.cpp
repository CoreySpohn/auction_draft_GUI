#include "cpp_core/auction.hpp"
#include <algorithm>

namespace cpp_core {

AuctionResult run_auction(std::size_t player_idx, const std::vector<Agent> &agents,
                          DraftState & state, const AuctionConfig &cfg) {
  AuctionResult res;
  int price = std::max(1, cfg.start_price);
  int active_count = 0;
  int last_bidder = -1;

  while (true) {
    active_count = 0;
    int next_bidder = -1;
    for (const auto &ag : agents) {
      DraftConstraints cons;
      cons.total_budget = ag.state().budget;
      cons.min_dollar_per_slot = 1;
      cons.remaining_slots_by_pos = ag.state().need_by_pos;
      cons.player_pos_idx = (player_idx < static_cast<std::size_t>(state.pos_idx.size())) ? state.pos_idx[static_cast<int>(player_idx)] : -1;
      cons.flex_pos_idx = state.flex_pos_idx;
      cons.bench_pos_idx = state.bench_pos_idx;
      if (ag.will_bid(player_idx, price, cons)) {
        ++active_count;
        next_bidder = ag.state().roster_id;
      }
    }
    if (active_count <= 1) {
      res.winner_roster_id = (active_count == 1) ? next_bidder : last_bidder;
      res.price = price - cfg.min_increment;
      if (res.price < 1)
        res.price = 1;
      return res;
    }
    last_bidder = next_bidder;
    price += cfg.min_increment;
  }
}

} // namespace cpp_core


