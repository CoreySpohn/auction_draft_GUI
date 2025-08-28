
# Basics
We draw the projections randomly and then simulate the draft for each set of
projections

The projections are currently set up in a simple manner.

Forecasting should use the agent-based system where each owner is an agent for
each draft and we simulate the rest of the draft. We can sync with the draft
board and attempt to determine how an owner is drafting during the draft.

## Private player evaluations
A core function is the ability to determine how someone is drafting and try
to determine what their private evaluations for player auction prices is.
The two main ones that I can think of are
- VORP based
- Sleeper price based
- Historical league prices (Pos rank vs price)
We already have the VORP system set up to calculate the player prices, we can
pull the Sleeper prices now directly, and I've set up the historical league
prices in the '.cache/herndon_202?' folders. Then we need a way to determine
each agent's private evaluation, which is likely a random draw using the two
price distributions as the base values.

### Agent (owner) behavior
- Based on an agent's private evaluation being higher than consensus they may not bid on a player up for auction because they are saving money for their "guy".
- An agent should act semi-rational. They should understand which positions they still need and not spend up for like their 5th WR when they still don't have a starting running back.
- Agents should be willing to spend up on a player when there are no strong replacements for that player left. This means that if they have every starting roster spot full except for tight end and the player up for auction is a tight end, the player's private evaluation of that player should be higher. Essentially owner's adjust the value of players based on scarcity.

## Nomination order
The Monte Carlo simulations will need to have a semi-realistic nomination order.
I think this is probably something like a random draw of the top players.

# Output
The purpose of this forecasting is so that I can test how the draft plays out in
the scenario where I take the player up for auction at $X vs $X+1, with the
objective of determining at which point my team is on average worse because
I paid too much for the player. This is accomplished by tying each forecasted
draft to a single draw of the player projected points and seeing how good my
team is at the end of each forecasted draft.