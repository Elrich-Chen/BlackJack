# tests/test_env_integration.py

from env.blackjackEnv import BlackjackEnv

def play_split_scenario(deck_sequence):
    """
    1) We seed the deck so that the player gets a pair (e.g. [8,8]) to allow a split,
       then controlled cards for each post-split hand and the dealer.
    2) We manually step through: SPLIT → stand/bust each hand → compare.
    3) We print PayoutTracker info after each hand.
    """
    env = BlackjackEnv()
    # Override the shuffled deck with our forced sequence:
    env.deck = deck_sequence.copy()
    env.set_bet(10)

    # 1) Deal initial hands:
    state = env.reset()  # this will pop 2 cards for player, 2 for dealer
    print("Initial hands:", env.player_hand, "Dealer:", env.dealer_hand)
    print("Tracker start:", env.payout_tracker.get_info())

    # 2) Player splits:
    state, reward, done, info = env.step(3)
    print("\nAfter SPLIT:")
    print(" Hands:", env.player_hands)
    print(" Tracker:", env.payout_tracker.get_info())

    # 3) First split‐hand: let’s *bust* it immediately by hitting until >21
    while env.current_hand_index == 0:
        state, reward, done, info = env.step(0)
    print("\nAfter FIRST hand bust:")
    print(" Hand 0:", env.player_hands[0], "Reward:", reward)
    print(" Tracker:", env.payout_tracker.get_info())

    # 4) Transition to second hand
    print("\nNow playing SECOND hand:")
    print(" Current hand:", env.player_hand)
    print(" Tracker:", env.payout_tracker.get_info())

    # 5) Second hand: we’ll stand immediately (so compare against dealer)
    state, reward, done, info = env.step(1)
    print("\nAfter SECOND hand stand (compare):")
    print(" Dealer hand:", env.dealer_hand)
    print(" Reward:", reward)
    print(" Tracker:", env.payout_tracker.get_info(), "\n")

if __name__ == "__main__":
    # Build a deck sequence:
    #   - Player: [8,8] → split allowed
    #   - Dealer: [10,6]  (will stand on 16)
    #   - After split: 
    #       * Hand 1: draws [K] → busts (8+K=18? Actually not bust—so you might want two hits)
    #         let’s force a bust: draw [K, 10] → 8+K+10=28
    #       * Hand 2: draws [5] → final = 8+5=13, dealer 16 beats 13 → loss
    #
    # Deck pops in this order:
    #   reset(): player [8,8], dealer [10,6] → remaining deck below
    forced_deck = [
        # remaining deck after initial deal:
        # for Hand1: K, 10
        10, 10,  
        # for Hand2: 5
        5,
        # then dealer would hit? but dealer stays on 16 so no more cards
    ]
    # Prepend initial deal cards in reverse pop order:
    # env.reset() pops last→first, so we need to reverse:
    initial_cards = [6, 10, 8, 8]  
    # full deck: bottom-of-deck first:
    full_sequence = forced_deck + initial_cards[::-1]  
    # now run:
    play_split_scenario(full_sequence)