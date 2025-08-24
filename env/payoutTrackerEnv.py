class PayoutTracker:
    def __init__(self, base_bet):
        self.base_bet = base_bet
        self.total_bet = base_bet
        self.bets = [base_bet]      # aligned 1:1 with env.player_hands by index
        self.payout = base_bet      # “money on table” model so net starts at 0

    # ---- NEW: split insert (not append) ----
    def on_split_at(self, hand_index: int):
        """
        Insert a new bet immediately AFTER the given hand index (env inserts the new hand there).
        This keeps self.bets aligned with env.player_hands indices.
        """
        insert_at = hand_index + 1
        self.bets.insert(insert_at, self.base_bet)
        self.total_bet += self.base_bet
        self.payout += self.base_bet  # stake put on table, net unchanged

    def on_double(self, hand_index: int):
        # put additional stake equal to current bet for THAT hand
        self.payout += self.bets[hand_index]
        self.bets[hand_index] *= 2
        self.total_bet += self.base_bet  # same accounting convention as before

    def calculate_payout(self, hand_index: int, reward: float):
        bet = self.bets[hand_index]
        if reward == 1:
            self.payout += bet
        elif reward == 1.5:
            self.payout += 1.5 * bet
        elif reward == 0:
            pass  # push
        else:
            # treat everything else (<0) as -1 * bet
            self.payout -= bet

    def calculate_profit(self):
        return self.payout - self.total_bet

    def get_info(self):
        return {
            "total_bet": self.total_bet,
            "total_payout": self.payout,
            "net_result": self.calculate_profit(),
            "hands_played": len(self.bets),   # helpful for $/hand
        }

    
if __name__ == "__main__":
    print("=== Testing PayoutTracker ===")
    
    # Test 1: Simple win
    print("\nTest 1: Bet $100, Win")
    tracker = PayoutTracker(100)
    tracker.calculate_payout(0, 1)  # Win
    info = tracker.get_info()
    print(f"Total bet: ${info['total_bet']}, Total payout: ${info['total_payout']}, Net result: ${info['net_result']}")
    expected = 100  # Should profit $100
    print(f"Expected: ${expected}, Got: ${info['net_result']}, {'✅ PASS' if info['net_result'] == expected else '❌ FAIL'}")
    
    # Test 2: Simple loss
    print("\nTest 2: Bet $100, Lose")
    tracker = PayoutTracker(100)
    tracker.calculate_payout(0, -1)  # Loss
    info = tracker.get_info()
    print(f"Total bet: ${info['total_bet']}, Total payout: ${info['total_payout']}, Net result: ${info['net_result']}")
    expected = -100  # Should lose $100
    print(f"Expected: ${expected}, Got: ${info['net_result']}, {'✅ PASS' if info['net_result'] == expected else '❌ FAIL'}")
    
    # Test 3: Push
    print("\nTest 3: Bet $100, Push")
    tracker = PayoutTracker(100)
    tracker.calculate_payout(0, 0)  # Push
    info = tracker.get_info()
    print(f"Total bet: ${info['total_bet']}, Total payout: ${info['total_payout']}, Net result: ${info['net_result']}")
    expected = 0  # Should break even
    print(f"Expected: ${expected}, Got: ${info['net_result']}, {'✅ PASS' if info['net_result'] == expected else '❌ FAIL'}")
    
    # Test 4: Blackjack
    print("\nTest 4: Bet $100, Blackjack")
    tracker = PayoutTracker(100)
    tracker.calculate_payout(0, 1.5)  # Blackjack
    info = tracker.get_info()
    print(f"Total bet: ${info['total_bet']}, Total payout: ${info['total_payout']}, Net result: ${info['net_result']}")
    expected = 150  # Should profit $150
    print(f"Expected: ${expected}, Got: ${info['net_result']}, {'✅ PASS' if info['net_result'] == expected else '❌ FAIL'}")
    
    # Test 5: Double down win
    print("\nTest 5: Bet $100, Double Down, Win")
    tracker = PayoutTracker(100)
    tracker.on_double(0)  # Double down
    tracker.calculate_payout(0, 1)  # Win
    info = tracker.get_info()
    print(f"Total bet: ${info['total_bet']}, Total payout: ${info['total_payout']}, Net result: ${info['net_result']}")
    expected = 200  # Should profit $200 (doubled bet)
    print(f"Expected: ${expected}, Got: ${info['net_result']}, {'✅ PASS' if info['net_result'] == expected else '❌ FAIL'}")