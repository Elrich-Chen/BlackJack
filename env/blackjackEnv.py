import random
from .payoutTrackerEnv import PayoutTracker as PT

class BlackjackEnv:
    BLACKJACK = [10, "A"]
    ALLOWED_SPLITS = 4
    # How to feed reward to the AGENT on split rounds:
    # "avg" (your current), "sum", "sum_clip", "sum_sqrt", "sum_tanh"
    SPLIT_REWARD_MODE = "sum_clip"
    SPLIT_CLIP = 2.0     # for "sum_clip": clamp [-2, +2]
    SPLIT_TAU  = 2.0     # for "sum_tanh": tanh(sum/τ)*τ

    def __init__(self):
        self.deck = []
        self.player_hand = []
        self.dealer_hand = []
        self.done = False  # Track if game is over
        self.split_active = False
        self.player_hands = []
        self.current_hand_index = 0
        self.splits_number = 0
        self.penetration = 0.25
        self.num_deck = 6

        self.reset()       # Start new round
    
    def reset(self):
        """Starts a new round: shuffle deck, deal 2 cards each."""
        self.deck = self.create_deck()
        self.player_hand = [self.deck.pop(), self.deck.pop()]
        self.dealer_hand = [self.deck.pop(), self.deck.pop()]
        self.player_hands = [self.player_hand]   # <-- keep list in sync from the start
        self.done = False  # Game just started, not done yet
        self.splits_number = 0
        self.current_hand_index = 0
        self.split_active = False

        return self.get_state()
    
    def set_bet(self, bet):
        self.base_bet = bet
        self.payout_tracker = PT(bet)
    
    def create_deck(self):
        if len(self.deck) <= (self.penetration* 52 * self.num_deck):
            deck = ["A",2,3,4,5,6,7,8,9,10,10,10,10] * 4 
            deck = self.num_deck*deck
            random.shuffle(deck)
            return deck
        else:
            return self.deck
    
    def calculate_hand_value(self, hand):
        score = 0
        aces = 0
        for card in hand:
            if card == "A":
                score += 11
                aces += 1
            else:
                score += card
        
        while score > 21 and aces:
            score -= 10
            aces -= 1
        
        return score
    
    def get_deck_distribution(self, betting=False):
        total_cards = len(self.deck)
        counts = {i: 0 for i in range(2, 12)}  # 2-10, 11 for Ace

        for card in self.deck:
            val = 11 if card == 'A' else card
            counts[val] += 1

        percentages = [counts[i] / total_cards for i in range(2, 12)]

        # Running count: Hi-Lo system
        running_count = -(
            sum(counts[i] for i in [2, 3, 4, 5, 6])  # Low cards = +1
            - sum(counts[i] for i in [10, 11])       # High cards = -1
        )

        decks_remaining = total_cards / 52
        true_count = running_count / decks_remaining if decks_remaining > 0 else 0

        return percentages + betting*[running_count, true_count, decks_remaining, total_cards]

    def get_state(self):
        """This is usually called after .reset() or a game finishing
        therefore the cards have been dealt already 

        We aim to return the player score, soft status, and dealer value
        Along with the percentages of the card left in the deck"""
        score = 0
        aces = 0
        for card in self.player_hand:
            if card == "A":
                score += 11
                aces += 1
            else:
                score += card
        
        while score > 21 and aces:
            score -= 10
            aces -= 1
        is_soft = 1 if aces > 0 else 0

        dealer_upcard = self.dealer_hand[0]
        dealer_value = 11 if dealer_upcard == "A" else dealer_upcard

        percentages = self.get_deck_distribution()
        tc = self.get_deck_distribution(betting=True)[-3]  # True Count
        tc = (max(-8, min(8, tc)) / 8.0)

        num_cards = len(self.player_hand)
        can_split = 1 if len(self.player_hand)==2 and self.player_hand[1] == self.player_hand[0] else 0
        can_double = 1 if len(self.player_hand)==2 else 0

        return [num_cards, score, is_soft, dealer_value, can_split, can_double] + percentages + [tc]
    
    def check_bust(self, hand):
        return self.calculate_hand_value(hand) > 21
    def add_card(self):
        self.player_hand.append(self.deck.pop())

    """
    ACTION 0: PLAYER CHOOSES TO HIT
    """
    def handle_player_hit(self):
        self.add_card()
        self.done = self.check_bust(self.player_hand)
        if self.done:
            if (self.split_active):
                return self.handle_split_transition()
            self.payout_tracker.calculate_payout(self.current_hand_index, -1)
            return self.get_state(), -1.0, True, {"Player has Lost/Busted"}
        else:
            return self.get_state(), 0, False, {"Player has hit, and can still continue !"}
    
    """
    DEFAULT ACTION: DEALER HITS
    """
    def dealer_hits(self):
        while self.calculate_hand_value(self.dealer_hand) < 17:
            self.dealer_hand.append(self.deck.pop())

    """
    ACTION 1: PLAYER CHOOSES TO STAND
    """
    def handle_player_stand(self):
        if (self.split_active):
            return self.handle_split_transition()
        self.dealer_hits()
        return self.compare_hands()
    
    """
    ACTION 2: PLAYER CHOOSES TO DOUBLE
    """
    def handle_player_double(self):
        if len(self.player_hand) != 2:
            self.done = True
            self.payout_tracker.calculate_payout(self.current_hand_index, -1)
            return self.get_state(), -1, True, {"Player does not have two cards"}  # Dealer wins
        else:
            self.add_card()
            self.payout_tracker.on_double(self.current_hand_index)

            if self.split_active:
                return self.handle_split_transition()
            
            if self.check_bust(self.player_hand):
                self.done = True
                self.payout_tracker.calculate_payout(self.current_hand_index, -1)
                return self.get_state(), -1, True, {"Player has Lost/Busted"}
            
            self.dealer_hits()
            return self.compare_hands()
            
    """
    ACTION 3: PLAYER CHOOSES TO SPLIT
    """
    def handle_player_split(self):
        if (
            len(self.player_hands) == 8 or 
            len(self.player_hand) != 2 or 
            self.player_hand[0] != self.player_hand[1] or
            self.splits_number == self.ALLOWED_SPLITS
        ):
            self.done=True
            return self.get_state(), -1, True, {"Illegal move done"}  # Dealer wins
        
        self.payout_tracker.on_split_at(self.current_hand_index)
        self.split_active = True
        self.splits_number += 1
        new_hand_1 = [self.player_hand[0], self.deck.pop()]
        new_hand_2 = [self.player_hand[1], self.deck.pop()]
        
        try:
            self.player_hands.pop(self.current_hand_index)
        except IndexError:
            pass
        finally:
            self.player_hands.insert(self.current_hand_index, new_hand_1)
            self.player_hands.insert(self.current_hand_index+1, new_hand_2)

        self.player_hand = new_hand_1
        return self.get_state(), 0, False, {"Split Started"}
        
    def advance_to_next_split_hand(self):
        self.done = False
        self.current_hand_index += 1
        self.player_hand = self.player_hands[self.current_hand_index]
        return self.get_state(), 0, False, {"Now playing the next hand"}

    def handle_split_transition(self):
        if (self.current_hand_index <len(self.player_hands)-1):
            return self.advance_to_next_split_hand()
        return self.compare_split_hands()

    # Drop-in helper (use in BOTH training and eval loops)
    def legal_actions(self):
        # hit(0) and stand(1) are always legal while not done
        legal = [0, 1]

        hand = self.player_hands[self.current_hand_index]  # use current hand

        # Double: only if exactly 2 cards in the current hand
        if len(hand) == 2:
            legal.append(2)

        # Split: only if exactly 2 cards, both same rank, and split limit not exceeded
        if len(hand) == 2 and hand[0] == hand[1] and self.splits_number < self.ALLOWED_SPLITS:
            legal.append(3)

        return legal


    def compare_hands(self, hand_index=0):
        player_total = self.calculate_hand_value(self.player_hand)
        dealer_total = self.calculate_hand_value(self.dealer_hand)
        self.done = True

        is_two = (len(self.player_hand) == 2)
        is_natural = is_two and set(self.player_hand) == set(self.BLACKJACK)
        dealer_natural = (len(self.dealer_hand) == 2 and set(self.dealer_hand) == set(self.BLACKJACK))

        if is_natural:
            if dealer_natural:
                self.payout_tracker.calculate_payout(hand_index, 0.0); r = 0.0
            else:
                self.payout_tracker.calculate_payout(hand_index, 1.5); r = 1.5
            return self.get_state(), r, True, {"BlackJack settle"}

        if dealer_total > 21:
            self.payout_tracker.calculate_payout(hand_index, 1);  return self.get_state(), +1, True, {"Dealer bust"}
        if player_total > dealer_total:
            self.payout_tracker.calculate_payout(hand_index, 1);  return self.get_state(), +1, True, {"Player higher"}
        if player_total < dealer_total:
            self.payout_tracker.calculate_payout(hand_index, -1); return self.get_state(), -1, True, {"Dealer higher"}
        self.payout_tracker.calculate_payout(hand_index, 0);      return self.get_state(), 0, True, {"Push"}


    def compare_split_hands(self):
        if self.split_active and self.current_hand_index > 0:
            score_sum = 0.0
            all_bust = True

            # First pass: settle busts immediately (as your code already does)
            for index, hand in enumerate(self.player_hands):
                if self.check_bust(hand):
                    score_sum += -1.0
                    self.payout_tracker.calculate_payout(index, -1.0)
                else:
                    all_bust = False

            # If at least one hand survives, dealer plays ONCE
            if not all_bust:
                self.dealer_hits()
                dealer_total = self.calculate_hand_value(self.dealer_hand)

                for index, hand in enumerate(self.player_hands):
                    if self.check_bust(hand):
                        continue  # already paid above

                    player_total = self.calculate_hand_value(hand)
                    is_two = (len(hand) == 2)
                    is_nat = is_two and set(hand) == set(self.BLACKJACK)
                    dealer_nat = (len(self.dealer_hand) == 2 and set(self.dealer_hand) == set(self.BLACKJACK))

                    if is_nat:
                        # After split, blackjack = 21 pays 1:1 (push if dealer natural)
                        r = 0.0 if dealer_nat else 1.0
                    elif dealer_total > 21:
                        r = 1.0
                    elif player_total > dealer_total:
                        r = 1.0
                    elif player_total < dealer_total:
                        r = -1.0
                    else:
                        r = 0.0

                    self.payout_tracker.calculate_payout(index, r)
                    score_sum += r

            self.split_active = False
            self.done = True

            # === Agent-facing reward (variance-controlled) ===
            k = max(1, len(self.player_hands))  # number of hands this round
            mode = getattr(self, "SPLIT_REWARD_MODE", "avg")

            if mode == "avg":
                training_reward = score_sum / k
            elif mode == "sum":
                training_reward = score_sum
            elif mode == "sum_clip":
                lo, hi = -float(self.SPLIT_CLIP), float(self.SPLIT_CLIP)
                training_reward = max(lo, min(hi, score_sum))
            elif mode == "sum_sqrt":
                # preserves sign; scales by sqrt(k) to reduce spikes
                import math
                training_reward = score_sum / math.sqrt(k)
            elif mode == "sum_tanh":
                # smooth soft-clip: tanh(sum/τ) * τ
                import math
                τ = float(self.SPLIT_TAU)
                training_reward = math.tanh(score_sum / τ) * τ
            else:
                training_reward = score_sum  # default fallback

            return self.get_state(), training_reward, True, {
                "Split hand done": True,
                "true_sum": score_sum,
                "hands_played": k
            }

        # Fallback (shouldn’t happen)
        self.done = True
        return self.get_state(), 0.0, True, {"Split hand done": True}

        
    def step(self, action):
        if self.done:
            return self.get_state(), 0, True, {"Game is finished !"}

        if action == 0:
            return self.handle_player_hit()
        elif action == 1:
            return self.handle_player_stand()
        elif action == 2:
            return self.handle_player_double()
        elif action == 3:
            assert len(self.player_hand) == 2 and self.player_hand[0] == self.player_hand[1] and self.splits_number < self.ALLOWED_SPLITS, \
            f"Illegal split attempted: hand={self.player_hand}, splits={self.splits_number}"
            return self.handle_player_split()
        else:
            return self.get_state(), -1, True, {"Invalid Game Option Chosen !"}


if __name__ == "__main__":
    trial_game = BlackjackEnv()
    trial_game.set_bet(10)

    while trial_game.player_hand[0] != trial_game.player_hand[1]:
        trial_game.reset()

    while not trial_game.done:
        print(trial_game.deck)
        print(f"Player Hand : {trial_game.player_hand}")
        print(f"Dealer Upcard : {trial_game.dealer_hand[0]}")
        print("\n")

        # Take action in the environment
        state, reward, done, msg = trial_game.step(int(input("Enter 0, 1, 2, 3 for Hit, Stand, Double and Split : ")))
        print(msg)

        # Show the updated hand
        print(f"Player hand: {trial_game.player_hand}")
        print(f"Dealer hdgchhand: {trial_game.dealer_hand[0]}")
        print("----------")
    
    print(f"Reward was {reward}")
    print(f"Player hand: {trial_game.player_hands}")
    print(f"Dealer hand: {trial_game.dealer_hand}")
    print(trial_game.payout_tracker.bets)
    print("----------")
    print(trial_game.payout_tracker.get_info())
    
            