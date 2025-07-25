import random
from .payoutTrackerEnv import PayoutTracker as PT

class BlackjackEnv:
    BLACKJACK = [10, "A"]
    ALLOWED_SPLITS = 4
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
        self.player_hands = []
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
    
    def get_deck_distribution(self):
        counts = {i :0 for i in range(2,12)}
        for card in self.deck:
            if card == "A":
                counts[11] += 1
            else:
                counts[card] += 1

        totalCardsLeft = len(self.deck)
        percentages = [counts[i]/totalCardsLeft for i in range(2,12)]

        return percentages

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

        num_cards = len(self.player_hand)
        can_split = 1 if len(self.player_hand)==2 and self.player_hand[1] == self.player_hand[0] else 0
        can_double = 1 if len(self.player_hand)==2 else 0

        return [num_cards, score, is_soft, dealer_value, can_split, can_double] + percentages
    
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
            self.payout_tracker.calculate_payout(self.current_hand_index, -1)
            if (self.split_active):
                return self.handle_split_transition()
            return self.get_state(), -1, True, {"Player has Lost/Busted"}
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

        self.player_hand = self.player_hands[0]
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

    def compare_hands(self, hand_index = 0):
        player_total = self.calculate_hand_value(self.player_hand)
        dealer_total = self.calculate_hand_value(self.dealer_hand)

        self.done = True  # Game is over now

        if set(self.player_hand) == set(self.BLACKJACK) and set(self.dealer_hand) != set(self.BLACKJACK) and len(self.player_hand)==2:
            self.payout_tracker.calculate_payout(hand_index, 1.5)
            return self.get_state(), +1.5, True, {"BlackJack Win"}  # Player wins
        elif set(self.player_hand) != set(self.BLACKJACK) and set(self.dealer_hand) == set(self.BLACKJACK) and len(self.dealer_hand)==2:
            self.payout_tracker.calculate_payout(hand_index, -1)
            return self.get_state(), -1.5, True, {"BlackJack Loss"}  # DEALER wins
        elif set(self.player_hand) == set(self.BLACKJACK) and set(self.dealer_hand) == set(self.BLACKJACK) and len(self.player_hand)==2 and len(self.dealer_hand)==2:
            self.payout_tracker.calculate_payout(hand_index, 0)
            return self.get_state(), 0, True, {"BlackJack Tie"}  # TIE
        
        if dealer_total > 21:
            self.payout_tracker.calculate_payout(hand_index, 1)
            return self.get_state(), +1, True, {"Dealer Bust"}  # Dealer busts → player wins
        elif player_total > dealer_total:
            self.payout_tracker.calculate_payout(hand_index, 1)
            return self.get_state(), +1, True, {"Player won, has a higher score"}  # Player wins
        elif player_total < dealer_total:
            self.payout_tracker.calculate_payout(hand_index, -1)
            return self.get_state(), -1, True, {"Dealer won, has a higher score"}  # Dealer wins
        else:
            self.payout_tracker.calculate_payout(hand_index, 0)
            return self.get_state(), 0, True, {"Tie, push game"}   # Tie (push)

    def compare_split_hands(self):
        if (self.split_active and self.current_hand_index > 0):
            
            score = 0

            for index, hand in enumerate(self.player_hands):
                if self.check_bust(hand):
                    score += -1
                    self.payout_tracker.calculate_payout(index, -1)
                else:
                    self.dealer_hits()
                    self.player_hand = hand
                    _, temp, _ , _ = self.compare_hands(hand_index=index)
                    score += temp
            score /= len(self.player_hands)
            self.split_active = False
            self.done = True
            return self.get_state(), score, True, {"Split hand done"}
        else:
            return self.get_state(), -1, True, {"Split hand done"}
        
    def calculate_payout(self, reward):
        if reward > 0:
            self.total_payout += self.total_bet
        elif reward == -1:
            self.total_payout = 0
        else:
            pass

    def step(self, action):
        if self.done:
            return self.get_state(), 0, True, {"Game is finished !"}

        if action == 0:
            return self.handle_player_hit()
        elif action == 1:
            return self.handle_player_stand()
        elif action == 2:
            self.payout_tracker.on_double(self.current_hand_index)
            return self.handle_player_double()
        elif action == 3:
            self.payout_tracker.on_split()
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
    
            