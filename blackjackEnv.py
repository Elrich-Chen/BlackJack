import random

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

        self.reset()       # Start new round
    
    def reset(self):
        """Starts a new round: shuffle deck, deal 2 cards each."""
        self.deck = self.create_deck()
        self.player_hand = [self.deck.pop(), self.deck.pop()]
        self.dealer_hand = [self.deck.pop(), self.deck.pop()]
        self.done = False  # Game just started, not done yet

        return self.get_state()
    
    def create_deck(self):
        deck = ["A",2,3,4,5,6,7,8,9,10,10,10,10] * 4 
        deck = 2*deck
        random.shuffle(deck)
        return deck
    
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
            score -= 11
            aces -= 1
        
        return score
    
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
            score -= 11
            aces -= 1
        is_soft = 1 if aces > 0 else 0

        dealer_upcard = self.dealer_hand[0]
        dealer_value = 11 if dealer_upcard == "A" else dealer_upcard

        counts = {i :0 for i in range(2,12)}
        for card in self.deck:
            if card == "A":
                counts[11] += 1
            else:
                counts[card] += 1

        totalCardsLeft = len(self.deck)
        percentages = [counts[i]/totalCardsLeft for i in range(2,12)]

        return [score, is_soft, dealer_value] + percentages
    
    def check_bust(self, hand):
        return self.calculate_hand_value(hand) > 21
    def add_card(self):
        self.player_hand.append(self.deck.pop())

    """ACTION 0: PLAYER CHOOSES TO HIT"""
    def handle_player_hit(self):
        self.add_card()
        self.done = self.check_bust(self.player_hand)
        if self.done:
            if (self.split_active):
                return self.handle_split_transition()
            return self.get_state(), -1, True, {"Player has Lost/Busted"}
        else:
            return self.get_state(), 0, False, {"Player has hit, and can still continue !"}
    
    """DEFAULT ACTION: DEALER HITS"""
    def dealer_hits(self):
        while self.calculate_hand_value(self.dealer_hand) < 17:
            self.dealer_hand.append(self.deck.pop())

    """ACTION 1: PLAYER CHOOSES TO STAND"""
    def handle_player_stand(self):
        if (self.split_active):
            return self.handle_split_transition()
        self.dealer_hits()
        return self.compare_hands()
    
    """ACTION 2: PLAYER CHOOSES TO DOUBLE"""
    def handle_player_double(self):
        if len(self.player_hand) != 2:
            self.done = True
            return self.get_state(), -1, True, {"Player does not have two cards"}  # Dealer wins
        else:
            self.add_card()

            if self.split_active:
                self.handle_split_transition()
            
            if self.check_bust(self.player_hand):
                self.done = True
                return self.get_state(), -1, True, {"Player has Lost/Busted"}
            
            self.dealer_hits()
            return self.compare_hands()
            
    """ACTION 3: PLAYER CHOOSES TO SPLIT"""
    def handle_player_split(self):
        if (
            len(self.player_hands) == 8 or 
            len(self.player_hand) != 2 or 
            self.player_hand[0] != self.player_hand[1] or
            self.splits_number == self.ALLOWED_SPLITS
        ):
            self.done=True
            return self.get_state(), -1, True, {}  # Dealer wins
        
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

    def compare_hands(self):
        player_total = self.calculate_hand_value(self.player_hand)
        dealer_total = self.calculate_hand_value(self.dealer_hand)

        self.done = True  # Game is over now

        if set(self.player_hand) == set(self.BLACKJACK) and set(self.dealer_hand) != set(self.BLACKJACK) and len(self.player_hand)==2:
            return self.get_state(), +1.5, True, {"BlackJack Win"}  # Player wins
        elif set(self.player_hand) != set(self.BLACKJACK) and set(self.dealer_hand) == set(self.BLACKJACK) and len(self.dealer_hand)==2:
            return self.get_state(), -1.5, True, {"BlackJack Loss"}  # DEALER wins
        elif set(self.player_hand) == set(self.BLACKJACK) and set(self.dealer_hand) == set(self.BLACKJACK) and len(self.player_hand)==2 and len(self.dealer_hand)==2:
            return self.get_state(), 0, True, {"BlackJack Tie"}  # TIE
        
        if dealer_total > 21:
            return self.get_state(), +1, True, {"Dealer Bust"}  # Dealer busts → player wins
        elif player_total > dealer_total:
            return self.get_state(), +1, True, {"Player won, has a higher score"}  # Player wins
        elif player_total < dealer_total:
            return self.get_state(), -1, True, {"Dealer won, has a higher score"}  # Dealer wins
        else:
            return self.get_state(), 0, True, {"Tie, push game"}   # Tie (push)

    def compare_split_hands(self):
        if (self.split_active and self.current_hand_index > 0):
            self.dealer_hits()
            
            score = 0

            for hand in self.player_hands:
                if self.is_bust(hand):
                    score += -1
                else:
                    self.player_hand = hand
                    _, temp, _ , _ = self.compare_hands()
                    score += temp
            score /= len(self.player_hands)
            self.split_active = False
            self.done = True
            return self.get_state(), score, True, {"Split hand done"}
        else:
            return self.get_state(), -1, False, {"Split hand done"}
        
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
            return self.handle_player_split()
        else:
            return self.get_state(), -1, True, {"Invalid Game Option Chosen !"}



if __name__ == "__main__":
    trial_game = BlackjackEnv()

    while not trial_game.done:
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
    print("----------")


    
            