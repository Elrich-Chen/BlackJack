import random

class BlackjackEnv:
    def __init__(self):
        self.deck = []
        self.player_hand = []
        self.dealer_hand = []
        self.done = False  # Track if game is over
        self.reset()       # Start new round
    
    def reset(self):
        """Starts a new round: shuffle deck, deal 2 cards each."""
        self.deck = self.create_deck()
        self.player_hand = [self.deck.pop(), self.deck.pop()]
        self.dealer_hand = [self.deck.pop(), self.deck.pop()]
        self.done = False  # Game just started, not done yet

        return self.get_state()
    
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
    
    
    def create_deck(self):
        deck = ["A",2,3,4,5,6,7,8,9,10,10,10,10] * 4 
        deck = 2*deck
        random.shuffle(deck)
        return deck
    
    def step(self, action):
        if self.done:
            return self.get_state(), 0, True, {}
        if action == 0:
            self.player_hand.append(self.deck.pop())
            player_total = self.calculate_hand_value(self.player_hand)

            if player_total > 21:
                self.done = True
                return self.get_state(), -1, True, {}
            else:
                return self.get_state(), 0, True, {}
        else:
            while self.calculate_hand_value(self.dealer_hand) < 17:
                self.dealer_hand.append(self.deck.pop())
             
            player_total = self.calculate_hand_value(self.player_hand)
            dealer_total = self.calculate_hand_value(self.dealer_hand)

            self.done = True  # Game is over now

            if dealer_total > 21:
                return self.get_state(), +1, True, {}  # Dealer busts → player wins
            elif player_total > dealer_total:
                return self.get_state(), +1, True, {}  # Player wins
            elif player_total < dealer_total:
                return self.get_state(), -1, True, {}  # Dealer wins
            else:
                return self.get_state(), 0, True, {}   # Tie (push)
