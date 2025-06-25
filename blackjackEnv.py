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
    
    
    def is_bust(self, hand):
        return self.calculate_hand_value(hand) > 21

    """Trying to encapsualte all the helper functions needed in step"""
    def player_hits(self):
        self.player_hand.append(self.deck.pop())
        if self.is_bust(self.player_hand):
            self.done = True
            return self.get_state(), -1, True, {"Player has Lost"}
        else:
            return self.get_state(), 0, True, {}
    
    def dealer_hits(self):
        while self.calculate_hand_value(self.dealer_hand) < 17:
            self.dealer_hand.append(self.deck.pop())

    def compare_hands(self):
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

    def step(self, action):
        """ 
        Here our AI robot can now 'play' with the dealer by using the function {step()} 
        If the AI chooses '0' : they are hitting
        If the AI chooses '1': they are standing
            Once they stand, the dealer hits until they reach a minimum score of 17.
        The score is also evaluated in this self.comapre_hands function

        also provides error checking for:
        in case the function is called by our AI model, but game has been completed
        """
        if self.done:
            return self.get_state(), 0, True, {}
        if action == 0:
            #player has chosen to hit
            return self.player_hits()
        else:
            #player has chosen to stand
            self.dealer_hits()
            #return the state and who has won
            return self.compare_hands()
