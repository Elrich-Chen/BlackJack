class PayoutTracker:
    def __init__(self, base_bet):
        self.base_bet = base_bet
        self.total_bet = base_bet
        self.bets = [base_bet]
        self.payout = base_bet

    def on_split(self):
        self.bets.append(self.base_bet)
        self.total_bet += self.base_bet
        self.payout += self.base_bet
    
    def on_double(self, hand_index):
        self.payout += self.bets[hand_index]
        self.bets[hand_index] *= 2
        self.total_bet += self.base_bet
    
    def calculate_payout(self, hand_index, reward):
        bet = self.bets[hand_index]
        if reward == 1:
            self.payout += bet
        elif reward == 1.5:
            self.payout += 1.5*bet
        elif reward == 0:
            pass
        else:
            self.payout -= bet

    def calculate_profit(self):
        return (self.payout - self.total_bet)
    
    def get_info(self):
        return {
            "total_bet": self.total_bet,
            "total_payout": self.payout,
            "net_result": self.calculate_profit()
        }