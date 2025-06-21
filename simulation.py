
PLAYER_HAND=0
DEALER_UP_CARD=0
DEALER_HAND=0


deck = ["A",1,2,3,4,5,6,7,8,9,10,10,10,10] * 4 


def is_bust():
    return 

def cal_hand_val(hand):
    value = 0
    aces = 0
    for card in hand:
        if card == "A":
            value += 11
            aces += 1
        else:
            value += card

    while value > 21 and aces:
        value -=10
        aces -=1
    
    return value

def player_turn(player_hand):
    return

def dealer_turn(dealer_hand):
    return


def deal():
    return player_hand, dealer_card

def play_game():

    return



def __main__():
    play_game()
    return