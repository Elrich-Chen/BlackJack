import random as rd
import time

PLAYER_HAND= []
DEALER_HAND= []


deck = ["A",1,2,3,4,5,6,7,8,9,10,10,10,10] * 4 
rd.shuffle(deck)

def is_bust(PLAYER_HAND):
    score = cal_hand_val(PLAYER_HAND)
    return score > 21

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

def player_turn():
    global PLAYER_HAND
    choice = "h"
    while not is_bust(PLAYER_HAND) and choice == "h":
        print("Your hand : ", PLAYER_HAND)
        choice = input("Enter if you want to hit or stand (h/s) :")
        if choice == "h":
            deal(PLAYER_HAND)
    return


def dealer_turn():
    global DEALER_HAND
    score = cal_hand_val(DEALER_HAND)
    print(f"DEALER HAND: {DEALER_HAND}")
    time.sleep(1.3)
    while score <17:
        deal(DEALER_HAND)
        score = cal_hand_val(DEALER_HAND)
        print(f"DEALER HAND: {DEALER_HAND}")
        time.sleep(1.3)
        


def deal(HAND):
    HAND.append(deck.pop())
    return

def play_game():
    global PLAYER_HAND, DEALER_HAND
    print("Hi. Let us play a basic blackjack game")

    PLAYER_HAND = [deck.pop(), deck.pop()]
    DEALER_HAND = [deck.pop(), deck.pop()]
    DEALER_UP_CARD= DEALER_HAND[0]
    print(f"Player Hand \t {PLAYER_HAND}")
    print(f"Dealer Hand \t {DEALER_UP_CARD}")

    print("\n It is the players turn now")
    player_turn()

    if is_bust(PLAYER_HAND):
        print(f"Player Hand \t {PLAYER_HAND}")
        print("PLAYER HAS LOST, DEALER WINS")
    else:
        print("\n It is the dealers turn now")
        dealer_turn()
        if is_bust(DEALER_HAND): 
            print("PLAYER HAS WON, DEALER LOSES") 
            return
        if cal_hand_val(PLAYER_HAND) > cal_hand_val(DEALER_HAND):
            print("PLAYER HAS WON, WITH A HIGHER SCORE")
        elif cal_hand_val(PLAYER_HAND) < cal_hand_val(DEALER_HAND):
            print("DEALER HAS WON, WITH A HIGHER SCORE")
        else:
            print("ITS A TIE, PUSH OUTCOME")
        return




    return



def main():
    play_game()
    return

if __name__ == "__main__":
    main()