basic_strategy = {}

for total in range(5,18):
    for dealer in range(2,12):
        if total >= 17:
            action = 1
        elif total <= 11:
            action = 0
        elif total == 12:
            action = 1 if dealer in [4,5,6] else 0
        elif total in [13, 14, 15, 16]:
            action = 1 if dealer in [2, 3, 4, 5, 6] else 0
        else:
            action = 0  # fallback
        
        basic_strategy[(total, 0, dealer)] = action


# --- Soft Totals Now ---
for total in range(13, 21):  # Soft 13 (A+2) to Soft 20 (A+9)
    for dealer in range(2, 12):
        if total <= 17:
            action = 0  # hit
        elif total == 18:
            action = 1 if dealer in [2, 7, 8] else 0
        else:
            action = 1  # stand
        basic_strategy[(total, 1, dealer)] = action