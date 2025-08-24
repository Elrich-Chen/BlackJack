def shaping_bonus(state, action, true_count):
    num_cards, score, is_soft, dealer, can_split, can_double = map(int, state[:6])
    bonus = 0.0
    # Never split 10s
    if can_split and action == 3 and score == 20: bonus -= 0.4
    # Split A,A
    if can_split and score == 12 and is_soft == 1 and action == 3: bonus += 0.2
    # Don't hit hard 17+
    if is_soft == 0 and score >= 17 and action == 0: bonus -= 0.4
    # High-TC: double 10/11 vs 2–9
    if true_count >= 3 and can_double and num_cards == 2 and score in (10,11) and 2 <= dealer <= 9:
        bonus += 0.2 if action == 2 else -0.1
    # Low/neutral TC: discourage random doubles
    if true_count <= 0 and action == 2: bonus -= 0.05
    return max(-0.5, min(0.5, bonus))
