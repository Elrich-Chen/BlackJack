import numpy as np
from env.blackjackEnv import BlackjackEnv
from agent.dqn_agent import DQN_Agent
import torch

UNIT = 10
BINS = ["≤-5","-4","-3","-2","-1","0","1","2","3","4","≥5"]

def tc_to_bin(tc):
    if tc <= -5: return "≤-5"
    if tc >=  5: return "≥5"
    return str(int(tc))

CKPT = "checkpoints/iteration3/blackjack_dqn_ep500000.pth"

def run_once(n_hands=200_000, ckpt=CKPT):
    env = BlackjackEnv()
    agent = DQN_Agent(epsilon_start=0.0)
    agent.q_network.load_state_dict(torch.load(ckpt, map_location="cpu"))
    agent.q_network.eval()

    stats = {}
    for _ in range(n_hands):
        pre_tc = env.get_deck_distribution(betting=True)[-3]
        b = tc_to_bin(pre_tc)

        state = env.reset()
        env.set_bet(UNIT)

        done = False
        while not done:
            legal = env.legal_actions()
            action = agent.select_action(state, legal_actions=legal)
            state, _, done, _ = env.step(action)

        pnl = env.payout_tracker.get_info()["net_result"]
        bucket = stats.setdefault(b, {'n':0,'pnl':0.0})
        bucket['n'] += 1
        bucket['pnl'] += pnl

    # convert to ev dict
    evs = {}
    for b in BINS:
        if b in stats:
            evs[b] = stats[b]['pnl'] / stats[b]['n']

    for b in BINS:
        if b in evs:
            print(f"TC {b:>3}: EV/hand=${evs[b]:+.3f}")

    return evs

def main():
    runs = 5
    all_evs = {b: [] for b in BINS}

    for r in range(runs):
        ev_dict = run_once()
        print(f"\n=== Run {r+1} ===")
        for b in BINS:
            if b in ev_dict:
                print(f"TC {b:>3}: EV/hand=${ev_dict[b]:+.3f}")
                all_evs[b].append(ev_dict[b])

    print("\n=== Aggregated over 5 runs ===")
    for b in BINS:
        if all_evs[b]:
            arr = np.array(all_evs[b])
            mean = arr.mean()
            std = arr.std(ddof=1)
            print(f"TC {b:>3}: mean={mean:+.3f}, std={std:.3f}, from {len(arr)} runs")

if __name__ == "__main__":
    main()

