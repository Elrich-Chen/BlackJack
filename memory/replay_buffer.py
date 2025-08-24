import numpy as np


class Replay_Buffer:
    """
    Fixed-size ring buffer with TC buckets for O(batch) weighted sampling.
    Buckets by the decision-time true count taken from `state[tc_idx]`.
    """
    def __init__(self, capacity: int, tc_idx: int = -1):
        self.capacity = capacity
        self.tc_idx = tc_idx  # index of true_count in STATE (use -1 if last)
        self.buffer = [None] * capacity            # (state, action, reward, next_state, done)
        self.tc_vals = np.full(capacity, np.nan, dtype=float)
        self.bins = np.full(capacity, -1, dtype=np.int8)   # -1 empty, 0 lo, 1 mid, 2 hi
        self.pos = 0
        self.size = 0

        # buckets as sets for fast add/remove
        self.idx_lo, self.idx_mid, self.idx_hi = set(), set(), set()

        # optional action mix tracker
        self.actions_count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    # ---- helpers ----
    @staticmethod
    def _bin_for_tc(tc: float) -> int:
        if tc >= 3:
            return 2  # hi
        if tc <= -2:
            return 0  # lo
        return 1      # mid

    def _remove_from_bucket(self, i: int):
        b = self.bins[i]
        if b == 0:
            self.idx_lo.discard(i)
        elif b == 1:
            self.idx_mid.discard(i)
        elif b == 2:
            self.idx_hi.discard(i)
        self.bins[i] = -1

    def _add_to_bucket(self, i: int, b: int):
        self.bins[i] = b
        if b == 0:
            self.idx_lo.add(i)
        elif b == 1:
            self.idx_mid.add(i)
        elif b == 2:
            self.idx_hi.add(i)

    # ---- public API ----
    def add(self, state, action, reward, next_state, done):
        """Add new experience, overwriting old slot if buffer is full."""
        if self.size == self.capacity and self.buffer[self.pos] is not None:
            self._remove_from_bucket(self.pos)

        self.buffer[self.pos] = (state, action, reward, next_state, done)
        if action in self.actions_count:
            self.actions_count[action] += 1

        # cache TC from STATE ONLY (decision-time)
        tc = float(state[self.tc_idx])
        self.tc_vals[self.pos] = tc
        self._add_to_bucket(self.pos, self._bin_for_tc(tc))

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        """Uniform sample."""
        if self.size < batch_size:
            raise ValueError("Buffer smaller than batch_size")
        idx = np.random.randint(0, self.size, size=batch_size)
        batch = [self.buffer[i] for i in idx]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.asarray(states),
            np.asarray(actions),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states),
            np.asarray(dones, dtype=np.bool_),
        )

    def sample_weighted(self, batch_size: int, mix: float = 0.7):
        """
        Weighted sample favoring decision-time high TC:
          - take ≈ mix * batch_size from hi bucket (TC>=3), with fallback
          - take the rest uniformly from all filled slots
        O(batch) without scanning.
        """
        if self.size < batch_size:
            raise ValueError("Buffer smaller than batch_size")

        n_hi = int(round(batch_size * mix))
        picks = []

        # sample from high-TC bucket (fallback if small)
        if self.idx_hi and n_hi > 0:
            hi_pool = np.fromiter(self.idx_hi, dtype=int)
            take = min(n_hi, hi_pool.size)
            picks.extend(np.random.choice(hi_pool, size=take, replace=True).tolist())

        need = batch_size - len(picks)
        if need > 0:
            all_idx = np.arange(self.size)
            picks.extend(np.random.choice(all_idx, size=need, replace=True).tolist())

        batch = [self.buffer[i] for i in picks]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.asarray(states),
            np.asarray(actions),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states),
            np.asarray(dones, dtype=np.bool_),
        )

    def __len__(self):
        return self.size

    def composition(self):
        print(f"Actions count in buffer{self.actions_count}")

