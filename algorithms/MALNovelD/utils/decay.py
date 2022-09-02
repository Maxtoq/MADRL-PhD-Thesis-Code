import math


class EpsilonDecay:

    def __init__(self, start, finish, n_steps, fn="linear"):
        self.start = start
        self.finish = finish
        self.diff = self.start - self.finish
        self.n_steps = n_steps
        if not fn in ["linear", "exp"]:
            print("ERROR: bad fn for epsilon decay, must be in [linear, exp].")
            exit()
        self.fn = fn

    def get_explo_rate(self, step_i):
        exp_pct_remain = max(0, self.n_steps - step_i) / self.n_steps
        if self.fn == "linear":
            return self.finish + self.diff * exp_pct_remain
        elif self.fn == "exp":
            return self.diff * math.exp(exp_pct_remain - 1) * exp_pct_remain \
                    + self.finish