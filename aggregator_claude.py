"""
Carbon Farming Contract Design - Aggregator (Minimal Workshop Version)
"""

import numpy as np
from gymnasium import spaces
from farmer_claude import CROPS, NUM_CROPS

# Contract bounds
MAX_ACRE_PAY = 50.0
MAX_TON_PAY = 60.0
PARAMS_PER_CONTRACT = 4   # acre_pay, ton_pay, upfront_frac, mrv_share
NUM_CONTRACTS = 3
ACTION_DIM = NUM_CONTRACTS * PARAMS_PER_CONTRACT  # 12

# MRV cost per farm-unit (100 acres)
MRV_COSTS = {"action": 5.0, "result": 25.0, "hybrid": 15.0}

# Carbon market
CARBON_PRICE = 35.0

# Fixed lambdas (Option B)
LAMBDAS = {"action": 1.0, "result": 0.0, "hybrid": 0.5}
CONTRACT_NAMES = ["action", "result", "hybrid"]


class Aggregator:
    def __init__(self, budget, carbon_weight=0.5):
        self.budget = budget
        self.carbon_weight = carbon_weight

        self.prev_accepted = {}   # fid -> contract idx or -1
        self.prev_carbon = {}     # fid -> float

        self.spent = 0.0
        self.mrv_cost = 0.0
        self.carbon = 0.0
        self.revenue = 0.0
        self.menu = None

    @staticmethod
    def action_space():
        return spaces.Box(low=0.0,
                          high=np.array([MAX_ACRE_PAY, MAX_TON_PAY, 1.0, 1.0] * 3, dtype=np.float32),
                          dtype=np.float32)

    @staticmethod
    def obs_dim(num_farmers=5):
        # per farmer: size(1) + soil_signal(1) + inferred_crop_pref(4) = 6
        # global: budget(1) + prev_accepted(N) + prev_carbon(N) = 1+2N
        return num_farmers * 6 + 1 + 2 * num_farmers

    @staticmethod
    def observation_space(num_farmers=5):
        dim = Aggregator.obs_dim(num_farmers)
        return spaces.Box(low=-1.0, high=1e6, shape=(dim,), dtype=np.float32)

    def get_obs(self, farmers):
        obs = []
        for f in farmers:
            obs.append(f.farm_size / 10.0)
            obs.append(f.noisy_soil())
            obs += f.inferred_crop_pref()
        obs.append((self.budget - self.spent - self.mrv_cost) / self.budget)
        for f in farmers:
            obs.append(self.prev_accepted.get(f.fid, -1.0))
        for f in farmers:
            obs.append(self.prev_carbon.get(f.fid, 0.0))
        return np.array(obs, dtype=np.float32)

    def decode_action(self, action):
        action = np.clip(action, 0.0, None)
        menu = {}
        for i, name in enumerate(CONTRACT_NAMES):
            o = i * PARAMS_PER_CONTRACT
            menu[name] = {
                "acre_pay": float(np.clip(action[o], 0, MAX_ACRE_PAY)),
                "ton_pay": float(np.clip(action[o+1], 0, MAX_TON_PAY)),
                "upfront_frac": float(np.clip(action[o+2], 0, 1)),
                "mrv_share": float(np.clip(action[o+3], 0, 1)),
                "lambda": LAMBDAS[name],
            }
        self.menu = menu
        return menu

    def get_mrv_cost(self, contract_type, farm_size):
        return MRV_COSTS[contract_type] * farm_size

    def process_outcome(self, farmer, ct_name, payment, carbon):
        params = self.menu[ct_name]
        mrv = self.get_mrv_cost(ct_name, farmer.farm_size)
        agg_mrv = (1 - params["mrv_share"]) * mrv

        self.spent += payment
        self.mrv_cost += agg_mrv
        self.carbon += carbon
        self.revenue += carbon * CARBON_PRICE

        self.prev_accepted[farmer.fid] = CONTRACT_NAMES.index(ct_name)
        self.prev_carbon[farmer.fid] = carbon

    def process_rejection(self, farmer):
        self.prev_accepted[farmer.fid] = -1.0
        self.prev_carbon[farmer.fid] = 0.0

    def reward(self):
        profit = self.revenue - self.spent - self.mrv_cost
        carbon_val = self.carbon * CARBON_PRICE
        r = (1 - self.carbon_weight) * profit + self.carbon_weight * carbon_val
        # Budget penalty
        overspend = (self.spent + self.mrv_cost) - self.budget
        if overspend > 0:
            r -= overspend * 2.0
        return r

    def reset(self):
        self.spent = self.mrv_cost = self.carbon = self.revenue = 0.0
        self.menu = None

    def hard_reset(self):
        self.reset()
        self.prev_accepted = {}
        self.prev_carbon = {}

    def summary(self):
        profit = self.revenue - self.spent - self.mrv_cost
        enrolled = sum(1 for v in self.prev_accepted.values() if v >= 0)
        return {"carbon": self.carbon, "revenue": self.revenue,
                "payments": self.spent, "mrv": self.mrv_cost,
                "profit": profit, "enrolled": enrolled,
                "budget_left": self.budget - self.spent - self.mrv_cost}

    def __repr__(self):
        return f"Agg(budget={self.budget:.0f}, spent={self.spent:.0f}, carbon={self.carbon:.1f})"