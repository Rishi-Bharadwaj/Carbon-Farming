"""
Carbon Farming Contract Design - Aggregator (Minimal Workshop Version)

Designs three structurally independent contracts:
    Action:  per-action payments for each payable practice + upfront_frac + mrv_share
    Result:  per_ton_payment + mrv_share
    Hybrid:  per-action payments + per_ton_payment + upfront_frac + mrv_share
"""

import numpy as np
from gymnasium import spaces

from farmer import (
    CROPS, PAYABLE_ACTION_NAMES, CONTRACT_TYPES,
)

# Bounds
MAX_PER_ACTION_PAY = 50.0    # $/acre per qualifying practice
MAX_PER_TON_PAY = 60.0       # $/ton CO2e

# MRV cost per farm-unit (100 acres) by contract type
MRV_COSTS = {"action": 5.0, "result": 25.0, "hybrid": 15.0}

# Carbon market price
CARBON_PRICE = 35.0

# Param counts per contract type (derived from domain constants)
ACTION_PARAMS = len(PAYABLE_ACTION_NAMES) + 2   # per-action pays + upfront_frac + mrv_share
RESULT_PARAMS = 2                                 # per_ton_payment + mrv_share
HYBRID_PARAMS = len(PAYABLE_ACTION_NAMES) + 3    # per-action pays + per_ton + upfront_frac + mrv_share
TOTAL_PARAMS = ACTION_PARAMS + RESULT_PARAMS + HYBRID_PARAMS


class Aggregator:
    """Carbon credit aggregator that designs a menu of three independent contracts."""

    def __init__(self, budget, carbon_weight=0.5):
        """
        Args:
            budget: total spending limit per period
            carbon_weight: weight on carbon in reward, 0=pure profit, 1=pure carbon
        """
        self.budget = budget
        self.carbon_weight = carbon_weight

        self.prev_accepted = {}
        self.prev_carbon = {}

        self.spent = 0.0
        self.mrv_cost_total = 0.0
        self.carbon_total = 0.0
        self.revenue_total = 0.0
        self.menu = None

    @staticmethod
    def build_action_space():
        """Continuous params for 3 independent contracts, stacked into one Box."""
        low = np.zeros(TOTAL_PARAMS, dtype=np.float32)
        high_parts = []
        # Action contract: per-action pays + upfront_frac + mrv_share
        high_parts += [MAX_PER_ACTION_PAY] * len(PAYABLE_ACTION_NAMES) + [1.0, 1.0]
        # Result contract: per_ton + mrv_share
        high_parts += [MAX_PER_TON_PAY, 1.0]
        # Hybrid contract: per-action pays + per_ton + upfront_frac + mrv_share
        high_parts += [MAX_PER_ACTION_PAY] * len(PAYABLE_ACTION_NAMES) + [MAX_PER_TON_PAY, 1.0, 1.0]
        high = np.array(high_parts, dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)

    @staticmethod
    def build_observation_space(num_farmers):
        """Aggregator sees per-farmer public/noisy info + global state."""
        per_farmer_dim = 1 + 1 + len(CROPS)  # farm_size + soil_signal + inferred_crop_pref
        return spaces.Dict({
            "farmer_features": spaces.Box(
                low=-1.0, high=np.inf,
                shape=(num_farmers, per_farmer_dim), dtype=np.float32),
            "remaining_budget_fraction": spaces.Box(
                low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "previous_acceptance": spaces.Box(
                low=-1.0, high=len(CONTRACT_TYPES),
                shape=(num_farmers,), dtype=np.float32),
            "previous_carbon": spaces.Box(
                low=0.0, high=np.inf,
                shape=(num_farmers,), dtype=np.float32),
        })


    def get_observation(self, farmers):
        """Build observation from public/noisy farmer info and own state."""
        farmer_features = []
        for f in farmers:
            row = [f.get_public_farm_size(),
                   f.get_noisy_soil_quality()]
            row += f.get_inferred_crop_preference()
            farmer_features.append(row)

        prev_accept = [self.prev_accepted.get(f.fid, -1.0) for f in farmers]
        prev_carbon = [self.prev_carbon.get(f.fid, 0.0) for f in farmers]
        budget_frac = (self.budget - self.spent - self.mrv_cost_total) / max(self.budget, 1e-8)

        return {
            "farmer_features": np.array(farmer_features, dtype=np.float32),
            "remaining_budget_fraction": np.array([budget_frac], dtype=np.float32),
            "previous_acceptance": np.array(prev_accept, dtype=np.float32),
            "previous_carbon": np.array(prev_carbon, dtype=np.float32),
        }


    def decode_action_to_contracts(self, raw_action):
        """
        Convert flat action vector into three independent contract param dicts.
        Returns dict with keys 'action', 'result', 'hybrid'.
        """
        raw_action = np.clip(raw_action, 0.0, None)
        idx = 0

        # --- Action contract ---
        action_pays = {}
        for name in PAYABLE_ACTION_NAMES:
            action_pays[name] = float(np.clip(raw_action[idx], 0, MAX_PER_ACTION_PAY))
            idx += 1
        action_contract = {
            "per_action_payments": action_pays,
            "upfront_fraction": float(np.clip(raw_action[idx], 0, 1)),
            "mrv_share": float(np.clip(raw_action[idx + 1], 0, 1)),
        }
        idx += 2

        # --- Result contract ---
        result_contract = {
            "per_ton_payment": float(np.clip(raw_action[idx], 0, MAX_PER_TON_PAY)),
            "mrv_share": float(np.clip(raw_action[idx + 1], 0, 1)),
        }
        idx += 2

        # --- Hybrid contract ---
        hybrid_pays = {}
        for name in PAYABLE_ACTION_NAMES:
            hybrid_pays[name] = float(np.clip(raw_action[idx], 0, MAX_PER_ACTION_PAY))
            idx += 1
        hybrid_contract = {
            "per_action_payments": hybrid_pays,
            "per_ton_payment": float(np.clip(raw_action[idx], 0, MAX_PER_TON_PAY)),
            "upfront_fraction": float(np.clip(raw_action[idx + 1], 0, 1)),
            "mrv_share": float(np.clip(raw_action[idx + 2], 0, 1)),
        }

        self.menu = {
            "action": action_contract,
            "result": result_contract,
            "hybrid": hybrid_contract,
        }
        return self.menu



    def get_mrv_cost(self, contract_type, farm_size):
        """Total MRV cost for a farmer given contract type and farm size."""
        return MRV_COSTS[contract_type] * farm_size

    def process_farmer_outcome(self, farmer, contract_type, payment, carbon):
        """Update running totals after a farmer's period is resolved."""
        params = self.menu[contract_type]
        mrv = self.get_mrv_cost(contract_type, farmer.farm_size)
        agg_mrv = (1.0 - params["mrv_share"]) * mrv

        self.spent += payment
        self.mrv_cost_total += agg_mrv
        self.carbon_total += carbon
        self.revenue_total += carbon * CARBON_PRICE

        self.prev_accepted[farmer.fid] = CONTRACT_TYPES.index(contract_type)
        self.prev_carbon[farmer.fid] = carbon

    def process_farmer_rejection(self, farmer):
        """Record that a farmer rejected all contracts."""
        self.prev_accepted[farmer.fid] = -1.0
        self.prev_carbon[farmer.fid] = 0.0


    def compute_reward(self):
        """
        Weighted combination of profit and carbon value, with budget penalty.
        carbon_weight=0 -> pure profit, carbon_weight=1 -> pure carbon.
        """
        profit = self.revenue_total - self.spent - self.mrv_cost_total

        reward = (1.0 - self.carbon_weight) * profit + self.carbon_weight * self.carbon_total

        overspend = (self.spent + self.mrv_cost_total) - self.budget
        if overspend > 0:
            reward -= overspend * 2.0
        return reward


    def reset_period(self):
        """Reset per-period totals. History preserved for multi-period."""
        self.spent = self.mrv_cost_total = self.carbon_total = self.revenue_total = 0.0
        self.menu = None

    def reset_episode(self):
        """Full reset including history."""
        self.reset_period()
        self.prev_accepted = {}
        self.prev_carbon = {}

    def get_period_summary(self):
        """Return dict summarizing the current period."""
        profit = self.revenue_total - self.spent - self.mrv_cost_total
        enrolled = sum(1 for v in self.prev_accepted.values() if v >= 0)
        return {
            "carbon": self.carbon_total,
            "revenue": self.revenue_total,
            "payments": self.spent,
            "mrv_cost": self.mrv_cost_total,
            "profit": profit,
            "enrolled": enrolled,
            "budget_remaining": self.budget - self.spent - self.mrv_cost_total,
        }

    def __repr__(self):
        return (f"Aggregator(budget={self.budget:.0f}, spent={self.spent:.0f}, "
                f"carbon={self.carbon_total:.1f})")