"""
Carbon Farming Contract Design - Aggregator (Minimal Workshop Version)

Designs three structurally independent contracts:
    Action:  per-action payments for each payable practice + upfront_frac + mrv_share
    Result:  per_ton_payment + mrv_share
    Hybrid:  per-action payments + per_ton_payment + upfront_frac + mrv_share
"""

import logging
import numpy as np
from gymnasium import spaces

from farmer import (
    CROPS, PAYABLE_ACTION_NAMES, CONTRACT_TYPES,
)

logger = logging.getLogger("carbon_farming.aggregator")

# Bounds
MAX_PER_ACTION_PAY = 50.0
MAX_PER_TON_PAY = 60.0

# MRV cost per acre by contract type
MRV_COST_PER_ACRE = {"action": 5.0, "result": 25.0, "hybrid": 15.0}

# Carbon market price
CARBON_PRICE = 35.0

# Param counts per contract (derived from domain constants)
ACTION_PARAMS = len(PAYABLE_ACTION_NAMES) + 2
RESULT_PARAMS = 2
HYBRID_PARAMS = len(PAYABLE_ACTION_NAMES) + 3
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

        logger.debug(f"[Aggregator] Created: budget=${budget:.0f}, carbon_weight={carbon_weight:.2f}")

    # ================================================================
    # Spaces
    # ================================================================

    @staticmethod
    def build_action_space():
        """Continuous params for 3 independent contracts."""
        low = np.zeros(TOTAL_PARAMS, dtype=np.float32)
        high_parts = []
        high_parts += [MAX_PER_ACTION_PAY] * len(PAYABLE_ACTION_NAMES) + [1.0, 1.0]
        high_parts += [MAX_PER_TON_PAY, 1.0]
        high_parts += [MAX_PER_ACTION_PAY] * len(PAYABLE_ACTION_NAMES) + [MAX_PER_TON_PAY, 1.0, 1.0]
        high = np.array(high_parts, dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)

    @staticmethod
    def build_observation_space(num_farmers):
        """Aggregator sees per-farmer public/noisy info + global state."""
        per_farmer_dim = 1 + 1 + len(CROPS)
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

    # ================================================================
    # Observations
    # ================================================================

    def get_observation(self, farmers):
        """Build observation from public/noisy farmer info and own state."""
        farmer_features = []
        for f in farmers:
            row = [f.get_public_farm_size(), f.get_noisy_soil_quality()]
            row += f.get_inferred_crop_preference()
            farmer_features.append(row)

        prev_accept = [self.prev_accepted.get(f.fid, -1.0) for f in farmers]
        prev_carbon = [self.prev_carbon.get(f.fid, 0.0) for f in farmers]
        budget_frac = (self.budget - self.spent - self.mrv_cost_total) / max(self.budget, 1e-8)

        logger.debug(f"[Aggregator] Observation: budget_frac={budget_frac:.3f}, "
                     f"prev_accept={prev_accept}, prev_carbon={[f'{c:.2f}' for c in prev_carbon]}")

        return {
            "farmer_features": np.array(farmer_features, dtype=np.float32),
            "remaining_budget_fraction": np.array([budget_frac], dtype=np.float32),
            "previous_acceptance": np.array(prev_accept, dtype=np.float32),
            "previous_carbon": np.array(prev_carbon, dtype=np.float32),
        }

    # ================================================================
    # Action decoding
    # ================================================================

    def decode_action_to_contracts(self, raw_action):
        """Convert flat action vector into three independent contract param dicts."""
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

        logger.info(f"[Aggregator] CONTRACT MENU POSTED:")
        for ct_name, params in self.menu.items():
            logger.info(f"  {ct_name}: {params}")

        return self.menu

    # ================================================================
    # Outcome processing
    # ================================================================

    def get_mrv_cost(self, contract_type, farm_size):
        """Total MRV cost = cost_per_acre × farm_size."""
        cost = MRV_COST_PER_ACRE[contract_type] * farm_size
        logger.debug(f"  MRV cost ({contract_type}): ${MRV_COST_PER_ACRE[contract_type]:.2f}/ac "
                     f"× {farm_size} ac = ${cost:.2f}")
        return cost

    def process_farmer_outcome(self, farmer, contract_type, payment, carbon):
        """Update running totals after a farmer's period is resolved."""
        params = self.menu[contract_type]
        mrv = self.get_mrv_cost(contract_type, farmer.farm_size)
        agg_mrv = (1.0 - params["mrv_share"]) * mrv
        credit_revenue = carbon * CARBON_PRICE

        self.spent += payment
        self.mrv_cost_total += agg_mrv
        self.carbon_total += carbon
        self.revenue_total += credit_revenue

        self.prev_accepted[farmer.fid] = CONTRACT_TYPES.index(contract_type)
        self.prev_carbon[farmer.fid] = carbon

        logger.info(f"[Aggregator] Processed Farmer {farmer.fid} ({contract_type}):")
        logger.info(f"  payment=${payment:.2f}, carbon={carbon:.4f}t, "
                    f"credit_rev=${credit_revenue:.2f}, "
                    f"agg_mrv=${agg_mrv:.2f} (total_mrv=${mrv:.2f} × agg_share={1 - params['mrv_share']:.2f})")
        logger.info(f"  Running totals: spent=${self.spent:.2f}, mrv=${self.mrv_cost_total:.2f}, "
                    f"carbon={self.carbon_total:.4f}t, revenue=${self.revenue_total:.2f}")

    def process_farmer_rejection(self, farmer):
        """Record that a farmer rejected all contracts."""
        self.prev_accepted[farmer.fid] = -1.0
        self.prev_carbon[farmer.fid] = 0.0
        logger.info(f"[Aggregator] Farmer {farmer.fid} REJECTED all contracts")

    # ================================================================
    # Reward
    # ================================================================

    def compute_reward(self):
        """Weighted profit + carbon, with budget penalty."""
        profit = self.revenue_total - self.spent - self.mrv_cost_total
        carbon_value = self.carbon_total * CARBON_PRICE
        base_reward = (1.0 - self.carbon_weight) * profit + self.carbon_weight * carbon_value

        overspend = (self.spent + self.mrv_cost_total) - self.budget
        penalty = max(overspend, 0) * 2.0
        reward = base_reward - penalty

        logger.info(f"[Aggregator] REWARD:")
        logger.info(f"  profit = revenue ${self.revenue_total:.2f} - payments ${self.spent:.2f} "
                    f"- mrv ${self.mrv_cost_total:.2f} = ${profit:.2f}")
        logger.info(f"  carbon_value = {self.carbon_total:.4f}t × ${CARBON_PRICE:.2f} = ${carbon_value:.2f}")
        logger.info(f"  base_reward = (1-{self.carbon_weight:.2f})×${profit:.2f} + "
                    f"{self.carbon_weight:.2f}×${carbon_value:.2f} = ${base_reward:.2f}")
        if penalty > 0:
            logger.info(f"  BUDGET PENALTY: overspend=${overspend:.2f} × 2 = -${penalty:.2f}")
        logger.info(f"  FINAL REWARD = ${reward:.2f}")
        return reward

    # ================================================================
    # State management
    # ================================================================

    def reset_period(self):
        """Reset per-period totals. History preserved for multi-period."""
        self.spent = self.mrv_cost_total = self.carbon_total = self.revenue_total = 0.0
        self.menu = None
        logger.debug(f"[Aggregator] Period reset")

    def reset_episode(self):
        """Full reset including history."""
        self.reset_period()
        self.prev_accepted = {}
        self.prev_carbon = {}
        logger.debug(f"[Aggregator] Episode reset")

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