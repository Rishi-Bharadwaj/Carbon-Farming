"""
Carbon Farming Contract Design - Aggregator Agent
==================================================
Aggregator class representing the carbon credit intermediary (leader)
in the Stackelberg game.

The aggregator:
    - Observes public/noisy farmer characteristics
    - Designs a menu of 3 contracts (action-based, result-based, hybrid)
    - Collects carbon credits from participating farmers
    - Sells credits on the voluntary carbon market
    - Maximizes a weighted combination of profit and total carbon sequestered

Information available to aggregator:
    - Public: farm_size, previous_crop_choices (per farmer)
    - Noisy: soil_quality signal (per farmer)
    - Inferred: crop_preference estimate from history (per farmer)
    - NOT observable: input_preference, risk_preference
"""

import numpy as np
from gymnasium import spaces

from farmer_claude import (
    CROPS, INPUTS, TILLAGE_OPTIONS,
    Farmer,
)


# ============================================================
# Aggregator Constants
# ============================================================

# Contract parameter bounds (aggregator chooses within these)
MAX_PER_ACRE_PAYMENT = 50.0       # $/acre max the aggregator can offer
MAX_PER_TON_PAYMENT = 60.0        # $/ton CO2e max
MAX_UPFRONT_FRACTION = 1.0        # fraction of action payment paid upfront
MAX_MRV_COST_SHARE = 1.0          # 0 = aggregator pays all MRV, 1 = farmer pays all

# MRV costs
MRV_COST_ACTION = 5.0             # $ per farm-unit (100 acres) — cheap satellite verification
MRV_COST_RESULT = 25.0            # $ per farm-unit (100 acres) — expensive soil sampling + modeling
MRV_COST_HYBRID = 15.0            # $ per farm-unit (100 acres) — moderate (satellite + spot checks)

# Carbon market
CARBON_MARKET_PRICE = 35.0        # $/ton CO2e — price aggregator receives when selling credits

# Number of parameters per contract the aggregator sets
# [per_acre_payment, per_ton_payment, upfront_fraction, mrv_cost_share]
# lambda is fixed per contract type in Option B:
#   action: lambda=1, result: lambda=0, hybrid: lambda=0.5
NUM_PARAMS_PER_CONTRACT = 4
NUM_CONTRACTS = 3  # action, result, hybrid


class Aggregator:
    """
    Represents the carbon credit aggregator (principal/leader) in the game.

    The aggregator designs a menu of three structurally distinct contracts
    (Option B) and offers them to all farmers. It observes public and noisy
    farmer characteristics but not private preferences.

    Attributes:
        budget (float): Total budget available for farmer payments per period.
        carbon_weight (float): Weight on carbon in objective. 0 = pure profit
            maximizer, 1 = pure carbon maximizer.
        previous_contracts_accepted (dict): Mapping farmer_id -> contract type
            accepted in last period (for multi-period tracking).
        previous_outcomes (dict): Mapping farmer_id -> dict of carbon, profit
            from last period.
        total_spent (float): Cumulative payments made this period.
        total_carbon (float): Cumulative carbon sequestered this period.
        total_mrv_cost (float): Cumulative MRV costs borne by aggregator.
    """

    def __init__(self, budget: float, carbon_weight: float = 0.5):
        assert budget > 0, "Budget must be positive"
        assert 0.0 <= carbon_weight <= 1.0, "carbon_weight must be in [0, 1]"

        self.budget = budget
        self.carbon_weight = carbon_weight

        # ---- History from previous periods ----
        self.previous_contracts_accepted = {}  # farmer_id -> int (0=action,1=result,2=hybrid,None=rejected)
        self.previous_outcomes = {}            # farmer_id -> {"carbon": float, "profit": float}

        # ---- Current period tracking ----
        self.total_spent = 0.0
        self.total_carbon = 0.0
        self.total_mrv_cost = 0.0
        self.total_revenue = 0.0

        # ---- Current contract menu (set by action) ----
        self.current_contracts = None

    # ================================================================
    # Gym Spaces
    # ================================================================

    @staticmethod
    def action_space() -> spaces.Box:
        """
        Aggregator's action space: parameters for 3 contracts.

        For each contract the aggregator sets 4 continuous parameters:
            per_acre_payment:  [0, MAX_PER_ACRE_PAYMENT]
            per_ton_payment:   [0, MAX_PER_TON_PAYMENT]
            upfront_fraction:  [0, 1]
            mrv_cost_share:    [0, 1]  (fraction of MRV cost passed to farmer)

        Lambda is fixed per contract type (Option B):
            action=1.0, result=0.0, hybrid=0.5

        Total action dim = 3 contracts * 4 params = 12
        """
        low = np.zeros(NUM_CONTRACTS * NUM_PARAMS_PER_CONTRACT, dtype=np.float32)
        high = np.array(
            # Action contract bounds
            [MAX_PER_ACRE_PAYMENT, MAX_PER_TON_PAYMENT, MAX_UPFRONT_FRACTION, MAX_MRV_COST_SHARE,
            # Result contract bounds
             MAX_PER_ACRE_PAYMENT, MAX_PER_TON_PAYMENT, MAX_UPFRONT_FRACTION, MAX_MRV_COST_SHARE,
            # Hybrid contract bounds
             MAX_PER_ACRE_PAYMENT, MAX_PER_TON_PAYMENT, MAX_UPFRONT_FRACTION, MAX_MRV_COST_SHARE],
            dtype=np.float32,
        )
        return spaces.Box(low=low, high=high, dtype=np.float32)

    @staticmethod
    def observation_space(num_farmers: int = 5) -> spaces.Dict:
        """
        Aggregator's observation space.

        Per farmer (public/noisy info):
            farm_size:              int [1, 10]
            soil_quality_signal:    float [0, 1] (noisy)
            inferred_crop_pref:     float[4] (empirical freq from history)

        Global state:
            remaining_budget:       float
            previous_acceptance:    float[num_farmers] (-1=none, 0=action, 1=result, 2=hybrid)
            previous_carbon:        float[num_farmers] (carbon from last period per farmer)
        """
        # Per-farmer obs: farm_size(1) + soil_signal(1) + inferred_crop_pref(4) = 6 per farmer
        per_farmer_dim = 6
        farmer_obs_dim = num_farmers * per_farmer_dim

        return spaces.Dict({
            "farmer_observations": spaces.Box(
                low=-1.0, high=1000.0,
                shape=(farmer_obs_dim,), dtype=np.float32
            ),
            "remaining_budget": spaces.Box(
                low=0.0, high=np.inf,
                shape=(1,), dtype=np.float32
            ),
            "previous_acceptance": spaces.Box(
                low=-1.0, high=2.0,
                shape=(num_farmers,), dtype=np.float32
            ),
            "previous_carbon": spaces.Box(
                low=0.0, high=np.inf,
                shape=(num_farmers,), dtype=np.float32
            ),
        })

    # ================================================================
    # Observation Construction
    # ================================================================

    def get_observation(self, farmers: list) -> dict:
        """
        Construct the aggregator's observation from the list of farmers.
        Only accesses public/noisy information — never private attributes.

        Args:
            farmers: list of Farmer objects

        Returns:
            dict matching observation_space
        """
        num_farmers = len(farmers)
        farmer_obs_list = []

        for f in farmers:
            # Public info
            public = f.get_public_info()
            farm_size = public["farm_size"]

            # Noisy soil signal
            soil_signal = f.get_noisy_soil_signal()

            # Inferred crop preference from history
            inferred_pref = f.get_inferred_crop_preference()
            pref_vec = [inferred_pref[c] for c in CROPS]

            farmer_obs_list.extend([farm_size, soil_signal] + pref_vec)

        # Previous period history
        prev_accept = np.array([
            self.previous_contracts_accepted.get(f.farmer_id, -1.0)
            for f in farmers
        ], dtype=np.float32)

        prev_carbon = np.array([
            self.previous_outcomes.get(f.farmer_id, {}).get("carbon", 0.0)
            for f in farmers
        ], dtype=np.float32)

        return {
            "farmer_observations": np.array(farmer_obs_list, dtype=np.float32),
            "remaining_budget": np.array([self.budget - self.total_spent], dtype=np.float32),
            "previous_acceptance": prev_accept,
            "previous_carbon": prev_carbon,
        }

    # ================================================================
    # Action -> Contract Menu
    # ================================================================

    # Fixed lambdas for Option B
    FIXED_LAMBDAS = {
        "action": 1.0,
        "result": 0.0,
        "hybrid": 0.5,
    }

    def decode_action(self, action: np.ndarray) -> dict:
        """
        Convert the raw action vector (12 floats) into a contract menu dict.

        Args:
            action: np.ndarray of shape (12,) from the policy network

        Returns:
            dict with keys "action", "result", "hybrid", each containing
            a contract parameter dict compatible with Farmer.update_contracts_offered
        """
        action = np.clip(action, 0.0, None)  # ensure non-negative

        contracts = {}
        contract_names = ["action", "result", "hybrid"]

        for i, name in enumerate(contract_names):
            offset = i * NUM_PARAMS_PER_CONTRACT
            params = action[offset: offset + NUM_PARAMS_PER_CONTRACT]

            contracts[name] = {
                "per_acre_payment": float(np.clip(params[0], 0.0, MAX_PER_ACRE_PAYMENT)),
                "per_ton_payment": float(np.clip(params[1], 0.0, MAX_PER_TON_PAYMENT)),
                "upfront_fraction": float(np.clip(params[2], 0.0, 1.0)),
                "mrv_cost_share": float(np.clip(params[3], 0.0, 1.0)),
                "lambda": self.FIXED_LAMBDAS[name],
            }

        self.current_contracts = contracts
        return contracts

    # ================================================================
    # Payment & Outcome Processing
    # ================================================================

    def get_mrv_cost(self, contract_type: str, farm_size: int) -> float:
        """
        Compute total MRV cost for a farmer given their contract type.

        Args:
            contract_type: "action", "result", or "hybrid"
            farm_size: farmer's farm_size attribute (units, each = 100 acres)

        Returns:
            Total MRV cost in dollars
        """
        acres = farm_size * 100
        cost_per_acre = {
            "action": MRV_COST_ACTION,
            "result": MRV_COST_RESULT,
            "hybrid": MRV_COST_HYBRID,
        }
        return cost_per_acre[contract_type] * (acres / 100.0)

    def process_farmer_outcome(
        self,
        farmer: Farmer,
        contract_type: str,
        farmer_payment: float,
        carbon_sequestered: float,
    ):
        """
        Process a single farmer's outcome for this period.
        Updates aggregator's running totals and history.

        Args:
            farmer: the Farmer object
            contract_type: "action", "result", or "hybrid"
            farmer_payment: total payment made to this farmer
            carbon_sequestered: tons CO2e this farmer sequestered
        """
        mrv_cost = self.get_mrv_cost(contract_type, farmer.farm_size)
        contract_params = self.current_contracts[contract_type]

        # MRV cost borne by aggregator = (1 - farmer's share) * total MRV cost
        agg_mrv_cost = (1.0 - contract_params["mrv_cost_share"]) * mrv_cost

        # Revenue from selling carbon credits
        credit_revenue = carbon_sequestered * CARBON_MARKET_PRICE

        # Update running totals
        self.total_spent += farmer_payment
        self.total_mrv_cost += agg_mrv_cost
        self.total_carbon += carbon_sequestered
        self.total_revenue += credit_revenue

        # Store for next period's observations
        self.previous_contracts_accepted[farmer.farmer_id] = (
            ["action", "result", "hybrid"].index(contract_type)
        )
        self.previous_outcomes[farmer.farmer_id] = {
            "carbon": carbon_sequestered,
            "profit": credit_revenue - farmer_payment - agg_mrv_cost,
        }

    def process_rejected_farmer(self, farmer: Farmer):
        """Record that a farmer rejected all contracts."""
        self.previous_contracts_accepted[farmer.farmer_id] = -1.0
        self.previous_outcomes[farmer.farmer_id] = {
            "carbon": 0.0,
            "profit": 0.0,
        }

    # ================================================================
    # Reward
    # ================================================================

    def reward(self) -> float:
        """
        Aggregator's reward for the current period.

        reward = (1 - carbon_weight) * profit + carbon_weight * carbon_value

        Where:
            profit = total_revenue - total_spent - total_mrv_cost
            carbon_value = total_carbon * CARBON_MARKET_PRICE
                (carbon valued at market price for comparability with profit)

        At carbon_weight = 0: pure profit maximizer
        At carbon_weight = 1: pure carbon maximizer (spends entire budget if needed)
        At carbon_weight = 0.5: balanced objective
        """
        profit = self.total_revenue - self.total_spent - self.total_mrv_cost
        carbon_value = self.total_carbon * CARBON_MARKET_PRICE

        reward = (1.0 - self.carbon_weight) * profit + self.carbon_weight * carbon_value
        return reward

    def is_budget_exceeded(self) -> bool:
        """Check if total spending has exceeded the budget."""
        return (self.total_spent + self.total_mrv_cost) > self.budget

    def budget_penalty(self) -> float:
        """
        Penalty for exceeding budget. Applied as negative reward.
        Returns 0 if within budget, negative value if exceeded.
        """
        overspend = (self.total_spent + self.total_mrv_cost) - self.budget
        if overspend > 0:
            return -overspend * 2.0  # 2x penalty for overspending
        return 0.0

    # ================================================================
    # State Management
    # ================================================================

    def reset(self):
        """Reset per-episode state. Budget and carbon_weight are preserved."""
        self.total_spent = 0.0
        self.total_carbon = 0.0
        self.total_mrv_cost = 0.0
        self.total_revenue = 0.0
        self.current_contracts = None
        # Note: previous_contracts_accepted and previous_outcomes are
        # preserved across periods within an episode for multi-period.
        # Call hard_reset() to clear everything.

    def hard_reset(self):
        """Full reset including history. Call between episodes."""
        self.reset()
        self.previous_contracts_accepted = {}
        self.previous_outcomes = {}

    # ================================================================
    # Logging / Debug
    # ================================================================

    def get_period_summary(self) -> dict:
        """Return a summary of the current period for logging."""
        profit = self.total_revenue - self.total_spent - self.total_mrv_cost
        return {
            "total_carbon": self.total_carbon,
            "total_revenue": self.total_revenue,
            "total_payments": self.total_spent,
            "total_mrv_cost": self.total_mrv_cost,
            "profit": profit,
            "budget_remaining": self.budget - self.total_spent - self.total_mrv_cost,
            "reward": self.reward(),
            "num_farmers_enrolled": sum(
                1 for v in self.previous_contracts_accepted.values() if v >= 0
            ),
            "contracts_offered": self.current_contracts,
        }

    def __repr__(self):
        return (
            f"Aggregator(budget={self.budget:.0f}, "
            f"carbon_weight={self.carbon_weight:.2f}, "
            f"spent={self.total_spent:.2f}, "
            f"carbon={self.total_carbon:.2f})"
        )


# ============================================================
# Quick Sanity Test
# ============================================================

if __name__ == "__main__":
    from farmer_claude import create_workshop_farmers

    # Create agents
    farmers = create_workshop_farmers()
    agg = Aggregator(budget=50000.0, carbon_weight=0.5)

    print("=" * 60)
    print("Aggregator Setup")
    print("=" * 60)
    print(f"{agg}")
    print(f"Action space: {Aggregator.action_space()}")
    print(f"Observation space keys: {list(Aggregator.observation_space(num_farmers=5).keys())}")

    # ---- Simulate one full round ----
    print("\n" + "=" * 60)
    print("Simulating One Round")
    print("=" * 60)

    # Step 1: Aggregator observes farmers
    obs = agg.get_observation(farmers)
    print(f"\nAggregator obs (farmer features): {obs['farmer_observations'][:12]}...")  # first 2 farmers
    print(f"Remaining budget: {obs['remaining_budget']}")

    # Step 2: Aggregator chooses contract parameters (random for test)
    raw_action = np.random.uniform(
        low=Aggregator.action_space().low,
        high=Aggregator.action_space().high,
    )
    contracts = agg.decode_action(raw_action)

    print(f"\nContracts offered:")
    for name, params in contracts.items():
        print(f"  {name}: {params}")

    # Step 3: Distribute contracts to farmers
    for f in farmers:
        f.update_contracts_offered(contracts)

    # Step 4: Each farmer decides (simulate random actions for test)
    weather_shock = np.random.normal(0, 1)
    contract_names = ["action", "result", "hybrid"]

    for f in farmers:
        # Random farmer action
        actions = {
            "contract_choice": np.random.randint(0, 4),
            "crop_choice": np.random.randint(0, 4),
            "input_choice": np.random.randint(0, len(INPUTS)),
            "tillage_choice": np.random.randint(0, 2),
        }
        f.step(actions)

        if f.current_contract is not None:
            ct_name = contract_names[f.current_contract]
            outcomes = f.compute_outcomes(actions, weather_shock)
            mrv_cost = agg.get_mrv_cost(ct_name, f.farm_size)
            payment = f.compute_contract_payment(contracts[ct_name], mrv_cost)
            farmer_reward = f.reward()

            agg.process_farmer_outcome(f, ct_name, payment, outcomes["carbon"])

            print(f"\n  {f} -> accepted {ct_name}")
            print(f"    Carbon: {outcomes['carbon']:.2f} tons, Payment: ${payment:.2f}")
            print(f"    Farmer reward: {farmer_reward:.2f}")
        else:
            agg.process_rejected_farmer(f)
            # Still compute outcomes for the farmer (they farm without a contract)
            outcomes = f.compute_outcomes(actions, weather_shock)
            farmer_reward = f.reward()

            print(f"\n  {f} -> rejected all contracts")
            print(f"    Farming independently, reward: {farmer_reward:.2f}")

    # Step 5: Aggregator reward
    print("\n" + "=" * 60)
    print("Period Summary")
    print("=" * 60)
    summary = agg.get_period_summary()
    for k, v in summary.items():
        if k != "contracts_offered":
            print(f"  {k}: {v}")
    print(f"\n  Aggregator reward: {agg.reward():.2f}")
    print(f"  Budget penalty: {agg.budget_penalty():.2f}")
    print(f"  Final reward (with penalty): {agg.reward() + agg.budget_penalty():.2f}")