"""
Carbon Farming Contract Design - Farmer Agent
=============================================
Farmer class representing a heterogeneous agricultural agent in the
carbon farming contract design Stackelberg game.

Information Structure:
    - Public (observed by aggregator): farm_size, previous_crop_choices
    - Observable with noise: soil_quality (aggregator sees signal + error)
    - Inferred: crop_preference (aggregator has prior from crop history)
    - Private: input_preference, risk_preference
"""

import numpy as np
from gymnasium import spaces


# ============================================================
# Constants — tune these as experimental parameters
# ============================================================

CROPS = ["corn", "soybean", "wheat", "cover_crop"]
INPUTS = ["chemical_fertilizer", "manure", "none"]  # 3 input types
TILLAGE_OPTIONS = ["conventional", "no_till"]

# Number of contract slots offered by aggregator (Option B: fixed structure)
# 0 = action-based, 1 = result-based, 2 = hybrid, 3 = no contract
NUM_CONTRACT_CHOICES = 4

# History length for previous crop choices
CROP_HISTORY_LENGTH = 5

# Yield model parameters
BASE_YIELD_PER_ACRE = 150.0       # bushels per acre baseline
FERTILIZER_YIELD_BOOST = 0.4      # max fractional yield increase from fertilizer
SOIL_QUALITY_YIELD_FACTOR = 0.5   # how much soil quality affects yield
WEATHER_YIELD_STD = 0.15          # std dev of weather shock on yield
NO_TILL_YIELD_PENALTY = 0.05      # short-term yield penalty for no-till

# Carbon model parameters
BASE_CARBON_PER_ACRE = 0.2        # tons CO2e per acre baseline
COVER_CROP_CARBON_BONUS = 0.3     # additional tons per acre from cover crops
NO_TILL_CARBON_BONUS = 0.25       # additional tons per acre from no-till
LOW_FERT_CARBON_BONUS = 0.15      # additional tons per acre from low fertilizer
SOIL_QUALITY_CARBON_FACTOR = 0.3  # how much soil quality affects sequestration
WEATHER_CARBON_STD = 0.1          # std dev of weather shock on carbon

# Cost parameters
CROP_BASE_COSTS = {
    "corn": 300.0,
    "soybean": 200.0,
    "wheat": 180.0,
    "cover_crop": 120.0,
}
INPUT_COSTS = {
    "chemical_fertilizer": 80.0,
    "manure": 50.0,
    "none": 0.0,
}
NO_TILL_TRANSITION_COST = 40.0    # per acre cost of switching to no-till
CROP_PRICES = {
    "corn": 4.5,       # $ per bushel
    "soybean": 12.0,
    "wheat": 6.5,
    "cover_crop": 0.0,  # no market value, grown for soil benefits
}


class Farmer:
    """
    Represents a heterogeneous farmer agent in the carbon farming game.

    Attributes:
        farm_size (int): Farm size in units [1, 10], each unit = 100 acres.
        soil_quality (float): True soil quality in [0, 1]. Higher is better.
        crop_preference (dict): Private preference weights over crops.
            e.g. {"corn": 0.5, "soybean": 0.3, "wheat": 0.15, "cover_crop": 0.05}
        input_preference (dict): Private preference weights over input types.
            e.g. {"chemical_fertilizer": 0.6, "manure": 0.3, "none": 0.1}
        risk_preference (float): Risk aversion coefficient in [0, 1].
            0 = risk neutral, 1 = extremely risk averse.
        previous_crop_choices (list): History of past crop selections (public).
    """

    def __init__(
        self,
        farm_size: int,
        soil_quality: float,
        crop_preference: dict,
        input_preference: dict,
        risk_preference: float,
        farmer_id: int = 0,
    ):
        # ---- Validate inputs ----
        assert 1 <= farm_size <= 10, "farm_size must be in [1, 10]"
        assert 0.0 <= soil_quality <= 1.0, "soil_quality must be in [0, 1]"
        assert 0.0 <= risk_preference <= 1.0, "risk_preference must be in [0, 1]"
        assert set(crop_preference.keys()) == set(CROPS), "crop_preference must have keys: " + str(CROPS)
        assert set(input_preference.keys()) == set(INPUTS), "input_preference must have keys: " + str(INPUTS)

        self.farmer_id = farmer_id
        self.farm_size = farm_size
        self.soil_quality = soil_quality
        self.crop_preference = crop_preference
        self.input_preference = input_preference
        self.risk_preference = risk_preference

        # ---- Public information ----
        # Initialize crop history by sampling from crop preference distribution
        crop_probs = np.array([crop_preference[c] for c in CROPS])
        crop_probs = crop_probs / crop_probs.sum()  # normalize
        self.previous_crop_choices = list(
            np.random.choice(CROPS, size=CROP_HISTORY_LENGTH, p=crop_probs)
        )

        # ---- Observation: contracts currently offered by aggregator ----
        self.contracts_offered = None  # set by environment each step

        # ---- Track state within an episode ----
        self.current_contract = None       # which contract was accepted (None if rejected all)
        self.current_actions = None        # dict of actions taken this period
        self.current_payment = 0.0         # total payment received this period
        self.current_yield = 0.0           # crop yield this period
        self.current_carbon = 0.0          # carbon sequestered this period
        self.current_cost = 0.0            # total cost incurred this period
        self.current_revenue = 0.0         # crop sale revenue this period

    # ================================================================
    # Gym Spaces
    # ================================================================

    @staticmethod
    def action_space() -> spaces.Dict:
        """
        Farmer's action space (Option B: fixed contract structure).

        Actions:
            contract_choice: Discrete(4)
                0 = accept action-based contract
                1 = accept result-based contract
                2 = accept hybrid contract
                3 = reject all (no contract)
            crop_choice: Discrete(4)
                Index into CROPS list.
            input_choice: Discrete(3)
                Index into INPUTS list.
            tillage_choice: Discrete(2)
                0 = conventional tillage
                1 = no-till
        """
        return spaces.Dict({
            "contract_choice": spaces.Discrete(NUM_CONTRACT_CHOICES),
            "crop_choice": spaces.Discrete(len(CROPS)),
            "input_choice": spaces.Discrete(len(INPUTS)),
            "tillage_choice": spaces.Discrete(len(TILLAGE_OPTIONS)),
        })

    @staticmethod
    def observation_space() -> spaces.Dict:
        """
        Farmer's observation space.

        Observations:
            own_farm_size: farm size (public, known to self)
            own_soil_quality: true soil quality (private, known only to self)
            own_risk_preference: own risk aversion (private)
            crop_preference_vector: own preference weights over crops (private)
            input_preference_vector: own preference weights over inputs (private)
            previous_crop_history: one-hot encoded crop history (public)

            -- Contract information (set by aggregator via environment) --
            action_contract: parameters of the action-based contract offered
            result_contract: parameters of the result-based contract offered
            hybrid_contract: parameters of the hybrid contract offered
        """
        num_crops = len(CROPS)
        num_contract_params = 5  # per_acre_payment, per_ton_payment, upfront_fraction, mrv_cost_share, lambda

        return spaces.Dict({
            # ---- Own characteristics (farmer sees its own full type) ----
            "own_farm_size": spaces.Box(low=1, high=10, shape=(1,), dtype=np.float32),
            "own_soil_quality": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "own_risk_preference": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "crop_preference_vector": spaces.Box(low=0.0, high=1.0, shape=(num_crops,), dtype=np.float32),
            "input_preference_vector": spaces.Box(low=0.0, high=1.0, shape=(len(INPUTS),), dtype=np.float32),
            "previous_crop_history": spaces.MultiDiscrete([num_crops] * CROP_HISTORY_LENGTH),

            # ---- Contract offers (populated by environment from aggregator) ----
            # Each contract: [per_acre_payment, per_ton_payment, upfront_fraction, mrv_cost_share, lambda]
            "action_contract": spaces.Box(low=0.0, high=1000.0, shape=(num_contract_params,), dtype=np.float32),
            "result_contract": spaces.Box(low=0.0, high=1000.0, shape=(num_contract_params,), dtype=np.float32),
            "hybrid_contract": spaces.Box(low=0.0, high=1000.0, shape=(num_contract_params,), dtype=np.float32),
        })

    # ================================================================
    # Observation Construction
    # ================================================================

    def get_observation(self) -> dict:
        """Construct the observation dict for this farmer."""
        crop_pref_vec = np.array(
            [self.crop_preference[c] for c in CROPS], dtype=np.float32
        )
        input_pref_vec = np.array(
            [self.input_preference[i] for i in INPUTS], dtype=np.float32
        )
        crop_history_encoded = np.array(
            [CROPS.index(c) for c in self.previous_crop_choices], dtype=np.int64
        )

        # Default zero contracts if none offered yet
        zero_contract = np.zeros(5, dtype=np.float32)
        action_c = zero_contract
        result_c = zero_contract
        hybrid_c = zero_contract

        if self.contracts_offered is not None:
            action_c = self._contract_to_vector(self.contracts_offered.get("action", {}))
            result_c = self._contract_to_vector(self.contracts_offered.get("result", {}))
            hybrid_c = self._contract_to_vector(self.contracts_offered.get("hybrid", {}))

        return {
            "own_farm_size": np.array([self.farm_size], dtype=np.float32),
            "own_soil_quality": np.array([self.soil_quality], dtype=np.float32),
            "own_risk_preference": np.array([self.risk_preference], dtype=np.float32),
            "crop_preference_vector": crop_pref_vec,
            "input_preference_vector": input_pref_vec,
            "previous_crop_history": crop_history_encoded,
            "action_contract": action_c,
            "result_contract": result_c,
            "hybrid_contract": hybrid_c,
        }

    def update_contracts_offered(self, contracts: dict):
        """
        Receive contract offers from the aggregator via the environment.

        Args:
            contracts: dict with keys "action", "result", "hybrid",
                each containing a contract parameter dict:
                {
                    "per_acre_payment": float,
                    "per_ton_payment": float,
                    "upfront_fraction": float,
                    "mrv_cost_share": float,   # 0 = aggregator pays, 1 = farmer pays
                    "lambda": float,           # action vs result weight
                }
        """
        self.contracts_offered = contracts

    # ================================================================
    # Environment Dynamics — called by the environment, not the agent
    # ================================================================

    def compute_outcomes(self, actions: dict, weather_shock: float):
        """
        Given the farmer's chosen actions and a weather realization,
        compute yield, carbon sequestered, costs, and revenue.

        Args:
            actions: dict with keys matching action_space
                contract_choice (int), crop_choice (int),
                input_choice (int), tillage_choice (int)
            weather_shock: float drawn from N(0,1) by the environment

        Returns:
            dict with yield, carbon, cost, crop_revenue
        """
        self.current_actions = actions

        crop = CROPS[actions["crop_choice"]]
        inp = INPUTS[actions["input_choice"]]
        tillage = TILLAGE_OPTIONS[actions["tillage_choice"]]
        acres = self.farm_size * 100  # each unit = 100 acres

        # ---- Yield Model ----
        # Base yield modified by soil quality, input choice, tillage, and weather
        fertilizer_boost = FERTILIZER_YIELD_BOOST if inp in ["chemical_fertilizer", "manure"] else 0.0
        tillage_effect = -NO_TILL_YIELD_PENALTY if tillage == "no_till" else 0.0
        soil_effect = SOIL_QUALITY_YIELD_FACTOR * (self.soil_quality - 0.5)  # centered
        weather_effect = WEATHER_YIELD_STD * weather_shock

        yield_per_acre = BASE_YIELD_PER_ACRE * (
            1.0 + fertilizer_boost + tillage_effect + soil_effect + weather_effect
        )
        yield_per_acre = max(yield_per_acre, 0.0)  # no negative yield
        total_yield = yield_per_acre * acres

        # ---- Carbon Sequestration Model ----
        carbon_per_acre = BASE_CARBON_PER_ACRE
        carbon_per_acre += SOIL_QUALITY_CARBON_FACTOR * self.soil_quality

        if crop == "cover_crop":
            carbon_per_acre += COVER_CROP_CARBON_BONUS
        if tillage == "no_till":
            carbon_per_acre += NO_TILL_CARBON_BONUS
        if inp == "none":
            carbon_per_acre += LOW_FERT_CARBON_BONUS

        # Weather affects carbon too (drought = less biomass = less sequestration)
        carbon_weather_effect = WEATHER_CARBON_STD * weather_shock
        carbon_per_acre *= (1.0 + carbon_weather_effect)
        carbon_per_acre = max(carbon_per_acre, 0.0)
        total_carbon = carbon_per_acre * acres

        # ---- Cost Model ----
        # Cost = crop production cost + input cost + tillage transition cost
        #        + private disutility from deviating from preferences
        crop_cost = CROP_BASE_COSTS[crop] * (acres / 100.0)  # scale by farm units
        input_cost = INPUT_COSTS[inp] * (acres / 100.0)
        tillage_cost = NO_TILL_TRANSITION_COST * (acres / 100.0) if tillage == "no_till" else 0.0

        # Private cost of deviating from preferred crop and input
        # Higher cost when farmer chooses something far from their preference
        crop_pref_weight = self.crop_preference.get(crop, 0.0)
        input_pref_weight = self.input_preference.get(inp, 0.0)
        # Deviation cost: high when preference weight is low (doing something you don't like)
        preference_deviation_cost = (
            (1.0 - crop_pref_weight) * 50.0 + (1.0 - input_pref_weight) * 30.0
        ) * (acres / 100.0)

        total_cost = crop_cost + input_cost + tillage_cost + preference_deviation_cost

        # ---- Crop Revenue (from selling the harvest) ----
        crop_revenue = total_yield * CROP_PRICES[crop]

        # ---- Store for reward computation ----
        self.current_yield = total_yield
        self.current_carbon = total_carbon
        self.current_cost = total_cost
        self.current_revenue = crop_revenue

        return {
            "yield": total_yield,
            "carbon": total_carbon,
            "cost": total_cost,
            "crop_revenue": crop_revenue,
        }

    def compute_contract_payment(self, contract_params: dict, mrv_cost: float) -> float:
        """
        Compute the total payment from the accepted contract.

        Args:
            contract_params: dict with per_acre_payment, per_ton_payment,
                upfront_fraction, mrv_cost_share, lambda
            mrv_cost: total MRV cost for this farmer (set by environment)

        Returns:
            Total net payment to farmer (can be negative if MRV cost > payment)
        """
        acres = self.farm_size * 100
        lam = contract_params["lambda"]

        # Action component: paid for practice adoption, scaled by lambda
        action_payment = lam * contract_params["per_acre_payment"] * acres

        # Result component: paid for carbon sequestered, scaled by (1-lambda)
        result_payment = (1.0 - lam) * contract_params["per_ton_payment"] * self.current_carbon

        # MRV cost borne by farmer
        farmer_mrv_cost = contract_params["mrv_cost_share"] * mrv_cost

        total_payment = action_payment + result_payment - farmer_mrv_cost
        self.current_payment = total_payment
        return total_payment

    def reward(self) -> float:
        """
        Compute the farmer's reward for the current period.

        Reward = profit - risk_penalty
        Profit = crop_revenue + contract_payment - costs
        Risk penalty = risk_preference * variance proxy

        For single-period: we use the absolute deviation of the result-based
        payment component from its expected value as a variance proxy.
        Risk-averse farmers are penalized more for income that depends on
        stochastic outcomes (weather-dependent carbon payments).
        """
        # ---- Profit ----
        profit = self.current_revenue + self.current_payment - self.current_cost

        # ---- Risk penalty ----
        # The variance proxy: how much of total payment came from the
        # stochastic result-based component (weather-dependent carbon)
        # If no contract accepted, no risk penalty from contract
        risk_penalty = 0.0
        if self.current_contract is not None and self.contracts_offered is not None:
            contract_key = ["action", "result", "hybrid"][self.current_contract]
            contract_params = self.contracts_offered[contract_key]
            lam = contract_params["lambda"]

            # Result-based portion of payment (stochastic part)
            result_portion = (1.0 - lam) * contract_params["per_ton_payment"] * self.current_carbon

            # Risk penalty proportional to result-based exposure
            # Scaled by weather variance parameters so units are consistent
            risk_penalty = self.risk_preference * abs(result_portion) * WEATHER_CARBON_STD

        # ---- No-contract baseline ----
        # If farmer rejected all contracts, they just get crop revenue - cost
        # No risk penalty from contracts (though crop revenue is still stochastic)
        if self.current_contract is None or self.current_contract == 3:
            self.current_payment = 0.0
            profit = self.current_revenue - self.current_cost
            risk_penalty = 0.0

        reward = profit - risk_penalty
        return reward

    # ================================================================
    # State Update
    # ================================================================

    def step(self, actions: dict):
        """
        Record which contract the farmer accepted.
        Called by the environment after the farmer's action.
        """
        contract_choice = actions["contract_choice"]
        if contract_choice < 3:
            self.current_contract = contract_choice
        else:
            self.current_contract = None  # rejected all contracts

    def update_crop_history(self, crop_choice: int):
        """
        After the period ends, update the public crop history.
        Drops the oldest entry, appends the new choice.
        """
        crop = CROPS[crop_choice]
        self.previous_crop_choices.pop(0)
        self.previous_crop_choices.append(crop)

    def reset(self):
        """Reset per-episode state. Type attributes are preserved."""
        self.contracts_offered = None
        self.current_contract = None
        self.current_actions = None
        self.current_payment = 0.0
        self.current_yield = 0.0
        self.current_carbon = 0.0
        self.current_cost = 0.0
        self.current_revenue = 0.0

    # ================================================================
    # Public Information (what the aggregator can see)
    # ================================================================

    def get_public_info(self) -> dict:
        """Return information observable by the aggregator."""
        return {
            "farm_size": self.farm_size,
            "previous_crop_choices": self.previous_crop_choices.copy(),
        }

    def get_noisy_soil_signal(self, noise_std: float = 0.15) -> float:
        """
        Return a noisy observation of soil quality for the aggregator.
        Aggregator sees: true_soil_quality + N(0, noise_std), clipped to [0,1].
        """
        noisy_signal = self.soil_quality + np.random.normal(0, noise_std)
        return float(np.clip(noisy_signal, 0.0, 1.0))

    def get_inferred_crop_preference(self) -> dict:
        """
        Aggregator's estimate of crop preference from observed history.
        Simply the empirical frequency of each crop in the history.
        """
        freq = {c: 0.0 for c in CROPS}
        for c in self.previous_crop_choices:
            freq[c] += 1.0
        total = len(self.previous_crop_choices)
        return {c: freq[c] / total for c in CROPS}

    # ================================================================
    # Helpers
    # ================================================================

    @staticmethod
    def _contract_to_vector(contract_params: dict) -> np.ndarray:
        """Convert a contract dict to a fixed-size numpy vector."""
        if not contract_params:
            return np.zeros(5, dtype=np.float32)
        return np.array([
            contract_params.get("per_acre_payment", 0.0),
            contract_params.get("per_ton_payment", 0.0),
            contract_params.get("upfront_fraction", 0.0),
            contract_params.get("mrv_cost_share", 0.0),
            contract_params.get("lambda", 0.0),
        ], dtype=np.float32)

    def __repr__(self):
        return (
            f"Farmer(id={self.farmer_id}, size={self.farm_size}, "
            f"soil={self.soil_quality:.2f}, risk={self.risk_preference:.2f})"
        )


# ============================================================
# Predefined Farmer Types for Workshop Paper
# ============================================================

def create_workshop_farmers() -> list:
    """
    Create 5 predefined farmer types for the workshop paper.
    These represent archetypal farmers spanning the type space.
    """

    farmers = [
        # Type 1: Large farm, good soil, risk-tolerant, prefers corn, uses chemical fertilizer
        # -> Should prefer result-based contracts (can generate lots of carbon cheaply)
        Farmer(
            farmer_id=0,
            farm_size=8,
            soil_quality=0.85,
            crop_preference={"corn": 0.6, "soybean": 0.25, "wheat": 0.1, "cover_crop": 0.05},
            input_preference={"chemical_fertilizer": 0.75, "manure": 0.2, "none": 0.05},
            risk_preference=0.15,
        ),

        # Type 2: Small farm, poor soil, very risk-averse, prefers wheat, uses manure
        # -> Should prefer action-based contracts (needs guaranteed income)
        Farmer(
            farmer_id=1,
            farm_size=3,
            soil_quality=0.3,
            crop_preference={"corn": 0.1, "soybean": 0.15, "wheat": 0.6, "cover_crop": 0.15},
            input_preference={"chemical_fertilizer": 0.1, "manure": 0.75, "none": 0.15},
            risk_preference=0.85,
        ),

        # Type 3: Medium farm, decent soil, moderate risk, diversified preferences
        # -> Good candidate for hybrid contracts
        Farmer(
            farmer_id=2,
            farm_size=5,
            soil_quality=0.55,
            crop_preference={"corn": 0.3, "soybean": 0.3, "wheat": 0.25, "cover_crop": 0.15},
            input_preference={"chemical_fertilizer": 0.4, "manure": 0.4, "none": 0.2},
            risk_preference=0.45,
        ),

        # Type 4: Large farm, good soil, risk-averse, already conservation-minded
        # -> Interesting case: can generate carbon easily but wants safety
        #    Tests whether hybrid attracts conservation-minded risk-averse farmers
        Farmer(
            farmer_id=3,
            farm_size=7,
            soil_quality=0.75,
            crop_preference={"corn": 0.2, "soybean": 0.25, "wheat": 0.2, "cover_crop": 0.35},
            input_preference={"chemical_fertilizer": 0.1, "manure": 0.4, "none": 0.5},
            risk_preference=0.7,
        ),

        # Type 5: Small farm, medium soil, risk-tolerant, conventional practices
        # -> Low carbon potential but willing to take risks
        #    Tests whether aggregator learns to offer lower payments to small farms
        Farmer(
            farmer_id=4,
            farm_size=2,
            soil_quality=0.45,
            crop_preference={"corn": 0.5, "soybean": 0.3, "wheat": 0.15, "cover_crop": 0.05},
            input_preference={"chemical_fertilizer": 0.65, "manure": 0.25, "none": 0.1},
            risk_preference=0.2,
        ),
    ]

    return farmers


# ============================================================
# Quick Sanity Test
# ============================================================

if __name__ == "__main__":
    farmers = create_workshop_farmers()

    print("=" * 60)
    print("Farmer Types for Workshop Paper")
    print("=" * 60)

    for f in farmers:
        print(f"\n{f}")
        print(f"  Crop pref:  {f.crop_preference}")
        print(f"  Input pref: {f.input_preference}")
        print(f"  Crop history: {f.previous_crop_choices}")
        print(f"  Noisy soil signal: {f.get_noisy_soil_signal():.3f} (true: {f.soil_quality:.2f})")
        print(f"  Inferred crop pref: {f.get_inferred_crop_preference()}")

    # Test action/observation spaces
    print("\n" + "=" * 60)
    print("Spaces")
    print("=" * 60)
    print(f"Action space: {Farmer.action_space()}")
    print(f"Observation space: {Farmer.observation_space()}")

    # Test a single step
    print("\n" + "=" * 60)
    print("Single Step Test (Farmer 0)")
    print("=" * 60)
    f = farmers[0]

    # Simulate aggregator offering contracts
    test_contracts = {
        "action": {
            "per_acre_payment": 15.0,
            "per_ton_payment": 0.0,
            "upfront_fraction": 0.5,
            "mrv_cost_share": 0.0,
            "lambda": 1.0,
        },
        "result": {
            "per_acre_payment": 0.0,
            "per_ton_payment": 30.0,
            "upfront_fraction": 0.0,
            "mrv_cost_share": 0.2,
            "lambda": 0.0,
        },
        "hybrid": {
            "per_acre_payment": 8.0,
            "per_ton_payment": 15.0,
            "upfront_fraction": 0.3,
            "mrv_cost_share": 0.1,
            "lambda": 0.5,
        },
    }
    f.update_contracts_offered(test_contracts)

    # Farmer chooses: accept hybrid, grow corn, no input, no-till
    actions = {
        "contract_choice": 2,  # hybrid
        "crop_choice": 0,      # corn
        "input_choice": 2,     # none
        "tillage_choice": 1,   # no-till
    }
    f.step(actions)

    # Environment draws weather and computes outcomes
    weather = np.random.normal(0, 1)
    outcomes = f.compute_outcomes(actions, weather_shock=weather)
    payment = f.compute_contract_payment(test_contracts["hybrid"], mrv_cost=200.0)
    reward = f.reward()

    print(f"  Weather shock: {weather:.3f}")
    print(f"  Yield: {outcomes['yield']:.1f} bushels")
    print(f"  Carbon: {outcomes['carbon']:.2f} tons CO2e")
    print(f"  Cost: ${outcomes['cost']:.2f}")
    print(f"  Crop revenue: ${outcomes['crop_revenue']:.2f}")
    print(f"  Contract payment: ${payment:.2f}")
    print(f"  Reward: {reward:.2f}")

    # Test observation
    obs = f.get_observation()
    print(f"\n  Observation keys: {list(obs.keys())}")
    print(f"  Farm size obs: {obs['own_farm_size']}")
    print(f"  Hybrid contract obs: {obs['hybrid_contract']}")