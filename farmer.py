"""
Carbon Farming Contract Design - Farmer (Minimal Workshop Version)

Information structure:
    Public:    farm_size, previous_crop_choices
    Noisy:     soil_quality (aggregator sees signal + noise)
    Inferred:  crop_preference (from history)
    Private:   input_preference, risk_preference
"""

import numpy as np
from gymnasium import spaces

# ============================================================
# Domain constants — change these lists to add/remove options
# ============================================================

CROPS = ["corn", "soybean", "wheat", "cover_crop"]
INPUTS = ["chemical_fertilizer", "manure", "none"]
TILLAGE = ["conventional", "no_till"]
CROP_HISTORY_LEN = 5

# Yield (bushels per acre)
BASE_YIELD = 150.0
FERT_YIELD_BOOST = {"chemical_fertilizer": 0.4, "manure": 0.2, "none": 0.0}
SOIL_YIELD_FACTOR = 0.5
WEATHER_YIELD_STD = 0.15

# Carbon (tons CO2e per acre)
BASE_CARBON = 0.2
SOIL_CARBON_FACTOR = 0.2
CROP_CARBON = {"corn": 0.0, "soybean": 0.05, "wheat": 0.05, "cover_crop": 0.3}
INPUT_CARBON = {"chemical_fertilizer": 0.0, "manure": 0.05, "none": 0.15}
TILLAGE_CARBON = {"conventional": 0.0, "no_till": 0.25}
WEATHER_CARBON_STD = 0.1

# Economics (per farm-unit = 100 acres)
CROP_COSTS = {"corn": 300, "soybean": 200, "wheat": 180, "cover_crop": 120}
INPUT_COSTS = {"chemical_fertilizer": 80, "manure": 50, "none": 0}
CROP_PRICES = {"corn": 4.5, "soybean": 12.0, "wheat": 6.5, "cover_crop": 0.0}

# Practices the aggregator can pay for individually in action/hybrid contracts
PAYABLE_ACTIONS = {
    "use_cover_crop": lambda crop, inp, till: crop == "cover_crop",
    "use_no_till": lambda crop, inp, till: till == "no_till",
    "use_manure": lambda crop, inp, till: inp == "manure",
    "use_no_fertilizer": lambda crop, inp, till: inp == "none",
}
PAYABLE_ACTION_NAMES = list(PAYABLE_ACTIONS.keys())

# Contract types (Option B: structurally distinct)
CONTRACT_TYPES = ["action", "result", "hybrid"]


class Farmer:
    """Heterogeneous farmer agent with private preferences and public observables."""

    def __init__(self, fid, farm_size, soil_quality,
                 crop_pref, input_pref, risk_pref):
        """
        Args:
            fid: unique farmer identifier
            farm_size: int [1,10], each unit = 100 acres
            soil_quality: float [0,1], true soil carbon potential
            crop_pref: dict over CROPS, weights summing to ~1
            input_pref: dict over INPUTS, weights summing to ~1
            risk_pref: float [0,1], 0=risk neutral, 1=very risk averse
        """
        self.fid = fid
        self.farm_size = farm_size
        self.soil_quality = soil_quality
        self.crop_pref = crop_pref
        self.input_pref = input_pref
        self.risk_pref = risk_pref

        probs = np.array([crop_pref[c] for c in CROPS])
        self.crop_history = list(np.random.choice(
            CROPS, size=CROP_HISTORY_LEN, p=probs / probs.sum()))

        self.contracts_offered = None
        self.accepted_contract = None
        self._last_action = None
        self.payment = 0.0
        self.carbon = 0.0
        self.cost = 0.0
        self.crop_revenue = 0.0


    @staticmethod
    def build_action_space():
        """Contract choice + crop + input + tillage, all independent."""
        return spaces.Dict({
            "contract_choice": spaces.Discrete(len(CONTRACT_TYPES) + 1),
            "crop_choice": spaces.Discrete(len(CROPS)),
            "input_choice": spaces.Discrete(len(INPUTS)),
            "tillage_choice": spaces.Discrete(len(TILLAGE)),
        })

    @staticmethod
    def build_observation_space():
        """Own type vector + three independent contract parameter vectors."""
        type_dim = 1 + 1 + 1 + len(CROPS) + len(INPUTS)
        action_contract_dim = len(PAYABLE_ACTION_NAMES) + 2   # per-action pays, upfront_frac, mrv_share
        result_contract_dim = 2                                 # per_ton_payment, mrv_share
        hybrid_contract_dim = len(PAYABLE_ACTION_NAMES) + 3    # per-action pays, per_ton, upfront_frac, mrv_share

        return spaces.Dict({
            "own_type": spaces.Box(low=0.0, high=np.inf, shape=(type_dim,), dtype=np.float32),
            "action_contract": spaces.Box(low=0.0, high=np.inf, shape=(action_contract_dim,), dtype=np.float32),
            "result_contract": spaces.Box(low=0.0, high=np.inf, shape=(result_contract_dim,), dtype=np.float32),
            "hybrid_contract": spaces.Box(low=0.0, high=np.inf, shape=(hybrid_contract_dim,), dtype=np.float32),
        })


    def get_observation(self):
        """Build observation dict: own type + offered contracts."""
        type_vec = ([self.farm_size, self.soil_quality, self.risk_pref]
                    + [self.crop_pref[c] for c in CROPS]
                    + [self.input_pref[i] for i in INPUTS])

        if self.contracts_offered is not None:
            action_vec = self._action_contract_to_vec(self.contracts_offered["action"])
            result_vec = self._result_contract_to_vec(self.contracts_offered["result"])
            hybrid_vec = self._hybrid_contract_to_vec(self.contracts_offered["hybrid"])
        else:
            action_vec = np.zeros(len(PAYABLE_ACTION_NAMES) + 2, dtype=np.float32)
            result_vec = np.zeros(2, dtype=np.float32)
            hybrid_vec = np.zeros(len(PAYABLE_ACTION_NAMES) + 3, dtype=np.float32)

        return {
            "own_type": np.array(type_vec, dtype=np.float32),
            "action_contract": np.array(action_vec, dtype=np.float32),
            "result_contract": np.array(result_vec, dtype=np.float32),
            "hybrid_contract": np.array(hybrid_vec, dtype=np.float32),
        }


    def receive_contracts(self, contracts):
        """Store the contract menu offered by the aggregator."""
        self.contracts_offered = contracts

    def choose_contract(self, action):
        """Record which contract was accepted from the action dict."""
        choice = int(action["contract_choice"])
        self.accepted_contract = choice if choice < len(CONTRACT_TYPES) else None
        self._last_action = action


    def compute_yield_and_carbon(self, action, weather_shock):
        """Compute crop yield, carbon sequestered, costs, revenue from actions and weather."""
        crop = CROPS[action["crop_choice"]]
        inp = INPUTS[action["input_choice"]]
        till = TILLAGE[action["tillage_choice"]]
        acres = self.farm_size * 100

        # Yield
        yield_per_acre = BASE_YIELD * (
            1.0
            + FERT_YIELD_BOOST[inp]
            + SOIL_YIELD_FACTOR * (self.soil_quality - 0.5)
            + WEATHER_YIELD_STD * weather_shock
        )
        yield_per_acre = max(yield_per_acre, 0.0)
        self.crop_revenue = yield_per_acre * acres * CROP_PRICES[crop]

        # Carbon
        carbon_per_acre = (BASE_CARBON
                           + SOIL_CARBON_FACTOR * self.soil_quality
                           + CROP_CARBON[crop]
                           + INPUT_CARBON[inp]
                           + TILLAGE_CARBON[till])
        carbon_per_acre *= (1.0 + WEATHER_CARBON_STD * weather_shock)
        carbon_per_acre = max(carbon_per_acre, 0.0)
        self.carbon = carbon_per_acre * acres

        # Cost: production + preference deviation
        pref_penalty = ((1.0 - self.crop_pref.get(crop, 0)) * 50.0
                        + (1.0 - self.input_pref.get(inp, 0)) * 30.0)
        self.cost = (CROP_COSTS[crop] + INPUT_COSTS[inp] + pref_penalty) * self.farm_size

    def compute_contract_payment(self, mrv_cost):
        """Compute net payment from the accepted contract given MRV cost."""
        ct_type = CONTRACT_TYPES[self.accepted_contract]
        params = self.contracts_offered[ct_type]
        acres = self.farm_size * 100
        crop = CROPS[self._last_action["crop_choice"]]
        inp = INPUTS[self._last_action["input_choice"]]
        till = TILLAGE[self._last_action["tillage_choice"]]

        if ct_type == "action":
            pay = self._sum_action_payments(params["per_action_payments"], crop, inp, till, acres)
            pay -= params["mrv_share"] * mrv_cost

        elif ct_type == "result":
            pay = params["per_ton_payment"] * self.carbon
            pay -= params["mrv_share"] * mrv_cost

        elif ct_type == "hybrid":
            action_pay = self._sum_action_payments(params["per_action_payments"], crop, inp, till, acres)
            result_pay = params["per_ton_payment"] * self.carbon
            pay = action_pay + result_pay - params["mrv_share"] * mrv_cost

        self.payment = pay

    @staticmethod
    def _sum_action_payments(per_action_payments, crop, inp, till, acres):
        """Sum per-action payments for all qualifying practices."""
        total = 0.0
        for name in PAYABLE_ACTION_NAMES:
            if PAYABLE_ACTIONS[name](crop, inp, till):
                total += per_action_payments[name] * acres
        return total


    def compute_reward(self):
        """Reward = profit - risk_penalty. Risk penalty scales with stochastic income exposure."""
        if self.accepted_contract is None:
            return self.crop_revenue - self.cost

        profit = self.crop_revenue + self.payment - self.cost

        ct_type = CONTRACT_TYPES[self.accepted_contract]
        params = self.contracts_offered[ct_type]

        # Result-based exposure (stochastic component)
        if ct_type == "action":
            result_exposure = 0.0
        else:
            result_exposure = params["per_ton_payment"] * self.carbon

        risk_penalty = self.risk_pref * abs(result_exposure) * WEATHER_CARBON_STD
        return profit - risk_penalty



    def update_crop_history(self, crop_idx):
        """Append latest crop choice and drop oldest entry."""
        self.crop_history.pop(0)
        self.crop_history.append(CROPS[crop_idx])

    def reset_period(self):
        """Reset per-period state. Type and history are preserved."""
        self.contracts_offered = None
        self.accepted_contract = None
        self._last_action = None
        self.payment = self.carbon = self.cost = self.crop_revenue = 0.0


    def get_public_farm_size(self):
        """Return farm size (fully public)."""
        return self.farm_size

    def get_public_crop_history(self):
        """Return copy of crop history (fully public)."""
        return self.crop_history[:]

    def get_noisy_soil_quality(self, noise_std=0.15):
        """Return soil quality with Gaussian noise, clipped to [0,1]."""
        return float(np.clip(self.soil_quality + np.random.normal(0, noise_std), 0, 1))

    def get_inferred_crop_preference(self):
        """Empirical frequency of each crop in history (aggregator's estimate)."""
        freq = {c: 0 for c in CROPS}
        for c in self.crop_history:
            freq[c] += 1
        n = len(self.crop_history)
        return [freq[c] / n for c in CROPS]


    @staticmethod
    def _action_contract_to_vec(params):
        """Flatten action contract: [per_action_pays..., upfront_frac, mrv_share]."""
        vec = [params["per_action_payments"][a] for a in PAYABLE_ACTION_NAMES]
        vec += [params["upfront_fraction"], params["mrv_share"]]
        return vec

    @staticmethod
    def _result_contract_to_vec(params):
        """Flatten result contract: [per_ton_payment, mrv_share]."""
        return [params["per_ton_payment"], params["mrv_share"]]

    @staticmethod
    def _hybrid_contract_to_vec(params):
        """Flatten hybrid contract: [per_action_pays..., per_ton, upfront_frac, mrv_share]."""
        vec = [params["per_action_payments"][a] for a in PAYABLE_ACTION_NAMES]
        vec += [params["per_ton_payment"], params["upfront_fraction"], params["mrv_share"]]
        return vec

    def __repr__(self):
        return (f"Farmer(id={self.fid}, size={self.farm_size}, "
                f"soil={self.soil_quality:.2f}, risk={self.risk_pref:.2f})")


# ============================================================
# Workshop Farmer Types
# ============================================================

def create_farmers():
    """5 archetypal farmers spanning the type space."""
    return [
        Farmer(0, 8, 0.85,
               {"corn": .6, "soybean": .25, "wheat": .1, "cover_crop": .05},
               {"chemical_fertilizer": .7, "manure": .2, "none": .1}, 0.15),
        Farmer(1, 3, 0.30,
               {"corn": .1, "soybean": .15, "wheat": .6, "cover_crop": .15},
               {"chemical_fertilizer": .1, "manure": .7, "none": .2}, 0.85),
        Farmer(2, 5, 0.55,
               {"corn": .3, "soybean": .3, "wheat": .25, "cover_crop": .15},
               {"chemical_fertilizer": .3, "manure": .4, "none": .3}, 0.45),
        Farmer(3, 7, 0.75,
               {"corn": .2, "soybean": .25, "wheat": .2, "cover_crop": .35},
               {"chemical_fertilizer": .1, "manure": .4, "none": .5}, 0.70),
        Farmer(4, 2, 0.45,
               {"corn": .5, "soybean": .3, "wheat": .15, "cover_crop": .05},
               {"chemical_fertilizer": .6, "manure": .25, "none": .15}, 0.20),
    ]