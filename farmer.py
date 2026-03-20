"""
Carbon Farming Contract Design - Farmer (Minimal Workshop Version)

All monetary values are per acre. farm_size is in acres [1, 10].
Total = per_acre_value * farm_size. No unit conversions needed.

Information structure:
    Public:    farm_size, previous_crop_choices
    Noisy:     soil_quality (aggregator sees signal + noise)
    Inferred:  crop_preference (from history)
    Private:   input_preference, risk_preference
"""

import logging
import numpy as np
from gymnasium import spaces

logger = logging.getLogger("carbon_farming.farmer")

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
SOIL_CARBON_FACTOR = 0.3
CROP_CARBON = {"corn": 0.0, "soybean": 0.05, "wheat": 0.05, "cover_crop": 0.3}
INPUT_CARBON = {"chemical_fertilizer": 0.0, "manure": 0.05, "none": 0.15}
TILLAGE_CARBON = {"conventional": 0.0, "no_till": 0.25}
WEATHER_CARBON_STD = 0.1

# Economics — all per acre
SEED_COST_PER_ACRE = {"corn": 120.0, "soybean": 60.0, "wheat": 50.0, "cover_crop": 35.0}
INPUT_COST_PER_ACRE = {"chemical_fertilizer": 80.0, "manure": 50.0, "none": 0.0}
CROP_PRICE_PER_BUSHEL = {"corn": 4.5, "soybean": 12.0, "wheat": 6.5, "cover_crop": 0.0}

# Interest rate: farmer values upfront cash more than deferred payment
INTEREST_RATE = 0.08

# Practices the aggregator can pay for individually
PAYABLE_ACTIONS = {
    "use_cover_crop": lambda crop, inp, till: crop == "cover_crop",
    "use_no_till": lambda crop, inp, till: till == "no_till",
    "use_manure": lambda crop, inp, till: inp == "manure",
    "use_no_fertilizer": lambda crop, inp, till: inp == "none",
}
PAYABLE_ACTION_NAMES = list(PAYABLE_ACTIONS.keys())

# Contract types (Option B: structurally distinct)
CONTRACT_TYPES = ["action", "result", "hybrid"]


def _build_crop_history_from_preferences(crop_pref, length):
    """Build deterministic crop history proportional to preference weights."""
    sorted_crops = sorted(crop_pref.keys(), key=lambda c: crop_pref[c], reverse=True)
    history = []
    remaining = length
    total_weight = sum(crop_pref.values())

    for crop in sorted_crops:
        if remaining <= 0:
            break
        share = crop_pref[crop] / total_weight
        count = max(1, round(share * length)) if share > 0 else 0
        count = min(count, remaining)
        history.extend([crop] * count)
        remaining -= count

    while len(history) < length:
        history.append(sorted_crops[0])

    return history[:length]


class Farmer:
    """Heterogeneous farmer agent with private preferences and public observables."""

    def __init__(self, fid, farm_size, soil_quality,
                 crop_pref, input_pref, risk_pref):
        """
        Args:
            fid: unique farmer identifier
            farm_size: int [1,10], in acres
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

        self.crop_history = _build_crop_history_from_preferences(crop_pref, CROP_HISTORY_LEN)

        self.contracts_offered = None
        self.accepted_contract = None
        self._last_action = None
        self.payment = 0.0
        self.carbon = 0.0
        self.cost = 0.0
        self.crop_revenue = 0.0

        logger.debug(f"[Farmer {self.fid}] Created: size={farm_size}ac, "
                     f"soil={soil_quality:.2f}, risk={risk_pref:.2f}")
        logger.debug(f"[Farmer {self.fid}] Crop pref: {crop_pref}")
        logger.debug(f"[Farmer {self.fid}] Input pref: {input_pref}")
        logger.debug(f"[Farmer {self.fid}] Initial crop history: {self.crop_history}")

    # ================================================================
    # Spaces
    # ================================================================

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
        action_contract_dim = len(PAYABLE_ACTION_NAMES) + 2
        result_contract_dim = 2
        hybrid_contract_dim = len(PAYABLE_ACTION_NAMES) + 3

        return spaces.Dict({
            "own_type": spaces.Box(low=0.0, high=np.inf, shape=(type_dim,), dtype=np.float32),
            "action_contract": spaces.Box(low=0.0, high=np.inf, shape=(action_contract_dim,), dtype=np.float32),
            "result_contract": spaces.Box(low=0.0, high=np.inf, shape=(result_contract_dim,), dtype=np.float32),
            "hybrid_contract": spaces.Box(low=0.0, high=np.inf, shape=(hybrid_contract_dim,), dtype=np.float32),
        })

    # ================================================================
    # Observations
    # ================================================================

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

    # ================================================================
    # Contract interaction
    # ================================================================

    def receive_contracts(self, contracts):
        """Store the contract menu offered by the aggregator."""
        self.contracts_offered = contracts
        logger.debug(f"[Farmer {self.fid}] Received contracts")

    def choose_contract(self, action):
        """Record which contract was accepted from the action dict."""
        choice = int(action["contract_choice"])
        self.accepted_contract = choice if choice < len(CONTRACT_TYPES) else None
        self._last_action = action

        ct_name = CONTRACT_TYPES[self.accepted_contract] if self.accepted_contract is not None else "REJECTED"
        crop = CROPS[action["crop_choice"]]
        inp = INPUTS[action["input_choice"]]
        till = TILLAGE[action["tillage_choice"]]
        logger.info(f"[Farmer {self.fid}] Action: contract={ct_name}, "
                    f"crop={crop}, input={inp}, tillage={till}")

    # ================================================================
    # Outcome computation
    # ================================================================

    def compute_yield_and_carbon(self, action, weather_shock):
        """Compute yield, carbon, costs, revenue from actions and weather."""
        crop = CROPS[action["crop_choice"]]
        inp = INPUTS[action["input_choice"]]
        till = TILLAGE[action["tillage_choice"]]

        # ---- Yield per acre ----
        fert_boost = FERT_YIELD_BOOST[inp]
        soil_effect = SOIL_YIELD_FACTOR * (self.soil_quality - 0.5)
        weather_effect = WEATHER_YIELD_STD * weather_shock
        yield_per_acre = BASE_YIELD * (1.0 + fert_boost + soil_effect + weather_effect)
        yield_per_acre = max(yield_per_acre, 0.0)
        price_per_bushel = CROP_PRICE_PER_BUSHEL[crop]
        revenue_per_acre = yield_per_acre * price_per_bushel
        self.crop_revenue = revenue_per_acre * self.farm_size

        logger.info(f"[Farmer {self.fid}] YIELD: base={BASE_YIELD:.1f} × "
                    f"(1 + fert={fert_boost:.2f} + soil={soil_effect:.2f} + weather={weather_effect:.3f}) "
                    f"= {yield_per_acre:.2f} bu/ac")
        logger.info(f"[Farmer {self.fid}] REVENUE: {yield_per_acre:.2f} bu/ac × "
                    f"${price_per_bushel:.2f}/bu × {self.farm_size} ac "
                    f"= ${self.crop_revenue:.2f}")

        # ---- Carbon per acre ----
        crop_c = CROP_CARBON[crop]
        input_c = INPUT_CARBON[inp]
        tillage_c = TILLAGE_CARBON[till]
        carbon_base = BASE_CARBON + SOIL_CARBON_FACTOR * self.soil_quality + crop_c + input_c + tillage_c
        carbon_weather = 1.0 + WEATHER_CARBON_STD * weather_shock
        carbon_per_acre = carbon_base * carbon_weather
        carbon_per_acre = max(carbon_per_acre, 0.0)
        self.carbon = carbon_per_acre * self.farm_size

        logger.info(f"[Farmer {self.fid}] CARBON: (base={BASE_CARBON:.2f} + "
                    f"soil={SOIL_CARBON_FACTOR * self.soil_quality:.2f} + "
                    f"crop={crop_c:.2f} + input={input_c:.2f} + till={tillage_c:.2f}) "
                    f"× weather_mult={carbon_weather:.3f} "
                    f"= {carbon_per_acre:.4f} tCO2e/ac × {self.farm_size} ac "
                    f"= {self.carbon:.4f} tCO2e")

        # ---- Cost per acre ----
        seed_cost = SEED_COST_PER_ACRE[crop]
        input_cost = INPUT_COST_PER_ACRE[inp]
        crop_pref_penalty = (1.0 - self.crop_pref.get(crop, 0)) * 50.0
        input_pref_penalty = (1.0 - self.input_pref.get(inp, 0)) * 30.0
        pref_penalty = crop_pref_penalty + input_pref_penalty
        cost_per_acre = seed_cost + input_cost + pref_penalty
        self.cost = cost_per_acre * self.farm_size

        logger.info(f"[Farmer {self.fid}] COST: (seed=${seed_cost:.2f} + input=${input_cost:.2f} + "
                    f"crop_pref_pen=${crop_pref_penalty:.2f} + input_pref_pen=${input_pref_penalty:.2f}) "
                    f"= ${cost_per_acre:.2f}/ac × {self.farm_size} ac "
                    f"= ${self.cost:.2f}")

    def compute_contract_payment(self, mrv_cost):
        """Compute net payment from the accepted contract with interest on upfront."""
        ct_type = CONTRACT_TYPES[self.accepted_contract]
        params = self.contracts_offered[ct_type]
        crop = CROPS[self._last_action["crop_choice"]]
        inp = INPUTS[self._last_action["input_choice"]]
        till = TILLAGE[self._last_action["tillage_choice"]]

        logger.info(f"[Farmer {self.fid}] PAYMENT ({ct_type}):")

        if ct_type == "action":
            action_pay_per_acre = self._sum_action_payments_per_acre(
                params["per_action_payments"], crop, inp, till)
            upfront = params["upfront_fraction"] * action_pay_per_acre * self.farm_size
            upfront_with_interest = upfront * (1.0 + INTEREST_RATE)
            deferred = (1.0 - params["upfront_fraction"]) * action_pay_per_acre * self.farm_size
            farmer_mrv = params["mrv_share"] * mrv_cost
            total_pay = upfront_with_interest + deferred - farmer_mrv

            logger.info(f"  action_pay/ac=${action_pay_per_acre:.2f}, "
                        f"upfront=${upfront:.2f} × (1+{INTEREST_RATE})=${upfront_with_interest:.2f}, "
                        f"deferred=${deferred:.2f}, "
                        f"farmer_mrv=${farmer_mrv:.2f} (mrv_total=${mrv_cost:.2f} × share={params['mrv_share']:.2f})")
            logger.info(f"  TOTAL: ${upfront_with_interest:.2f} + ${deferred:.2f} - ${farmer_mrv:.2f} = ${total_pay:.2f}")

        elif ct_type == "result":
            result_pay = params["per_ton_payment"] * self.carbon
            farmer_mrv = params["mrv_share"] * mrv_cost
            total_pay = result_pay - farmer_mrv

            logger.info(f"  result_pay=${params['per_ton_payment']:.2f}/t × {self.carbon:.4f}t = ${result_pay:.2f}, "
                        f"farmer_mrv=${farmer_mrv:.2f}")
            logger.info(f"  TOTAL: ${result_pay:.2f} - ${farmer_mrv:.2f} = ${total_pay:.2f}")

        elif ct_type == "hybrid":
            action_pay_per_acre = self._sum_action_payments_per_acre(
                params["per_action_payments"], crop, inp, till)
            upfront = params["upfront_fraction"] * action_pay_per_acre * self.farm_size
            upfront_with_interest = upfront * (1.0 + INTEREST_RATE)
            deferred_action = (1.0 - params["upfront_fraction"]) * action_pay_per_acre * self.farm_size
            result_pay = params["per_ton_payment"] * self.carbon
            farmer_mrv = params["mrv_share"] * mrv_cost
            total_pay = upfront_with_interest + deferred_action + result_pay - farmer_mrv

            logger.info(f"  action_pay/ac=${action_pay_per_acre:.2f}, "
                        f"upfront=${upfront:.2f} × (1+{INTEREST_RATE})=${upfront_with_interest:.2f}, "
                        f"deferred_action=${deferred_action:.2f}, "
                        f"result_pay=${params['per_ton_payment']:.2f}/t × {self.carbon:.4f}t = ${result_pay:.2f}, "
                        f"farmer_mrv=${farmer_mrv:.2f}")
            logger.info(f"  TOTAL: ${upfront_with_interest:.2f} + ${deferred_action:.2f} + "
                        f"${result_pay:.2f} - ${farmer_mrv:.2f} = ${total_pay:.2f}")

        self.payment = total_pay

    @staticmethod
    def _sum_action_payments_per_acre(per_action_payments, crop, inp, till):
        """Sum per-acre payments for all qualifying practices."""
        total = 0.0
        qualified = []
        for name in PAYABLE_ACTION_NAMES:
            if PAYABLE_ACTIONS[name](crop, inp, till):
                total += per_action_payments[name]
                qualified.append(f"{name}=${per_action_payments[name]:.2f}")
        if qualified:
            logger.debug(f"    Qualifying actions: {', '.join(qualified)} = ${total:.2f}/ac")
        else:
            logger.debug(f"    No qualifying actions")
        return total

    # ================================================================
    # Reward
    # ================================================================

    def compute_reward(self):
        """Reward = profit - risk_penalty. Risk scales with stochastic income exposure."""
        if self.accepted_contract is None:
            reward = self.crop_revenue - self.cost
            logger.info(f"[Farmer {self.fid}] REWARD (no contract): "
                        f"revenue=${self.crop_revenue:.2f} - cost=${self.cost:.2f} = ${reward:.2f}")
            return reward

        profit = self.crop_revenue + self.payment - self.cost

        ct_type = CONTRACT_TYPES[self.accepted_contract]
        params = self.contracts_offered[ct_type]

        if ct_type == "action":
            result_exposure = 0.0
        else:
            result_exposure = params["per_ton_payment"] * self.carbon

        risk_penalty = self.risk_pref * abs(result_exposure) * WEATHER_CARBON_STD
        reward = profit - risk_penalty

        logger.info(f"[Farmer {self.fid}] REWARD ({ct_type}): "
                    f"profit=(${self.crop_revenue:.2f} + ${self.payment:.2f} - ${self.cost:.2f}) "
                    f"= ${profit:.2f}, "
                    f"risk_pen=(risk={self.risk_pref:.2f} × exposure=${result_exposure:.2f} "
                    f"× σ={WEATHER_CARBON_STD}) = ${risk_penalty:.2f}, "
                    f"reward = ${reward:.2f}")
        return reward

    # ================================================================
    # State management
    # ================================================================

    def update_crop_history(self, crop_idx):
        """Append latest crop choice and drop oldest entry."""
        old = self.crop_history[0]
        new = CROPS[crop_idx]
        self.crop_history.pop(0)
        self.crop_history.append(new)
        logger.debug(f"[Farmer {self.fid}] Crop history: dropped '{old}', added '{new}' -> {self.crop_history}")

    def reset_period(self):
        """Reset per-period state. Type and history are preserved."""
        self.contracts_offered = None
        self.accepted_contract = None
        self._last_action = None
        self.payment = self.carbon = self.cost = self.crop_revenue = 0.0
        logger.debug(f"[Farmer {self.fid}] Period reset")

    # ================================================================
    # Public / noisy info for aggregator
    # ================================================================

    def get_public_farm_size(self):
        """Return farm size in acres (fully public)."""
        return self.farm_size

    def get_public_crop_history(self):
        """Return copy of crop history (fully public)."""
        return self.crop_history[:]

    def get_noisy_soil_quality(self, noise_std=0.15):
        """Return soil quality with Gaussian noise, clipped to [0,1]."""
        noise = np.random.normal(0, noise_std)
        noisy = float(np.clip(self.soil_quality + noise, 0, 1))
        logger.debug(f"[Farmer {self.fid}] Soil signal: true={self.soil_quality:.2f} + "
                     f"noise={noise:.3f} = {noisy:.3f}")
        return noisy

    def get_inferred_crop_preference(self):
        """Empirical frequency of each crop in history (aggregator's estimate)."""
        freq = {c: 0 for c in CROPS}
        for c in self.crop_history:
            freq[c] += 1
        n = len(self.crop_history)
        return [freq[c] / n for c in CROPS]

    # ================================================================
    # Serialization
    # ================================================================

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
        return (f"Farmer(id={self.fid}, size={self.farm_size}ac, "
                f"soil={self.soil_quality:.2f}, risk={self.risk_pref:.2f})")


# ============================================================
# Workshop Farmer Types
# ============================================================

def create_farmers():
    """5 archetypal farmers spanning the type space. Farm sizes in acres."""
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