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

CROPS = ["corn", "soybean", "wheat", "cover_crop"]
INPUTS = ["chemical_fertilizer", "manure", "none"]
NUM_CROPS = len(CROPS)
NUM_INPUTS = len(INPUTS)
CROP_HISTORY_LEN = 5

# Yield parameters (bushels per acre)
BASE_YIELD = 150.0
FERT_YIELD_BOOST = {"chemical_fertilizer": 0.4, "manure": 0.2, "none": 0.0}
SOIL_YIELD_FACTOR = 0.5
WEATHER_YIELD_STD = 0.15

# Carbon parameters (tons CO2e per acre)
BASE_CARBON = 0.2
SOIL_CARBON_FACTOR = 0.3
CROP_CARBON = {"corn": 0.0, "soybean": 0.05, "wheat": 0.05, "cover_crop": 0.3}
INPUT_CARBON = {"chemical_fertilizer": 0.0, "manure": 0.05, "none": 0.15}
NOTILL_CARBON = 0.25
WEATHER_CARBON_STD = 0.1

# Economics (per farm-unit = 100 acres)
CROP_COSTS = {"corn": 300, "soybean": 200, "wheat": 180, "cover_crop": 120}
INPUT_COSTS = {"chemical_fertilizer": 80, "manure": 50, "none": 0}
CROP_PRICES = {"corn": 4.5, "soybean": 12.0, "wheat": 6.5, "cover_crop": 0.0}

# Observation dimension: type(10) + contracts(12) = 22
OBS_DIM = 10 + 12


class Farmer:
    def __init__(self, fid, farm_size, soil_quality,
                 crop_pref, input_pref, risk_pref):
        self.fid = fid
        self.farm_size = farm_size          # [1,10], unit=100 acres
        self.soil_quality = soil_quality    # [0,1]
        self.crop_pref = crop_pref          # dict over CROPS
        self.input_pref = input_pref        # dict over INPUTS
        self.risk_pref = risk_pref          # [0,1]

        probs = np.array([crop_pref[c] for c in CROPS])
        self.crop_history = list(np.random.choice(
            CROPS, size=CROP_HISTORY_LEN, p=probs / probs.sum()))

        self.contracts = None       # offered menu
        self.accepted = None        # 0/1/2 or None
        self.payment = 0.0
        self.carbon = 0.0
        self.cost = 0.0
        self.revenue = 0.0

    @staticmethod
    def action_space():
        return spaces.MultiDiscrete([4, NUM_CROPS, NUM_INPUTS, 2]) #(contract, crops, inputs, till)

    @staticmethod
    def observation_space():
        return spaces.Box(low=-1.0, high=1e4, shape=(OBS_DIM,), dtype=np.float32)

    def get_obs(self):
        o = [self.farm_size / 10.0, self.soil_quality, self.risk_pref]
        o += [self.crop_pref[c] for c in CROPS]
        o += [self.input_pref[i] for i in INPUTS]
        if self.contracts:
            for k in ["action", "result", "hybrid"]:
                c = self.contracts[k]
                o += [c["acre_pay"] / 50.0, c["ton_pay"] / 60.0,
                      c["upfront_frac"], c["mrv_share"]]
        else:
            o += [0.0] * 12
        return np.array(o, dtype=np.float32)

    def set_contracts(self, contracts):
        self.contracts = contracts

    def act(self, action):
        self.accepted = int(action[0]) if action[0] < 3 else None

    def compute(self, action, weather):
        crop = CROPS[action[1]]
        inp = INPUTS[action[2]]
        notill = action[3] == 1
        acres = self.farm_size * 100

        # Yield
        y = BASE_YIELD * (1 + FERT_YIELD_BOOST[inp]
                          + SOIL_YIELD_FACTOR * (self.soil_quality - 0.5)
                          + WEATHER_YIELD_STD * weather)
        self.revenue = max(y, 0) * acres * CROP_PRICES[crop]

        # Carbon
        c = (BASE_CARBON + SOIL_CARBON_FACTOR * self.soil_quality
             + CROP_CARBON[crop] + INPUT_CARBON[inp])
        if notill:
            c += NOTILL_CARBON
        c *= (1 + WEATHER_CARBON_STD * weather)
        self.carbon = max(c, 0) * acres

        # Cost
        pref_pen = ((1 - self.crop_pref.get(crop, 0)) * 50
                    + (1 - self.input_pref.get(inp, 0)) * 30)
        self.cost = (CROP_COSTS[crop] + INPUT_COSTS[inp] + pref_pen) * self.farm_size

    def compute_payment(self, params, mrv_cost):
        acres = self.farm_size * 100
        lam = params["lambda"]
        self.payment = (lam * params["acre_pay"] * acres
                        + (1 - lam) * params["ton_pay"] * self.carbon
                        - params["mrv_share"] * mrv_cost)

    def reward(self):
        if self.accepted is None:
            return self.revenue - self.cost
        profit = self.revenue + self.payment - self.cost
        ct = ["action", "result", "hybrid"][self.accepted]
        lam = self.contracts[ct]["lambda"]
        exposure = (1 - lam) * self.contracts[ct]["ton_pay"] * self.carbon
        return profit - self.risk_pref * abs(exposure) * WEATHER_CARBON_STD

    def update_history(self, crop_idx):
        self.crop_history.pop(0)
        self.crop_history.append(CROPS[crop_idx])

    def reset(self):
        self.contracts = self.accepted = None
        self.payment = self.carbon = self.cost = self.revenue = 0.0

    # Public/noisy info for aggregator
    def public_info(self):
        return self.farm_size, self.crop_history[:]

    def noisy_soil(self, std=0.15):
        return float(np.clip(self.soil_quality + np.random.normal(0, std), 0, 1))

    def inferred_crop_pref(self):
        freq = {c: 0 for c in CROPS}
        for c in self.crop_history:
            freq[c] += 1
        n = len(self.crop_history)
        return [freq[c] / n for c in CROPS]

    def __repr__(self):
        return f"F{self.fid}(sz={self.farm_size},soil={self.soil_quality:.2f},risk={self.risk_pref:.2f})"


def create_farmers():
    return [
        Farmer(0, 8, 0.85,
               {"corn":.6,"soybean":.25,"wheat":.1,"cover_crop":.05},
               {"chemical_fertilizer":.7,"manure":.2,"none":.1}, 0.15),
        Farmer(1, 3, 0.30,
               {"corn":.1,"soybean":.15,"wheat":.6,"cover_crop":.15},
               {"chemical_fertilizer":.1,"manure":.7,"none":.2}, 0.85),
        Farmer(2, 5, 0.55,
               {"corn":.3,"soybean":.3,"wheat":.25,"cover_crop":.15},
               {"chemical_fertilizer":.3,"manure":.4,"none":.3}, 0.45),
        Farmer(3, 7, 0.75,
               {"corn":.2,"soybean":.25,"wheat":.2,"cover_crop":.35},
               {"chemical_fertilizer":.1,"manure":.4,"none":.5}, 0.70),
        Farmer(4, 2, 0.45,
               {"corn":.5,"soybean":.3,"wheat":.15,"cover_crop":.05},
               {"chemical_fertilizer":.6,"manure":.25,"none":.15}, 0.20),
    ]