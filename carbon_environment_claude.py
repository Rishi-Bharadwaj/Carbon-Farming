"""
Carbon Farming Contract Design - PettingZoo AEC Environment (Minimal Workshop Version)

Step sequence per period:
    1. Aggregator observes farmers, sets contract menu
    2. Each farmer observes contracts + own type, chooses contract + practices
    3. Environment resolves weather, computes outcomes, assigns rewards
"""

import functools
import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
from gymnasium import spaces

from farmer_claude import Farmer, create_farmers, CROPS, INPUTS, NUM_CROPS, NUM_INPUTS
from aggregator_claude import Aggregator, CONTRACT_NAMES, ACTION_DIM


class CarbonFarmingEnv(AECEnv):
    metadata = {"render_modes": ["human"], "name": "carbon_farming_v0",
                "is_parallelizable": False}

    def __init__(self, farmers=None, budget=50000.0, carbon_weight=0.5,
                 num_periods=1, render_mode=None):
        super().__init__()

        self.farmer_objs = farmers or create_farmers()
        self.num_farmers = len(self.farmer_objs)
        self.agg = Aggregator(budget, carbon_weight)
        self.num_periods = num_periods
        self.render_mode = render_mode

        self.possible_agents = ["aggregator"] + [f"farmer_{i}" for i in range(self.num_farmers)]
        self.agents = self.possible_agents[:]

        self._selector = agent_selector(self.agents)
        self.agent_selection = self._selector.reset()

        self.period = 0
        self.weather = 0.0
        self.farmer_actions = {}
        self.acted = set()

        self.rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        if agent == "aggregator":
            return Aggregator.observation_space(self.num_farmers)
        return Farmer.observation_space()

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        if agent == "aggregator":
            return Aggregator.action_space()
        return Farmer.action_space()

    def observe(self, agent):
        if agent == "aggregator":
            return self.agg.get_obs(self.farmer_objs)
        idx = int(agent.split("_")[1])
        return self.farmer_objs[idx].get_obs()

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.agents = self.possible_agents[:]
        self._selector = agent_selector(self.agents)
        self.agent_selection = self._selector.reset()

        self.agg.hard_reset()
        for f in self.farmer_objs:
            f.reset()

        self.period = 0
        self.weather = 0.0
        self.farmer_actions = {}
        self.acted = set()

        self.rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}

    def step(self, action):
        agent = self.agent_selection

        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        self._cumulative_rewards[agent] = 0.0

        if agent == "aggregator":
            menu = self.agg.decode_action(action)
            for f in self.farmer_objs:
                f.set_contracts(menu)
        else:
            idx = int(agent.split("_")[1])
            farmer = self.farmer_objs[idx]
            farmer.act(action)
            self.farmer_actions[idx] = action

        self.acted.add(agent)

        # All agents acted -> resolve period
        if len(self.acted) == len(self.agents):
            self._resolve()

        self.agent_selection = self._selector.next()
        self._accumulate_rewards()

    def _resolve(self):
        self.weather = np.random.normal(0, 1)

        for idx, farmer in enumerate(self.farmer_objs):
            action = self.farmer_actions.get(idx)
            if action is None:
                continue

            farmer.compute(action, self.weather)

            if farmer.accepted is not None:
                ct = CONTRACT_NAMES[farmer.accepted]
                mrv = self.agg.get_mrv_cost(ct, farmer.farm_size)
                farmer.compute_payment(self.agg.menu[ct], mrv)
                self.agg.process_outcome(farmer, ct, farmer.payment, farmer.carbon)
            else:
                self.agg.process_rejection(farmer)

            farmer.update_history(action[1])

        # Rewards
        self.rewards["aggregator"] = self.agg.reward()
        for i, f in enumerate(self.farmer_objs):
            self.rewards[f"farmer_{i}"] = f.reward()

        self.infos["aggregator"] = self.agg.summary()

        self.period += 1
        if self.period >= self.num_periods:
            for a in self.agents:
                self.terminations[a] = True
        else:
            # Next period setup
            self.acted = set()
            self.farmer_actions = {}
            self.agg.reset()
            for f in self.farmer_objs:
                f.contracts = f.accepted = None
                f.payment = f.carbon = f.cost = f.revenue = 0.0
            self._selector = agent_selector(self.agents)
            self.agent_selection = self._selector.reset()

    def render(self):
        if self.render_mode != "human":
            return
        print(f"\n{'='*50}")
        print(f"Period {self.period}/{self.num_periods} | Weather: {self.weather:.3f}")
        print(f"{self.agg}")
        if self.agg.menu:
            for n, p in self.agg.menu.items():
                print(f"  {n}: acre=${p['acre_pay']:.1f} ton=${p['ton_pay']:.1f} "
                      f"up={p['upfront_frac']:.0%} mrv={p['mrv_share']:.0%} λ={p['lambda']}")
        for i, f in enumerate(self.farmer_objs):
            ct = CONTRACT_NAMES[f.accepted] if f.accepted is not None else "none"
            print(f"  {f} -> {ct} | C={f.carbon:.0f}t pay=${f.payment:.0f} R={self.rewards[f'farmer_{i}']:.0f}")
        s = self.agg.summary()
        print(f"  Total: carbon={s['carbon']:.0f}t profit=${s['profit']:.0f} "
              f"enrolled={s['enrolled']}/{self.num_farmers}")


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    env = CarbonFarmingEnv(budget=50000, carbon_weight=0.5,
                           num_periods=1, render_mode="human")
    print(f"Agents: {env.possible_agents}")
    print(f"Agg action dim: {env.action_space('aggregator').shape}")
    print(f"Farmer action: {env.action_space('farmer_0')}")
    print(f"Farmer obs: {env.observation_space('farmer_0')}")

    # ---- Random episode ----
    print("\n--- Random Actions ---")
    env.reset(seed=42)
    for agent in env.agent_iter():
        obs, rew, term, trunc, info = env.last()
        if term or trunc:
            action = None
        elif agent == "aggregator":
            action = env.action_space(agent).sample()
        else:
            action = env.action_space(agent).sample()
        env.step(action)
    env.render()

    # ---- Heuristic episode ----
    print("\n--- Heuristic Actions ---")
    env.reset(seed=123)
    for agent in env.agent_iter():
        obs, rew, term, trunc, info = env.last()
        if term or trunc:
            action = None
        elif agent == "aggregator":
            action = np.array([
                12, 0, 0.5, 0,      # action: $12/acre
                0, 40, 0, 0.2,      # result: $40/ton
                6, 20, 0.3, 0.1,    # hybrid: $6/acre + $20/ton
            ], dtype=np.float32)
        else:
            idx = int(agent.split("_")[1])
            f = env.farmer_objs[idx]
            ct = 0 if f.risk_pref > 0.6 else (1 if f.risk_pref < 0.3 else 2)
            best_crop = CROPS.index(max(f.crop_pref, key=f.crop_pref.get))
            best_input = INPUTS.index(max(f.input_pref, key=f.input_pref.get))
            action = np.array([ct, best_crop, best_input, 1])
        env.step(action)
    env.render()

    # ---- Multi-period ----
    print("\n--- 3-Period Episode ---")
    env3 = CarbonFarmingEnv(budget=50000, carbon_weight=0.5,
                            num_periods=3, render_mode="human")
    env3.reset(seed=99)
    for agent in env3.agent_iter():
        obs, rew, term, trunc, info = env3.last()
        if term or trunc:
            action = None
        elif agent == "aggregator":
            action = np.array([10,0,0.5,0, 0,35,0,0.2, 5,18,0.3,0.1], dtype=np.float32)
        else:
            action = env3.action_space(agent).sample()
        env3.step(action)
    env3.render()
    print(f"\nFinal rewards:")
    for a in env3.possible_agents:
        print(f"  {a}: {env3.rewards[a]:.2f}")