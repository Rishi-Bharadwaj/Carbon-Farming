"""
Carbon Farming Contract Design - PettingZoo AEC Environment (Minimal Workshop Version)

Step sequence per period:
    1. Aggregator observes farmers, posts contract menu
    2. Each farmer observes contracts + own type, chooses contract + practices
    3. Environment resolves weather, computes outcomes, assigns rewards
"""

import functools
import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from gymnasium import spaces

from farmer import Farmer, create_farmers, CROPS, INPUTS, TILLAGE, CONTRACT_TYPES
from aggregator import Aggregator


class CarbonFarmingEnv(AECEnv):
    """PettingZoo AEC environment for the carbon farming contract design game."""

    metadata = {"render_modes": ["human"], "name": "carbon_farming_v0",
                "is_parallelizable": False}

    def __init__(self, farmers=None, budget=50000.0, carbon_weight=0.5,
                 num_periods=1, render_mode=None):
        """
        Args:
            farmers: list of Farmer objects (default: 5 workshop types)
            budget: aggregator budget per period
            carbon_weight: aggregator objective weight on carbon vs profit
            num_periods: periods per episode (1 for workshop)
            render_mode: 'human' for console output
        """
        super().__init__()

        self.farmer_objs = farmers or create_farmers()
        self.num_farmers = len(self.farmer_objs)
        self.agg = Aggregator(budget, carbon_weight)
        self.num_periods = num_periods
        self.render_mode = render_mode

        # Agent names: aggregator moves first, then farmers
        self.possible_agents = ["aggregator"] + [
            f"farmer_{i}" for i in range(self.num_farmers)]
        self.agents = self.possible_agents[:]

        self._selector = agent_selector(self.agents)
        self.agent_selection = self._selector.reset()

        # Period state
        self.period = 0
        self.weather = 0.0
        self.farmer_actions = {}
        self.acted = set()

        # PettingZoo required
        self.rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}

    # ================================================================
    # Spaces
    # ================================================================

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        """Return observation space for the given agent."""
        if agent == "aggregator":
            return Aggregator.build_observation_space(self.num_farmers)
        return Farmer.build_observation_space()

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        """Return action space for the given agent."""
        if agent == "aggregator":
            return Aggregator.build_action_space()
        return Farmer.build_action_space()

    # ================================================================
    # Observations
    # ================================================================

    def observe(self, agent):
        """Return current observation for the given agent."""
        if agent == "aggregator":
            return self.agg.get_observation(self.farmer_objs)
        idx = int(agent.split("_")[1])
        return self.farmer_objs[idx].get_observation()

    # ================================================================
    # Core AEC loop
    # ================================================================

    def reset(self, seed=None, options=None):
        """Reset environment for a new episode."""
        if seed is not None:
            np.random.seed(seed)

        self.agents = self.possible_agents[:]
        self._selector = agent_selector(self.agents)
        self.agent_selection = self._selector.reset()

        self.agg.reset_episode()
        for f in self.farmer_objs:
            f.reset_period()

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
        """Process action from the current agent in the AEC sequence."""
        agent = self.agent_selection

        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        self._cumulative_rewards[agent] = 0.0

        if agent == "aggregator":
            self._handle_aggregator_step(action)
        else:
            self._handle_farmer_step(agent, action)

        self.acted.add(agent)

        if len(self.acted) == len(self.agents):
            self._resolve_period()

        self.agent_selection = self._selector.next()
        self._accumulate_rewards()

    def _handle_aggregator_step(self, raw_action):
        """Decode aggregator action into contract menu and distribute to farmers."""
        menu = self.agg.decode_action_to_contracts(raw_action)
        for f in self.farmer_objs:
            f.receive_contracts(menu)

    def _handle_farmer_step(self, agent, action):
        """Record farmer's contract choice and practice decisions."""
        idx = int(agent.split("_")[1])
        farmer = self.farmer_objs[idx]
        farmer.choose_contract(action)
        self.farmer_actions[idx] = action

    def _resolve_period(self):
        """Draw weather, compute all outcomes and payments, assign rewards."""
        self.weather = np.random.normal(0, 1)

        for idx, farmer in enumerate(self.farmer_objs):
            action = self.farmer_actions.get(idx)
            if action is None:
                continue

            farmer.compute_yield_and_carbon(action, self.weather)

            if farmer.accepted_contract is not None:
                ct_name = CONTRACT_TYPES[farmer.accepted_contract]
                mrv_cost = self.agg.get_mrv_cost(ct_name, farmer.farm_size)
                farmer.compute_contract_payment(mrv_cost)
                self.agg.process_farmer_outcome(
                    farmer, ct_name, farmer.payment, farmer.carbon)
            else:
                self.agg.process_farmer_rejection(farmer)

            farmer.update_crop_history(action["crop_choice"])

        # Assign rewards
        self.rewards["aggregator"] = self.agg.compute_reward()
        for i, f in enumerate(self.farmer_objs):
            self.rewards[f"farmer_{i}"] = f.compute_reward()

        self.infos["aggregator"] = self.agg.get_period_summary()

        # Advance period
        self.period += 1
        if self.period >= self.num_periods:
            for a in self.agents:
                self.terminations[a] = True
        else:
            self._setup_next_period()

    def _setup_next_period(self):
        """Reset per-period state for multi-period episodes."""
        self.acted = set()
        self.farmer_actions = {}
        self.agg.reset_period()
        for f in self.farmer_objs:
            f.reset_period()
        self._selector = agent_selector(self.agents)
        self.agent_selection = self._selector.reset()

    # ================================================================
    # Rendering
    # ================================================================

    def render(self):
        """Print human-readable summary of the current period."""
        if self.render_mode != "human":
            return

        print(f"\n{'='*55}")
        print(f"Period {self.period}/{self.num_periods} | Weather: {self.weather:.3f}")
        print(f"{self.agg}")

        if self.agg.menu:
            print("\nContract menu:")
            for name in CONTRACT_TYPES:
                p = self.agg.menu[name]
                parts = [f"  {name}:"]
                if "per_action_payments" in p:
                    for a, v in p["per_action_payments"].items():
                        if v > 0:
                            parts.append(f"{a}=${v:.1f}/ac")
                if "per_ton_payment" in p:
                    parts.append(f"ton=${p['per_ton_payment']:.1f}")
                parts.append(f"mrv_share={p['mrv_share']:.0%}")
                print(" | ".join(parts))

        print("\nFarmer outcomes:")
        for i, f in enumerate(self.farmer_objs):
            ct = CONTRACT_TYPES[f.accepted_contract] if f.accepted_contract is not None else "none"
            print(f"  {f} -> {ct} | carbon={f.carbon:.0f}t "
                  f"pay=${f.payment:.0f} reward={self.rewards[f'farmer_{i}']:.0f}")

        s = self.agg.get_period_summary()
        print(f"\nTotals: carbon={s['carbon']:.0f}t profit=${s['profit']:.0f} "
              f"enrolled={s['enrolled']}/{self.num_farmers} "
              f"budget_left=${s['budget_remaining']:.0f}")


# ============================================================
# Sanity test
# ============================================================

if __name__ == "__main__":
    from farmer import PAYABLE_ACTION_NAMES

    env = CarbonFarmingEnv(budget=50000, carbon_weight=0.5,
                           num_periods=1, render_mode="human")

    print(f"Agents: {env.possible_agents}")
    print(f"Agg action dim: {env.action_space('aggregator')}")
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
            action = {k: int(s.sample()) for k, s in env.action_space(agent).items()}
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
            # Hand-designed contracts
            act = []
            # Action contract: $10/acre for cover crop and no-till, $5 for manure, $0 for no-fert
            for a in PAYABLE_ACTION_NAMES:
                if a in ("use_cover_crop", "use_no_till"):
                    act.append(10.0)
                elif a == "use_manure":
                    act.append(5.0)
                else:
                    act.append(0.0)
            act += [0.5, 0.0]  # upfront_frac=50%, mrv_share=0 (agg pays)

            # Result contract: $40/ton, 20% MRV to farmer
            act += [40.0, 0.2]

            # Hybrid: $5/acre for cover+notill, $20/ton, 30% upfront, 10% MRV
            for a in PAYABLE_ACTION_NAMES:
                act.append(5.0 if a in ("use_cover_crop", "use_no_till") else 0.0)
            act += [20.0, 0.3, 0.1]

            action = np.array(act, dtype=np.float32)
        else:
            idx = int(agent.split("_")[1])
            f = env.farmer_objs[idx]
            ct = 0 if f.risk_pref > 0.6 else (1 if f.risk_pref < 0.3 else 2)
            best_crop = CROPS.index(max(f.crop_pref, key=f.crop_pref.get))
            best_input = INPUTS.index(max(f.input_pref, key=f.input_pref.get))
            action = {"contract_choice": ct, "crop_choice": best_crop,
                      "input_choice": best_input, "tillage_choice": 1}
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
            action = env3.action_space(agent).sample()
        else:
            action = {k: int(s.sample()) for k, s in env3.action_space(agent).items()}
        env3.step(action)
    env3.render()

    print(f"\nFinal rewards:")
    for a in env3.possible_agents:
        print(f"  {a}: {env3.rewards[a]:.2f}")