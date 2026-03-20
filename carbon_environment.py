"""
Carbon Farming Contract Design - PettingZoo AEC Environment (Minimal Workshop Version)

Logging levels:
    DEBUG:   every internal calculation, soil noise, history updates
    INFO:    every action, yield/carbon/cost breakdown, payment breakdown, reward breakdown
    WARNING: budget exceeded, unexpected states

Usage:
    # Verbose mode (see every calculation):
    python carbon_environment.py --verbose

    # Normal mode (just final summaries):
    python carbon_environment.py

    # Or configure programmatically:
    import logging
    logging.getLogger("carbon_farming").setLevel(logging.DEBUG)
"""

import functools
import logging
import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector
from gymnasium import spaces

from farmer import Farmer, create_farmers, CROPS, INPUTS, TILLAGE, CONTRACT_TYPES
from aggregator import Aggregator

logger = logging.getLogger("carbon_farming.env")


def setup_logging(level=logging.INFO, logfile=None):
    """
    Configure logging for the entire carbon_farming package.

    Args:
        level: logging.DEBUG for every calculation, logging.INFO for actions+rewards
        logfile: if provided, also write logs to this file
    """
    root = logging.getLogger("carbon_farming")
    root.setLevel(level)
    root.handlers.clear()

    fmt = logging.Formatter("[%(name)s] %(message)s")

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(fmt)
    root.addHandler(console)

    if logfile:
        fh = logging.FileHandler(logfile, mode="w")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s"))
        root.addHandler(fh)

    return root


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

        self.possible_agents = ["aggregator"] + [
            f"farmer_{i}" for i in range(self.num_farmers)]
        self.agents = self.possible_agents[:]

        self._selector = AgentSelector(self.agents)
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

        logger.info(f"Environment created: {self.num_farmers} farmers, "
                    f"budget=${budget:.0f}, carbon_weight={carbon_weight:.2f}, "
                    f"periods={num_periods}")

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
        self._selector = AgentSelector(self.agents)
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

        logger.info(f"\n{'='*60}")
        logger.info(f"EPISODE RESET (seed={seed})")
        logger.info(f"{'='*60}")

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
        logger.info(f"\n{'—'*55}")
        logger.info(f"PERIOD {self.period + 1}/{self.num_periods} — AGGREGATOR POSTS CONTRACTS")
        logger.info(f"{'—'*55}")
        logger.info(f"Raw action vector ({len(raw_action)} dims): "
                    f"[{', '.join(f'{x:.2f}' for x in raw_action)}]")

        menu = self.agg.decode_action_to_contracts(raw_action)
        for f in self.farmer_objs:
            f.receive_contracts(menu)

    def _handle_farmer_step(self, agent, action):
        """Record farmer's contract choice and practice decisions."""
        idx = int(agent.split("_")[1])
        farmer = self.farmer_objs[idx]

        logger.info(f"\n{'—'*55}")
        logger.info(f"FARMER {idx} DECIDES")
        logger.info(f"{'—'*55}")

        farmer.choose_contract(action)
        self.farmer_actions[idx] = action

    def _resolve_period(self):
        """Draw weather, compute all outcomes and payments, assign rewards."""
        self.weather = np.random.normal(0, 1)

        logger.info(f"\n{'='*55}")
        logger.info(f"RESOLVING PERIOD {self.period + 1}/{self.num_periods}")
        logger.info(f"Weather shock: {self.weather:.4f}")
        logger.info(f"{'='*55}")

        for idx, farmer in enumerate(self.farmer_objs):
            action = self.farmer_actions.get(idx)
            if action is None:
                logger.warning(f"[Farmer {idx}] No action recorded — skipping")
                continue

            logger.info(f"\n--- Farmer {idx} outcomes ---")
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
        logger.info(f"\n{'—'*55}")
        logger.info(f"COMPUTING REWARDS")
        logger.info(f"{'—'*55}")

        for i, f in enumerate(self.farmer_objs):
            self.rewards[f"farmer_{i}"] = f.compute_reward()

        self.rewards["aggregator"] = self.agg.compute_reward()

        self.infos["aggregator"] = self.agg.get_period_summary()

        # Log summary
        s = self.agg.get_period_summary()
        logger.info(f"\n{'—'*55}")
        logger.info(f"PERIOD {self.period + 1} SUMMARY")
        logger.info(f"{'—'*55}")
        logger.info(f"  Enrolled: {s['enrolled']}/{self.num_farmers}")
        logger.info(f"  Total carbon: {s['carbon']:.4f} tCO2e")
        logger.info(f"  Total revenue: ${s['revenue']:.2f}")
        logger.info(f"  Total payments: ${s['payments']:.2f}")
        logger.info(f"  Total MRV cost: ${s['mrv_cost']:.2f}")
        logger.info(f"  Aggregator profit: ${s['profit']:.2f}")
        logger.info(f"  Budget remaining: ${s['budget_remaining']:.2f}")
        logger.info(f"  Aggregator reward: {self.rewards['aggregator']:.2f}")
        for i in range(self.num_farmers):
            logger.info(f"  Farmer {i} reward: {self.rewards[f'farmer_{i}']:.2f}")

        # Advance period
        self.period += 1
        if self.period >= self.num_periods:
            logger.info(f"\nEPISODE COMPLETE after {self.period} period(s)")
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
        self._selector = AgentSelector(self.agents)
        self.agent_selection = self._selector.reset()
        logger.info(f"\nAdvancing to period {self.period + 1}/{self.num_periods}")

    # ================================================================
    # Rendering (minimal — logging does the heavy lifting now)
    # ================================================================

    def render(self):
        """Print compact summary (use logging for detailed view)."""
        if self.render_mode != "human":
            return
        s = self.agg.get_period_summary()
        print(f"\nPeriod {self.period}/{self.num_periods} | "
              f"Carbon={s['carbon']:.2f}t | Profit=${s['profit']:.0f} | "
              f"Enrolled={s['enrolled']}/{self.num_farmers}")


# ============================================================
# Test with full logging
# ============================================================

if __name__ == "__main__":
    import argparse
    from farmer import PAYABLE_ACTION_NAMES

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true",
                        help="Show DEBUG-level logs (every internal calculation)")
    parser.add_argument("--logfile", type=str, default=None,
                        help="Write logs to file (e.g. --logfile run.log)")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=level, logfile=args.logfile)

    env = CarbonFarmingEnv(budget=50000, carbon_weight=0.5,
                           num_periods=1, render_mode="human")

    print(f"Agents: {env.possible_agents}")
    print(f"Agg action space: {env.action_space('aggregator')}")
    print(f"Farmer action space: {env.action_space('farmer_0')}")

    # ---- Heuristic episode ----
    env.reset(seed=42)

    for agent in env.agent_iter():
        obs, rew, term, trunc, info = env.last()

        if term or trunc:
            action = None
        elif agent == "aggregator":
            act = []
            # Action contract
            for a in PAYABLE_ACTION_NAMES:
                act.append(10.0 if a in ("use_cover_crop", "use_no_till") else 5.0)
            act += [0.5, 0.0]  # upfront_frac, mrv_share
            # Result contract
            act += [40.0, 0.2]
            # Hybrid contract
            for a in PAYABLE_ACTION_NAMES:
                act.append(5.0 if a in ("use_cover_crop", "use_no_till") else 2.0)
            act += [20.0, 0.3, 0.1]
            action = np.array(act, dtype=np.float32)
        else:
            idx = int(agent.split("_")[1])
            f = env.farmer_objs[idx]
            # Heuristic: risk-averse -> action, risk-tolerant -> result, moderate -> hybrid
            ct = 0 if f.risk_pref > 0.6 else (1 if f.risk_pref < 0.3 else 2)
            best_crop = CROPS.index(max(f.crop_pref, key=f.crop_pref.get))
            best_input = INPUTS.index(max(f.input_pref, key=f.input_pref.get))
            action = {"contract_choice": ct, "crop_choice": best_crop,
                      "input_choice": best_input, "tillage_choice": 1}

        env.step(action)

    env.render()