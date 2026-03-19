"""
Carbon Farming Contract Design - PettingZoo AEC Environment
============================================================
Multi-agent environment implementing the Stackelberg game between
an aggregator (leader) and heterogeneous farmers (followers).

Step sequence per period:
    1. Aggregator observes farmer public/noisy info, chooses contract menu
    2. Each farmer observes contract offers + own type, chooses:
       - Which contract to accept (or reject all)
       - Crop, input, and tillage decisions
    3. Environment resolves weather, computes yields, carbon, payments
    4. Rewards are distributed

Uses PettingZoo's AEC (Agent Environment Cycle) API which naturally
encodes the sequential Stackelberg timing.
"""

import functools
import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector
from gymnasium import spaces

from farmer_claude import (
    Farmer, create_workshop_farmers,
    CROPS, INPUTS, TILLAGE_OPTIONS,
    NUM_CONTRACT_CHOICES,
)
from aggregator_claude import (
    Aggregator,
    NUM_CONTRACTS, NUM_PARAMS_PER_CONTRACT,
    MAX_PER_ACRE_PAYMENT, MAX_PER_TON_PAYMENT,
    MAX_UPFRONT_FRACTION, MAX_MRV_COST_SHARE,
    CARBON_MARKET_PRICE,
)


class CarbonFarmingEnv(AECEnv):
    """
    PettingZoo AEC environment for the carbon farming contract design game.

    Agents:
        - "aggregator": the carbon credit intermediary (moves first)
        - "farmer_0" ... "farmer_N": heterogeneous farmers (move after aggregator)

    Episode structure (single-period version for workshop paper):
        1. aggregator acts: sets contract menu parameters
        2. farmer_0 acts: chooses contract + farming practices
        3. farmer_1 acts: chooses contract + farming practices
        ...
        N+1. farmer_N acts
        N+2. environment resolves weather, computes outcomes, assigns rewards
        Episode terminates.

    For multi-period extension: wrap steps 1–(N+2) in a loop over periods,
    updating soil health between periods.

    Args:
        farmers: list of Farmer objects (default: 5 workshop types)
        budget: aggregator's total budget per period
        carbon_weight: aggregator's weight on carbon vs profit in [0,1]
        num_periods: number of periods per episode (1 for workshop)
        soil_noise_std: std dev of noise on soil quality signal to aggregator
    """

    metadata = {
        "render_modes": ["human"],
        "name": "carbon_farming_v0",
        "is_parallelizable": False,
    }

    def __init__(
        self,
        farmers: list = None,
        budget: float = 50000.0,
        carbon_weight: float = 0.5,
        num_periods: int = 1,
        soil_noise_std: float = 0.15,
        render_mode: str = None,
    ):
        super().__init__()

        # ---- Agents ----
        self.farmer_objects = farmers if farmers is not None else create_workshop_farmers()
        self.num_farmers = len(self.farmer_objects)
        self.aggregator_obj = Aggregator(budget=budget, carbon_weight=carbon_weight)

        # Agent names
        self.possible_agents = ["aggregator"] + [
            f"farmer_{i}" for i in range(self.num_farmers)
        ]
        self.agents = self.possible_agents[:]

        # ---- Environment config ----
        self.num_periods = num_periods
        self.soil_noise_std = soil_noise_std
        self.render_mode = render_mode

        # ---- AEC internals ----
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        # ---- State tracking ----
        self.current_period = 0
        self.weather_shock = 0.0
        self.farmer_actions = {}       # farmer_id -> action dict
        self.has_acted = set()         # agents that have acted this period
        self.period_done = False

        # ---- PettingZoo required attributes ----
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}

    # ================================================================
    # Spaces
    # ================================================================

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Dict:
        """Return the observation space for the given agent."""
        if agent == "aggregator":
            return Aggregator.observation_space(num_farmers=self.num_farmers)
        else:
            return Farmer.observation_space()

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str):
        """Return the action space for the given agent."""
        if agent == "aggregator":
            return Aggregator.action_space()
        else:
            return Farmer.action_space()

    # ================================================================
    # Observations
    # ================================================================

    def observe(self, agent: str) -> dict:
        """Return the current observation for the given agent."""
        if agent == "aggregator":
            return self.aggregator_obj.get_observation(self.farmer_objects)
        else:
            farmer_idx = int(agent.split("_")[1])
            return self.farmer_objects[farmer_idx].get_observation()

    # ================================================================
    # Core AEC Methods
    # ================================================================

    def reset(self, seed=None, options=None):
        """Reset the environment for a new episode."""
        if seed is not None:
            np.random.seed(seed)

        # Reset agents list
        self.agents = self.possible_agents[:]
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        # Reset aggregator
        self.aggregator_obj.hard_reset()

        # Reset farmers (preserves types, clears per-episode state)
        for f in self.farmer_objects:
            f.reset()

        # Reset environment state
        self.current_period = 0
        self.weather_shock = 0.0
        self.farmer_actions = {}
        self.has_acted = set()
        self.period_done = False

        # Reset PettingZoo attributes
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}

    def step(self, action):
        """
        Process an action from the current agent.

        The AEC API calls step() once per agent in sequence:
            1. aggregator posts contracts
            2. farmer_0 chooses contract + practices
            ...
            N+1. farmer_N chooses contract + practices
            -> after last farmer, resolve the period
        """
        agent = self.agent_selection

        # If agent is already done, skip
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        # Clear cumulative rewards for this agent at the start of their turn
        self._cumulative_rewards[agent] = 0.0

        # ---- Aggregator's turn ----
        if agent == "aggregator":
            self._handle_aggregator_action(action)

        # ---- Farmer's turn ----
        else:
            self._handle_farmer_action(agent, action)

        # ---- Check if all agents have acted this period ----
        self.has_acted.add(agent)

        if len(self.has_acted) == len(self.agents):
            # All agents have acted — resolve the period
            self._resolve_period()
            # If a new period started, _resolve_period already reset the
            # agent selector to the aggregator — skip the next() call.
            self._accumulate_rewards()
            return

        # Advance to next agent
        self.agent_selection = self._agent_selector.next()

        # Accumulate rewards
        self._accumulate_rewards()

    def _handle_aggregator_action(self, action: np.ndarray):
        """Process the aggregator's contract menu action."""
        # Decode raw action vector into contract menu
        contracts = self.aggregator_obj.decode_action(action)

        # Distribute contracts to all farmers
        for f in self.farmer_objects:
            f.update_contracts_offered(contracts)

        self.infos["aggregator"] = {"contracts_offered": contracts}

    def _handle_farmer_action(self, agent: str, action: dict):
        """Process a farmer's contract choice and farming decisions."""
        farmer_idx = int(agent.split("_")[1])
        farmer = self.farmer_objects[farmer_idx]

        # Record the farmer's choice
        farmer.step(action)
        self.farmer_actions[farmer_idx] = action

        self.infos[agent] = {
            "contract_chosen": action["contract_choice"],
            "crop": CROPS[action["crop_choice"]],
            "input": INPUTS[action["input_choice"]],
            "tillage": TILLAGE_OPTIONS[action["tillage_choice"]],
        }

    def _resolve_period(self):
        """
        After all agents have acted, resolve weather, compute outcomes,
        calculate payments, and assign rewards.
        """
        # Draw weather shock (same for all farmers this period)
        self.weather_shock = np.random.normal(0, 1)

        contract_names = ["action", "result", "hybrid"]

        # ---- Process each farmer's outcomes ----
        for farmer_idx, farmer in enumerate(self.farmer_objects):
            actions = self.farmer_actions.get(farmer_idx)
            if actions is None:
                continue

            # Compute yield and carbon
            outcomes = farmer.compute_outcomes(actions, self.weather_shock)

            if farmer.current_contract is not None:
                # Farmer accepted a contract
                ct_name = contract_names[farmer.current_contract]
                contract_params = self.aggregator_obj.current_contracts[ct_name]

                # Compute MRV cost and payment
                mrv_cost = self.aggregator_obj.get_mrv_cost(ct_name, farmer.farm_size)
                payment = farmer.compute_contract_payment(contract_params, mrv_cost)

                # Update aggregator tracking
                self.aggregator_obj.process_farmer_outcome(
                    farmer, ct_name, payment, outcomes["carbon"]
                )
            else:
                # Farmer rejected all contracts
                self.aggregator_obj.process_rejected_farmer(farmer)

            # Update farmer crop history
            farmer.update_crop_history(actions["crop_choice"])

        # ---- Compute rewards ----
        # Aggregator reward
        agg_reward = self.aggregator_obj.reward() + self.aggregator_obj.budget_penalty()
        self.rewards["aggregator"] = agg_reward

        # Farmer rewards
        for farmer_idx, farmer in enumerate(self.farmer_objects):
            agent_name = f"farmer_{farmer_idx}"
            self.rewards[agent_name] = farmer.reward()

        # ---- Advance period ----
        self.current_period += 1

        if self.current_period >= self.num_periods:
            # Episode is over
            for agent in self.agents:
                self.terminations[agent] = True
        else:
            # Reset for next period (multi-period mode)
            self.has_acted = set()
            self.farmer_actions = {}
            self.aggregator_obj.reset()  # soft reset (preserves history)
            for f in self.farmer_objects:
                f.contracts_offered = None
                f.current_contract = None
                f.current_actions = None
                f.current_payment = 0.0

            # Reset agent selector for next period
            self._agent_selector = AgentSelector(self.agents)
            self.agent_selection = self._agent_selector.reset()

        # ---- Populate infos ----
        self.infos["aggregator"]["period_summary"] = self.aggregator_obj.get_period_summary()

    # ================================================================
    # Rendering
    # ================================================================

    def render(self):
        """Print a human-readable summary of the current state."""
        if self.render_mode != "human":
            return

        print(f"\n{'='*60}")
        print(f"Carbon Farming Environment — Period {self.current_period}/{self.num_periods}")
        print(f"{'='*60}")
        print(f"Weather shock: {self.weather_shock:.3f}")
        print(f"\nAggregator: {self.aggregator_obj}")

        if self.aggregator_obj.current_contracts:
            print("\nContract Menu:")
            for name, params in self.aggregator_obj.current_contracts.items():
                print(f"  {name}: acre=${params['per_acre_payment']:.1f}, "
                      f"ton=${params['per_ton_payment']:.1f}, "
                      f"upfront={params['upfront_fraction']:.1%}, "
                      f"mrv_share={params['mrv_cost_share']:.1%}, "
                      f"λ={params['lambda']:.1f}")

        contract_names = ["action", "result", "hybrid"]
        print("\nFarmer Outcomes:")
        for i, f in enumerate(self.farmer_objects):
            ct = f.current_contract
            ct_str = contract_names[ct] if ct is not None else "rejected"
            print(f"  {f} -> {ct_str}, "
                  f"carbon={f.current_carbon:.1f}t, "
                  f"payment=${f.current_payment:.0f}, "
                  f"reward={self.rewards.get(f'farmer_{i}', 0):.0f}")

        summary = self.aggregator_obj.get_period_summary()
        print(f"\nPeriod Summary:")
        print(f"  Total carbon: {summary['total_carbon']:.1f} tons")
        print(f"  Total payments: ${summary['total_payments']:.0f}")
        print(f"  Aggregator profit: ${summary['profit']:.0f}")
        print(f"  Budget remaining: ${summary['budget_remaining']:.0f}")
        print(f"  Aggregator reward: {self.rewards.get('aggregator', 0):.0f}")


# ============================================================
# Sanity Test
# ============================================================

if __name__ == "__main__":
    print("Creating environment...")
    env = CarbonFarmingEnv(
        budget=50000.0,
        carbon_weight=0.5,
        num_periods=1,
        render_mode="human",
    )

    print(f"Agents: {env.possible_agents}")
    print(f"Aggregator action space: {env.action_space('aggregator')}")
    print(f"Farmer action space: {env.action_space('farmer_0')}")

    # ---- Run one full episode with random actions ----
    print("\n" + "=" * 60)
    print("Running Episode with Random Actions")
    print("=" * 60)

    env.reset(seed=42)

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        elif agent == "aggregator":
            # Random contract parameters
            action = env.action_space(agent).sample()
        else:
            # Random farmer decisions
            action = {
                "contract_choice": np.random.randint(0, 4),
                "crop_choice": np.random.randint(0, 4),
                "input_choice": np.random.randint(0, len(INPUTS)),
                "tillage_choice": np.random.randint(0, 2),
            }

        env.step(action)

    env.render()

    # ---- Print final rewards ----
    print("\n" + "=" * 60)
    print("Final Rewards")
    print("=" * 60)
    for agent in env.possible_agents:
        print(f"  {agent}: {env.rewards.get(agent, 0.0):.2f}")

    # ---- Run a second episode with strategic aggregator ----
    print("\n" + "=" * 60)
    print("Running Episode with Hand-Designed Contracts")
    print("=" * 60)

    env.reset(seed=123)

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        elif agent == "aggregator":
            # Hand-designed contract menu
            action = np.array([
                # Action contract: $12/acre, $0/ton, 50% upfront, aggregator pays MRV
                12.0, 0.0, 0.5, 0.0,
                # Result contract: $0/acre, $40/ton, no upfront, 20% MRV to farmer
                0.0, 40.0, 0.0, 0.2,
                # Hybrid contract: $6/acre, $20/ton, 30% upfront, 10% MRV to farmer
                6.0, 20.0, 0.3, 0.1,
            ], dtype=np.float32)
        else:
            # Farmers choose based on simple heuristic
            farmer_idx = int(agent.split("_")[1])
            farmer = env.farmer_objects[farmer_idx]

            # Risk-averse farmers prefer action, risk-tolerant prefer result
            if farmer.risk_preference > 0.6:
                contract = 0  # action-based
            elif farmer.risk_preference < 0.3:
                contract = 1  # result-based
            else:
                contract = 2  # hybrid

            # Choose most preferred crop and input
            best_crop = max(farmer.crop_preference, key=farmer.crop_preference.get)
            best_input = max(farmer.input_preference, key=farmer.input_preference.get)

            action = {
                "contract_choice": contract,
                "crop_choice": CROPS.index(best_crop),
                "input_choice": INPUTS.index(best_input),
                "tillage_choice": 1,  # all try no-till for carbon
            }

        env.step(action)

    env.render()

    print("\n" + "=" * 60)
    print("Final Rewards (Hand-Designed)")
    print("=" * 60)
    for agent in env.possible_agents:
        print(f"  {agent}: {env.rewards.get(agent, 0.0):.2f}")

    # ---- Multi-period test ----
    print("\n" + "=" * 60)
    print("Running 3-Period Episode")
    print("=" * 60)

    env_mp = CarbonFarmingEnv(
        budget=50000.0,
        carbon_weight=0.5,
        num_periods=3,
        render_mode="human",
    )
    env_mp.reset(seed=99)

    period_rewards = {agent: [] for agent in env_mp.possible_agents}

    for agent in env_mp.agent_iter():
        observation, reward, termination, truncation, info = env_mp.last()

        if termination or truncation:
            action = None
        elif agent == "aggregator":
            action = np.array([
                10.0, 0.0, 0.5, 0.0,
                0.0, 35.0, 0.0, 0.2,
                5.0, 18.0, 0.3, 0.1,
            ], dtype=np.float32)
        else:
            action = {
                "contract_choice": np.random.randint(0, 4),
                "crop_choice": np.random.randint(0, 4),
                "input_choice": np.random.randint(0, len(INPUTS)),
                "tillage_choice": np.random.randint(0, 2),
            }

        env_mp.step(action)

    print(f"\nEnvironment terminated after {env_mp.current_period} periods.")
    print("Final rewards:")
    for agent in env_mp.possible_agents:
        print(f"  {agent}: {env_mp.rewards.get(agent, 0.0):.2f}")