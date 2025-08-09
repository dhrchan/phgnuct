from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import Self

import numpy as np
from unified_planning.model.action import ProbabilisticAction, InstantaneousAction
from unified_planning.model.phgn.method import PHGNMethod
from unified_planning.model.fnode import FNode
from unified_planning.model.phgn.goal_network import PartialOrderGoalNetwork
from unified_planning.model.phgn.phgn_problem import PHGNProblem
from unified_planning.engines.phgn_simulator import PHGNSimulator
from unified_planning.engines.compilers import PHGNGrounderHelper
from unified_planning.model.state import UPState
from phgn_planner.config import UCTConfig
from phgn_planner.factored_tree import (
    DefaultPolicy,
    MaxPolicy,
    TreeNode,
    TreeNodeFactory,
    UCBPolicy,
)


class PlanningResult(Enum):
    """Enum representing the possible outcomes of LAMPPlanner.run().

    SUCCESS         -> Successfully reached the goal
    DEADLOCKED      -> Reached a deadlocking state
    EXCEEDED_BUDGET -> Unable to reach goal within cost budget
    """

    SUCCESS = auto()  # Successfully reached the goal
    FAILURE_DEADLOCKED = auto()  # Reached a deadlocking state
    FAILURE_BUDGET = auto()  # Unable to reach goal within cost budget

    def __repr__(self):
        cls_name = self.__class__.__name__
        return f"{cls_name}.{self.name}"


@dataclass
class PlanningContext:
    problem: PHGNProblem
    simulator: PHGNSimulator
    grounder: PHGNGrounderHelper
    initial_state: UPState
    initial_gtn: PartialOrderGoalNetwork
    node_factory: TreeNodeFactory
    n_rollouts: int
    horizon: int
    budget: float
    exploration_const: float
    normalize_exploration_const: bool
    goal_utility: float
    risk_factor: float
    cost_fn: Callable[[UPState, InstantaneousAction | ProbabilisticAction], float]
    utility_fn: Callable[[float], float]
    h_util: Callable[[UPState], float]
    h_ptg: Callable[[UPState], float]
    q_init: Callable[[UPState, InstantaneousAction | ProbabilisticAction], float]
    n_init: float
    default_policy: DefaultPolicy
    ucb_policy: UCBPolicy
    max_policy: MaxPolicy
    rng: np.random.RandomState


class RolloutResult:
    """The result of a LAMP rollout."""

    def __init__(
        self,
        gtn: PartialOrderGoalNetwork | None = None,
        costs: int = 0,
        has_goal: bool = False,
    ):
        self.costs = {}
        self.has_goal = {}
        if gtn:
            for node in gtn.network:
                self.costs[node.get_content()] = costs
                self.has_goal[node.get_content()] = has_goal

    def increment(self, cost: float | int) -> Self:
        """Increment the costs by `cost`.

        Returns this `RolloutResult` for chaining.
        """
        for goal in self.costs:
            self.costs[goal] += cost
        return self

    def extend(self, subgoal: FNode, cost: float | int, has_goal: bool):
        """Extend the costs for `subgoal`."""
        self.costs.update({subgoal: cost})
        self.has_goal.update({subgoal: has_goal})
        return self


class PHGNPlanner:
    """A UCT planning algorithm for probabilistic HGN planning problems."""

    def __init__(self, cfg: UCTConfig | None = None):
        """The Landmark-Aided Monte Carlo Planning (LAMP) algorithm.

        Parameters
        ----------

        cfg : Optional[UCTConfig]
            The configuration parameters for this PHGNPlanner. If none is provided,
            default values will be used.
        """
        self.cfg = cfg or UCTConfig()

    def _setup(self, problem: PHGNProblem, cfg: UCTConfig) -> PlanningContext:
        """Setup the PlanningContext for a run of this PHGNPlanner."""
        rng = np.random.RandomState(seed=cfg.seed)
        simulator = PHGNSimulator(problem=problem, rng=rng)
        ctx = PlanningContext(
            problem=problem,
            simulator=simulator,
            grounder=PHGNGrounderHelper(problem),
            initial_state=simulator.get_initial_state(),
            initial_gtn=problem.goal_network,
            node_factory=TreeNodeFactory[
                UPState, InstantaneousAction | ProbabilisticAction, PHGNMethod, FNode
            ](simulator),
            n_rollouts=cfg.n_rollouts,
            horizon=cfg.horizon,
            budget=cfg.budget,
            exploration_const=cfg.exploration_const,
            normalize_exploration_const=cfg.normalize_exploration_const,
            goal_utility=cfg.goal_utility,
            risk_factor=cfg.risk_factor,
            cost_fn=lambda s, u: 1
            if isinstance(u[0], (InstantaneousAction, ProbabilisticAction))
            else 0,  # 0 if check_goal(s, g) else 1
            utility_fn=lambda cost: np.exp(cfg.risk_factor * cost),
            h_util=cfg.h_util,
            h_ptg=cfg.h_ptg,
            q_init=lambda s, a: cfg.h_util(s) + cfg.h_ptg(s) * cfg.goal_utility,
            n_init=cfg.n_init,
            default_policy=DefaultPolicy(simulator, rng),
            ucb_policy=UCBPolicy(
                simulator, rng, cfg.exploration_const, cfg.normalize_exploration_const
            ),
            max_policy=MaxPolicy(simulator, rng),
            rng=rng,
        )
        return ctx

    def run(
        self,
        problem: PHGNProblem,
        **override_config,
    ):
        """Iteratively select and apply the best action in each state according to UCT.

        Parameters
        ----------
        problem : Problem
            The problem to run the planner on.

        gtn : PartialOrderGoalNetwork
            The initial goal-task network for the HGN planning problem.

        **override_config : Any
            Configuration parameters (overrides any default parameters, or parameters
            set during the initialization of this PHGNPlanner)
        """
        cfg = replace(self.cfg, **override_config)
        ctx = self._setup(problem, cfg)
        gtn = ctx.initial_gtn

        node: TreeNode = ctx.node_factory.new_node(ctx.initial_state)
        cumulative_cost = 0
        while True:
            if cumulative_cost >= ctx.budget:
                return (
                    PlanningResult.FAILURE_BUDGET,
                    cumulative_cost,
                    ctx.node_factory.num_nodes(),
                )
            if len(node.get_applicable_actions()) == 0:
                return (
                    PlanningResult.FAILURE_DEADLOCKED,
                    cumulative_cost,
                    ctx.node_factory.num_nodes(),
                )
            unconstrained = gtn.get_unconstrained().copy()
            while unconstrained:
                subgoal = unconstrained.pop()
                if ctx.simulator.satisfies(node.state, [subgoal.get_content()]):
                    for successor in gtn.network.successors(subgoal):
                        unconstrained.add(successor)
                    gtn.release(subgoal)
            if gtn.is_empty():
                return (
                    PlanningResult.SUCCESS,
                    cumulative_cost,
                    ctx.node_factory.num_nodes(),
                )
            action_or_method = self._plan(ctx, node, gtn, cumulative_cost)
            if isinstance(action_or_method[0], PHGNMethod):
                gtn.decompose(
                    ctx.grounder.ground_method(
                        action_or_method[0], action_or_method[1]
                    ).goal_network,
                    action_or_method[2],
                )
                if cfg.show_progress:
                    print(
                        f"\rSelected method {action_or_method[0].name, action_or_method[1]}",
                        flush=True,
                    )
            else:
                action_cost = ctx.cost_fn(node.state, action_or_method)
                node = ctx.node_factory.new_node(
                    ctx.simulator.apply(node.state, *action_or_method)
                )
                cumulative_cost += action_cost
                if cfg.show_progress:
                    print(
                        f"\r\tSelected action {action_or_method[0].name, action_or_method[1]}",
                        flush=True,
                    )

    def _plan(
        self,
        ctx: PlanningContext,
        node: TreeNode,
        gtn: PartialOrderGoalNetwork,
        cumulative_cost: int,
    ) -> InstantaneousAction | ProbabilisticAction | PHGNMethod:
        """Perform iterations of PHGN UCT, then return the best action or method."""
        if gtn.is_empty():
            return
        for _ in range(ctx.n_rollouts):
            self._simulate(ctx, node, gtn.copy(), 0, cumulative_cost)
        return ctx.max_policy(node, gtn)

    def _simulate(
        self,
        ctx: PlanningContext,
        node: TreeNode,
        gtn: PartialOrderGoalNetwork,
        depth: int,
        cumulative_cost: int,
    ) -> RolloutResult:
        """Perform one rollout of PHGN UCT and backpropagate costs."""
        # Base Cases
        if gtn.is_empty():
            return RolloutResult()
        for unconstrained in gtn.get_unconstrained():
            if node.satisfies(unconstrained.get_content()):
                gtn.release(unconstrained)
                result = self._simulate(ctx, node, gtn, depth, cumulative_cost)
                return result.extend(unconstrained.get_content(), 0, True)
        if node.is_deadend():
            future_cost = ctx.horizon - 1 - depth
            return RolloutResult(gtn, future_cost, False)
        if depth == ctx.horizon - 1:
            return RolloutResult(gtn, 0, False)
        # The unconstrained subgoal has not yet been achieved. Select an action/method to execute.
        if not node.is_expanded():
            node.expand()
            u = ctx.default_policy(node, gtn)
            if isinstance(u[0], (InstantaneousAction, ProbabilisticAction)):
                next_node = ctx.node_factory.new_node(
                    ctx.simulator.apply(node.state, *u)
                )
                result = self._rollout(ctx, next_node, gtn, depth + 1)
            else:  # u is a Method
                gtn.decompose(ctx.grounder.ground_method(u[0], u[1]).goal_network, u[2])
                result = self._rollout(ctx, node, gtn, depth + 1)
        else:
            u = ctx.ucb_policy(node, gtn)
            if isinstance(u[0], (InstantaneousAction, ProbabilisticAction)):
                next_node = ctx.node_factory.new_node(
                    ctx.simulator.apply(node.state, *u)
                )
                result = self._simulate(
                    ctx, next_node, gtn, depth + 1, cumulative_cost + 1
                )
            else:  # u is a Method
                gtn.decompose(ctx.grounder.ground_method(u[0], u[1]).goal_network, u[2])
                result = self._simulate(ctx, node, gtn, depth + 1, cumulative_cost)
        u_cost = ctx.cost_fn(node.state, u)
        node.update(
            u[:2], result, cumulative_cost + u_cost, ctx.goal_utility, ctx.utility_fn
        )
        return result.increment(u_cost)

    def _rollout(
        self,
        ctx: PlanningContext,
        node: TreeNode,
        gtn: PartialOrderGoalNetwork,
        depth: int,
    ) -> RolloutResult:
        """Perform one rollout of LAMP and backpropagate costs."""
        # Base Cases
        if gtn.is_empty():
            return RolloutResult()
        for unconstrained in gtn.get_unconstrained():
            if node.satisfies(unconstrained.get_content()):
                gtn.release(unconstrained)
                result = self._rollout(ctx, node, gtn, depth)
                return result.extend(unconstrained.get_content(), 0, True)
        if node.is_deadend():
            future_cost = ctx.horizon - 1 - depth
            return RolloutResult(gtn, future_cost, False)
        if depth == ctx.horizon - 1:
            return RolloutResult(gtn, 0, False)
        # The unconstrained subgoal has not yet been achieved. Select an action/method to execute.
        node.expand()
        u = ctx.default_policy(node, gtn)
        if isinstance(u[0], (InstantaneousAction, ProbabilisticAction)):
            next_node = ctx.node_factory.new_node(ctx.simulator.apply(node.state, *u))
            result = self._rollout(ctx, next_node, gtn, depth + 1)
        else:  # u is a Method
            gtn.decompose(ctx.grounder.ground_method(u[0], u[1]).goal_network, u[2])
            result = self._rollout(ctx, node, gtn, depth + 1)
        u_cost = ctx.cost_fn(node.state, u)
        return result.increment(u_cost)
