from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Hashable
from math import log, sqrt
from typing import TYPE_CHECKING

import numpy as np
from unified_planning.model.phgn.goal_network import PartialOrderGoalNetwork
from unified_planning.engines.phgn_simulator import PHGNSimulator
from unified_planning.model.phgn import PHGNMethod


if TYPE_CHECKING:
    from phgn_planner.factored_uct import RolloutResult


class TreeNodeFactory[S: Hashable, A: Hashable, M: Hashable, G: Hashable]:
    """A factory for TreeNodes.

    Ensures that there is only one node created for each underlying state.
    """

    def __init__(self, simulator: PHGNSimulator):
        self._simulator = simulator
        self._nodes = {}
        self._num_nodes = 0

    def new_node(self, state: S) -> TreeNode:
        """Create a new TreeNode.

        Creates a new TreeNode only if the underlying state has not
        yet been encountered. If it has, return the existing instance.
        """
        if state not in self._nodes:
            self._nodes[state] = TreeNode[S, A, M, G](state, self._simulator)
            self._num_nodes += 1
        return self._nodes[state]

    def num_nodes(self) -> int:
        return self._num_nodes


class TreeNode[S: Hashable, A: Hashable, M: Hashable, G: Hashable]:
    """A decision node in a Monte Carlo search tree."""

    def __init__(self, state: S, simulator: PHGNSimulator) -> None:
        self.state: S = state
        self._simulator: PHGNSimulator = simulator
        self._appliable_actions: set[A] | None = None
        self._applicable_methods: set[M] | None = None
        self.visits: dict[G, int] = defaultdict(int)
        self.Q: dict[G, dict[A | M, float]] = defaultdict(lambda: defaultdict(float))
        self.N: dict[G, dict[A | M, int]] = defaultdict(lambda: defaultdict(int))
        self._expanded: bool = False

    def __str__(self):
        s = f"Expanded: {self._expanded}\n"
        s += "Applicable Actions/Methods:\n"
        for a in self.get_applicable_actions():
            for subgoal in self.Q:
                s += (
                    f"  {a[0].name, a[1]}, {subgoal}: Q={self.Q[subgoal][a]}, N={self.N[subgoal][a]}\n"
                    if self.N[subgoal][a] > 0
                    else ""
                )
        for m in self.get_applicable_methods():
            for subgoal in self.Q:
                s += (
                    f"  {m[0].name, m[1]}, {subgoal}: Q={self.Q[subgoal][m]}, N={self.N[subgoal][m]}\n"
                    if self.N[subgoal][m] > 0
                    else ""
                )
        for subgoal in self.visits:
            s += (
                f"Visits {subgoal}: {self.visits[subgoal]}\n"
                if self.visits[subgoal] > 0
                else ""
            )
        return s

    def is_expanded(self) -> bool:
        """Whether this decision node has been expanded.

        Only expanded nodes will have statisics updated after a rollout.
        """
        return self._expanded

    def is_deadend(self) -> bool:
        """Whether there are any applicable actions at this node."""
        return len(self.get_applicable_actions()) == 0

    def expand(self) -> None:
        """Expand this decision node, allowing this node to update UCT statistics."""
        self._expanded = True

    def get_applicable_actions(self) -> set[A]:
        """Generate and return the set of applicable actions at this decision node."""
        if not self._appliable_actions:
            self._appliable_actions = set(
                self._simulator.get_applicable_actions(self.state)
            )
        return self._appliable_actions

    def get_applicable_methods(self) -> set[M]:
        """Generate and return the set of applicable actions at this decision node."""
        if not self._applicable_methods:
            self._applicable_methods = set(
                self._simulator.get_applicable_methods(self.state)
            )
        return self._applicable_methods

    def select(
        self,
        policy: TreePolicy,
        gtn: PartialOrderGoalNetwork,
    ) -> A | M:
        """Select an applicable action at this decision node according to the `policy`."""
        return policy(self, gtn)

    def satisfies(self, goal: G) -> bool:
        return self._simulator.satisfies(self.state, [goal])

    def update(
        self,
        action_or_method: A | M,
        result: RolloutResult,
        cumulative_cost: int,
        goal_utility: float,
        utility_fn: Callable[[float], float],
    ) -> None:
        """Perform a UCB update on this node."""
        for subgoal in result.costs:
            k = goal_utility if result.has_goal[subgoal] else 0
            self.Q[subgoal][action_or_method] = (
                self.N[subgoal][action_or_method] * self.Q[subgoal][action_or_method]
                + utility_fn(result.costs[subgoal] + cumulative_cost)
                + k
            ) / (1 + self.N[subgoal][action_or_method])
            self.N[subgoal][action_or_method] += 1
            self.visits[subgoal] += 1


class TreePolicy[A: Hashable](ABC):
    """A policy used to select among applicable actions at a TreeNode."""

    @abstractmethod
    def __call__(
        self,
        node: TreeNode,
        gtn: PartialOrderGoalNetwork,
        simulator: PHGNSimulator,
        rng: np.random.RandomState | None = None,
    ) -> A:
        raise NotImplementedError()


class UCBPolicy[A: Hashable](TreePolicy):
    """A TreePolicy based using the UCB1 formula."""

    def __init__(
        self,
        simulator: PHGNSimulator,
        rng: np.random.RandomState | None = None,
        c: float = sqrt(2),
        normalize: bool = True,
    ):
        self.normalize = normalize
        self.c = c
        self.simulator = simulator = simulator
        self.rng = rng or np.random.RandomState()

    def __call__(self, node: TreeNode, gtn: PartialOrderGoalNetwork) -> A:
        unconstrained = gtn.get_unconstrained()
        actions = list(node.get_applicable_actions())
        methods = {}
        for m in node.get_applicable_methods():
            relevant_to = self.simulator.is_relevant(*m, gtn)
            if relevant_to:
                methods[m] = relevant_to
        progressions = actions + list(methods.keys())
        c = self.c
        if self.normalize:
            c *= max(
                sum(node.Q[subgoal.get_content()][u] for subgoal in unconstrained)
                for u in progressions
            )
        vals = [
            sum(
                self._ucb_value(
                    node.Q[subgoal.get_content()][u],
                    node.N[subgoal.get_content()][u],
                    node.visits[subgoal.get_content()],
                    c,
                )
                for subgoal in unconstrained
            )
            for u in progressions
        ]
        max_val = max(vals)
        max_progressions = [p for i, p in enumerate(progressions) if vals[i] == max_val]
        i = self.rng.choice(len(max_progressions))
        r = max_progressions[i]
        if isinstance(r[0], PHGNMethod):
            r += tuple(methods[r])
        return r

    def _ucb_value(self, q_a: float, n_a: int, n: int, c: float) -> float:
        if n_a == 0:
            return np.inf
        return q_a + c * sqrt(log(n) / n_a)


class MaxPolicy[A: Hashable](TreePolicy):
    """A purely exploitative TreePolicy which selects the action with the maximum Q value."""

    def __init__(
        self, simulator: PHGNSimulator, rng: np.random.RandomState | None = None
    ):
        self.simulator: PHGNSimulator = simulator
        self.rng = rng or np.random.RandomState()

    def __call__(self, node: TreeNode, gtn: PartialOrderGoalNetwork) -> A:
        unconstrained = gtn.get_unconstrained()
        actions = list(node.get_applicable_actions())
        methods = {}
        for m in node.get_applicable_methods():
            relevant_to = self.simulator.is_relevant(*m, gtn)
            if relevant_to:
                methods[m] = relevant_to
        progressions = actions + list(methods.keys())
        print([p[0].name for p in progressions])
        vals = [
            sum(node.Q[subgoal.get_content()][u] for subgoal in unconstrained)
            for u in progressions
        ]
        print(vals)
        max_val = max(vals)
        max_progressions = [p for i, p in enumerate(progressions) if vals[i] == max_val]
        i = self.rng.choice(len(max_progressions))
        print(len(max_progressions))
        print(i)
        r = max_progressions[i]
        if isinstance(r[0], PHGNMethod):
            r += tuple(methods[r])
        return r


class RobustPolicy[A: Hashable](TreePolicy):
    """A TreePolicy which selects the action with the maximum N value."""

    def __init__(
        self, simulator: PHGNSimulator, rng: np.random.RandomState | None = None
    ) -> None:
        self.simulator: PHGNSimulator = simulator
        self.rng = rng or np.random.RandomState()

    def __call__(self, node: TreeNode, gtn: PartialOrderGoalNetwork) -> A:
        unconstrained = gtn.get_unconstrained()
        actions = list(node.get_applicable_actions())
        methods = {}
        for m in node.get_applicable_methods():
            relevant_to = self.simulator.is_relevant(*m, gtn)
            if relevant_to:
                methods[m] = relevant_to
        progressions = actions + list(methods.keys())
        vals = [
            sum(node.N[subgoal.get_content()][u] for subgoal in unconstrained)
            for u in progressions
        ]
        max_val = max(vals)
        max_progressions = [p for i, p in enumerate(progressions) if vals[i] == max_val]
        i = self.rng.choice(len(max_progressions))
        r = max_progressions[i]
        if isinstance(r[0], PHGNMethod):
            r += tuple(methods[r])
        return r


class DefaultPolicy[A: Hashable](TreePolicy):
    """A TreePolicy which selects the action at random."""

    def __init__(
        self, simulator: PHGNSimulator, rng: np.random.RandomState | None = None
    ) -> None:
        self.simulator: PHGNSimulator = simulator
        self.rng = rng or np.random.RandomState()

    def __call__(self, node: TreeNode, gtn: PartialOrderGoalNetwork) -> A:
        actions = list(node.get_applicable_actions())
        methods = {}
        for m in node.get_applicable_methods():
            relevant_to = self.simulator.is_relevant(*m, gtn)
            if relevant_to:
                methods[m] = relevant_to
        progressions = actions + list(methods.keys())
        i = self.rng.choice(len(progressions))
        r = progressions[i]
        if isinstance(r[0], PHGNMethod):
            r += tuple(methods[r])
        return r
