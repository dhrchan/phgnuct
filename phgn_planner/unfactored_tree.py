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

    def new_node(self, state: S, gtn: PartialOrderGoalNetwork) -> TreeNode:
        """Create a new TreeNode.

        Creates a new TreeNode only if the underlying state has not
        yet been encountered. If it has, return the existing instance.
        """
        unconstrained = gtn.get_unconstrained().copy()
        while unconstrained:
            subgoal = unconstrained.pop()
            if self._simulator.satisfies(state, [subgoal.get_content()]):
                for successor in gtn.network.successors(subgoal):
                    unconstrained.add(successor)
                gtn.release(subgoal)
        if state not in self._nodes:
            self._nodes[state] = {}
        if gtn not in self._nodes[state]:
            self._nodes[state][gtn] = TreeNode[S, A, M, G](state, gtn, self._simulator)
            self._num_nodes += 1
        return self._nodes[state][gtn]

    def num_nodes(self) -> int:
        return self._num_nodes


class TreeNode[S: Hashable, A: Hashable, M: Hashable, G: Hashable]:
    """A decision node in a Monte Carlo search tree."""

    def __init__(self, state: S, gtn, simulator: PHGNSimulator) -> None:
        self.state: S = state
        self.gtn: PartialOrderGoalNetwork = gtn
        self._simulator: PHGNSimulator = simulator
        self._appliable_actions: set[A] | None = set()
        self._applicable_methods: set[M] | None = set()
        self.visits: int = 0
        self.Q: dict[A | M, float] = defaultdict(float)
        self.N: dict[A | M, int] = defaultdict(float)
        self._expanded: bool = False

    def __str__(self):
        s = f"Goal Network: {self.gtn}\n"
        s += f"Expanded: {self._expanded}\n"
        s += "Applicable Actions/Methods:\n"
        for a in self._appliable_actions:
            s += f"  {a[0].name, a[1]}: Q={self.Q[a]}, N={self.N[a]}\n"
        for m in self._applicable_methods:
            s += f"  {m[0].name, m[1]}: Q={self.Q[m]}, N={self.N[m]}\n"
        s += f"Visits: {self.visits}\n"
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
    ) -> A | M:
        """Select an applicable action at this decision node according to the `policy`."""
        return policy(self)

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
        k = goal_utility if result.has_goal else 0
        self.Q[action_or_method] = (
            self.N[action_or_method] * self.Q[action_or_method]
            + utility_fn(result.cost + cumulative_cost)
            + k
        ) / (1 + self.N[action_or_method])
        self.N[action_or_method] += 1
        self.visits += 1
        self._locked = True


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

    def __call__(self, node: TreeNode) -> A:
        actions = list(node.get_applicable_actions())
        methods = {}
        for m in node.get_applicable_methods():
            relevant_to = self.simulator.is_relevant(*m, node.gtn)
            if relevant_to:
                methods[m] = relevant_to
        progressions = actions + list(methods.keys())
        c = self.c
        if self.normalize:
            c *= max(node.Q[u] for u in progressions)
        vals = [
            self._ucb_value(node.Q[u], node.N[u], node.visits, c) for u in progressions
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

    def __call__(self, node: TreeNode) -> A:
        actions = list(node.get_applicable_actions())
        methods = {}
        for m in node.get_applicable_methods():
            relevant_to = self.simulator.is_relevant(*m, node.gtn)
            if relevant_to:
                methods[m] = relevant_to
        progressions = actions + list(methods.keys())
        vals = [node.Q[u] for u in progressions]
        max_val = max(vals)
        max_progressions = [p for i, p in enumerate(progressions) if vals[i] == max_val]
        i = self.rng.choice(len(max_progressions))
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

    def __call__(self, node: TreeNode) -> A:
        actions = list(node.get_applicable_actions())
        methods = {}
        for m in node.get_applicable_methods():
            relevant_to = self.simulator.is_relevant(*m, node.gtn)
            if relevant_to:
                methods[m] = relevant_to
        progressions = actions + list(methods.keys())
        vals = [node.N[u] for u in progressions]
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

    def __call__(self, node: TreeNode) -> A:
        actions = list(node.get_applicable_actions())
        methods = {}
        for m in node.get_applicable_methods():
            relevant_to = self.simulator.is_relevant(*m, node.gtn)
            if relevant_to:
                methods[m] = relevant_to
        progressions = actions + list(methods.keys())
        i = self.rng.choice(len(progressions))
        r = progressions[i]
        if isinstance(r[0], PHGNMethod):
            r += tuple(methods[r])
        return r
