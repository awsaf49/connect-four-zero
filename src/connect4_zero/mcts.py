from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import numpy as np

from .game import COLS, ConnectFour


class Predictor(Protocol):
    def predict(self, states: Sequence[ConnectFour]) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass(slots=True)
class SearchSnapshot:
    simulation: int
    visits: list[int]
    q_values: list[float]
    priors: list[float]
    root_value: float


@dataclass(slots=True)
class SearchResult:
    visits: np.ndarray
    policy: np.ndarray
    q_values: np.ndarray
    priors: np.ndarray
    root_value: float
    simulations: int
    tree_size: int
    principal_variations: list[list[int]]
    snapshots: list[SearchSnapshot]


class Node:
    __slots__ = ("prior", "visits", "value_sum", "children", "expanded", "network_value")

    def __init__(self, prior: float = 0.0) -> None:
        self.prior = float(prior)
        self.visits = 0
        self.value_sum = 0.0
        self.children: dict[int, Node] = {}
        self.expanded = False
        self.network_value = 0.0

    @property
    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


class MCTS:
    def __init__(
        self,
        predictor: Predictor,
        simulations: int = 256,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        seed: int | None = None,
    ) -> None:
        if simulations < 1:
            raise ValueError("simulations must be positive")
        self.predictor = predictor
        self.simulations = simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.rng = np.random.default_rng(seed)

    def search(
        self,
        state: ConnectFour,
        *,
        add_noise: bool = False,
        root: Node | None = None,
        record_snapshots: bool = False,
    ) -> tuple[SearchResult, Node]:
        results, roots = self.search_many(
            [state],
            add_noise=add_noise,
            roots=[root] if root is not None else None,
            record_snapshots=record_snapshots,
        )
        return results[0], roots[0]

    def search_many(
        self,
        states: Sequence[ConnectFour],
        *,
        add_noise: bool = False,
        roots: Sequence[Node | None] | None = None,
        record_snapshots: bool = False,
    ) -> tuple[list[SearchResult], list[Node]]:
        if not states:
            return [], []
        actual_roots = [root if root is not None else Node() for root in (roots or [None] * len(states))]
        if len(actual_roots) != len(states):
            raise ValueError("roots and states must have equal length")

        pending = [(index, state, root) for index, (state, root) in enumerate(zip(states, actual_roots)) if not state.is_terminal and not root.expanded]
        if pending:
            logits, values = self.predictor.predict([item[1] for item in pending])
            for batch_index, (_, state, node) in enumerate(pending):
                self._expand(node, state, logits[batch_index], float(values[batch_index]))
        if add_noise:
            for state, root in zip(states, actual_roots):
                if not state.is_terminal:
                    self._add_root_noise(root)

        snapshots: list[list[SearchSnapshot]] = [[] for _ in states]
        snapshot_steps = {1, 2, 4, 8, 16, 32, 64, 128, self.simulations}
        for simulation in range(1, self.simulations + 1):
            traversals: list[tuple[Node, ConnectFour, list[Node]]] = []
            pending_states: list[ConnectFour] = []
            pending_nodes: list[Node] = []
            terminal_values: dict[int, float] = {}

            for index, (root_state, root) in enumerate(zip(states, actual_roots)):
                if root_state.is_terminal:
                    terminal_values[index] = root_state.terminal_value()
                    traversals.append((root, root_state, []))
                    continue
                node, leaf_state, path = self._traverse(root, root_state)
                traversals.append((node, leaf_state, path))
                if leaf_state.is_terminal:
                    terminal_values[index] = leaf_state.terminal_value()
                else:
                    pending_states.append(leaf_state)
                    pending_nodes.append(node)

            predicted: dict[int, float] = {}
            if pending_states:
                logits, values = self.predictor.predict(pending_states)
                cursor = 0
                for index, (node, leaf_state, _) in enumerate(traversals):
                    if index in terminal_values:
                        continue
                    self._expand(node, leaf_state, logits[cursor], float(values[cursor]))
                    predicted[index] = float(values[cursor])
                    cursor += 1

            for index, (_, _, path) in enumerate(traversals):
                self._backup(path, terminal_values.get(index, predicted.get(index, 0.0)))

            if record_snapshots and simulation in snapshot_steps:
                for index, root in enumerate(actual_roots):
                    snapshots[index].append(self._snapshot(root, simulation))

        results = [self._result(root, snapshots[index]) for index, root in enumerate(actual_roots)]
        return results, actual_roots

    def _traverse(self, root: Node, state: ConnectFour) -> tuple[Node, ConnectFour, list[Node]]:
        node = root
        path: list[Node] = []
        while node.expanded and node.children and not state.is_terminal:
            action, child = self._select(node)
            state = state.play(action)
            path.append(child)
            node = child
        return node, state, path

    def _select(self, node: Node) -> tuple[int, Node]:
        parent_scale = np.sqrt(max(1, sum(child.visits for child in node.children.values())))
        best_action = -1
        best_score = -float("inf")
        best_child: Node | None = None
        for action, child in node.children.items():
            exploration = self.c_puct * child.prior * parent_scale / (1 + child.visits)
            score = child.q + exploration
            if score > best_score:
                best_action, best_score, best_child = action, score, child
        assert best_child is not None
        return best_action, best_child

    @staticmethod
    def _softmax_legal(logits: np.ndarray, legal: Sequence[int]) -> np.ndarray:
        probabilities = np.zeros(COLS, dtype=np.float64)
        selected = np.asarray([logits[action] for action in legal], dtype=np.float64)
        selected = np.exp(selected - np.max(selected))
        selected /= selected.sum()
        probabilities[list(legal)] = selected
        return probabilities

    def _expand(self, node: Node, state: ConnectFour, logits: np.ndarray, value: float) -> None:
        if node.expanded:
            return
        probabilities = self._softmax_legal(logits, state.legal_actions)
        node.children = {action: Node(float(probabilities[action])) for action in state.legal_actions}
        node.network_value = value
        node.expanded = True

    def _add_root_noise(self, root: Node) -> None:
        if not root.children:
            return
        actions = list(root.children)
        noise = self.rng.dirichlet([self.dirichlet_alpha] * len(actions))
        for action, sample in zip(actions, noise):
            child = root.children[action]
            child.prior = (1 - self.dirichlet_epsilon) * child.prior + self.dirichlet_epsilon * float(sample)

    @staticmethod
    def _backup(path: Sequence[Node], leaf_value: float) -> None:
        value = float(leaf_value)
        for child in reversed(path):
            value = -value
            child.visits += 1
            child.value_sum += value

    def _result(self, root: Node, snapshots: list[SearchSnapshot]) -> SearchResult:
        visits = np.zeros(COLS, dtype=np.int32)
        q_values = np.zeros(COLS, dtype=np.float32)
        priors = np.zeros(COLS, dtype=np.float32)
        for action, child in root.children.items():
            visits[action] = child.visits
            q_values[action] = child.q
            priors[action] = child.prior
        total = int(visits.sum())
        policy = visits.astype(np.float32) / total if total else priors.copy()
        return SearchResult(
            visits=visits,
            policy=policy,
            q_values=q_values,
            priors=priors,
            root_value=root.network_value,
            simulations=total,
            tree_size=self._tree_size(root),
            principal_variations=self._principal_variations(root),
            snapshots=snapshots,
        )

    def _snapshot(self, root: Node, simulation: int) -> SearchSnapshot:
        result = self._result(root, [])
        return SearchSnapshot(
            simulation=simulation,
            visits=result.visits.tolist(),
            q_values=[float(value) for value in result.q_values],
            priors=[float(value) for value in result.priors],
            root_value=result.root_value,
        )

    @staticmethod
    def _tree_size(root: Node) -> int:
        return 1 + sum(MCTS._tree_size(child) for child in root.children.values())

    @staticmethod
    def _principal_variations(root: Node, top_k: int = 3, depth: int = 8) -> list[list[int]]:
        variations: list[list[int]] = []
        first_actions = sorted(root.children, key=lambda action: root.children[action].visits, reverse=True)[:top_k]
        for first in first_actions:
            line = [first]
            node = root.children[first]
            while node.children and len(line) < depth:
                action = max(node.children, key=lambda candidate: node.children[candidate].visits)
                line.append(action)
                node = node.children[action]
            variations.append(line)
        return variations

    def select_action(self, result: SearchResult, temperature: float) -> int:
        legal = np.flatnonzero(result.visits > 0)
        if not len(legal):
            legal = np.flatnonzero(result.priors > 0)
        if not len(legal):
            raise ValueError("search result has no legal action")
        if temperature <= 1e-6:
            return int(legal[np.argmax(result.visits[legal])])
        weights = result.visits[legal].astype(np.float64) ** (1.0 / temperature)
        if weights.sum() == 0:
            weights = result.priors[legal].astype(np.float64)
        weights /= weights.sum()
        return int(self.rng.choice(legal, p=weights))

    @staticmethod
    def advance_root(root: Node, action: int) -> Node | None:
        return root.children.get(action)

