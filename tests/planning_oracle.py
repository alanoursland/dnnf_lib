"""Deliberately small exhaustive oracle for horizon-two belief policies.

This module does not use ModeNexus inference or planning internals. It is
intended to be obvious rather than fast.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Mapping, Optional, Sequence


State = Mapping[str, object]
Action = object
Distribution = Sequence[tuple[Mapping[str, object], float]]


def _normalized(entries):
    entries = [(value, float(probability)) for value, probability in entries]
    total = sum(probability for _, probability in entries)
    if not entries or abs(total - 1.0) > 1e-9:
        raise ValueError(f"oracle distribution total is {total!r}")
    return entries


def _observation_key(observation: Mapping[str, object]):
    return tuple(sorted(observation.items()))


@dataclass(frozen=True)
class OraclePolicy:
    root_action: Action
    continuations: tuple[tuple[tuple, Action], ...]

    def continuation(self, observation: Mapping[str, object]) -> Action:
        return dict(self.continuations)[_observation_key(observation)]


@dataclass(frozen=True)
class OracleEvaluation:
    policy: OraclePolicy
    goal_probability: float
    expected_action_cost: float
    expected_utility: float
    branch_goal_probabilities: tuple[tuple[tuple, float], ...]

    @property
    def minimum_branch_goal_probability(self) -> Optional[float]:
        if not self.branch_goal_probabilities:
            return None
        return min(value for _, value in self.branch_goal_probabilities)


def enumerate_horizon_two_policies(
    belief: Distribution,
    actions: Sequence[Action],
    outcome_models: Sequence[Callable[[dict, Action], Distribution]],
    observation_model: Callable[[dict, Action], Distribution],
    target: Callable[[dict], bool],
):
    """Enumerate every reachable horizon-two contingent policy."""
    policies = []
    for root_action in actions:
        observations = set()
        for outcome_model in outcome_models:
            for state, state_mass in _normalized(belief):
                if state_mass <= 0.0 or target(dict(state)):
                    continue
                for updates, outcome_probability in _normalized(
                    outcome_model(dict(state), root_action)
                ):
                    if outcome_probability <= 0.0:
                        continue
                    following = dict(state)
                    following.update(updates)
                    if target(following):
                        continue
                    for observation, probability in _normalized(
                        observation_model(following, root_action)
                    ):
                        if probability > 0.0:
                            observations.add(_observation_key(observation))
        ordered = tuple(sorted(observations, key=repr))
        assignments = product(actions, repeat=len(ordered))
        for continuation_actions in assignments:
            policies.append(
                OraclePolicy(
                    root_action=root_action,
                    continuations=tuple(
                        zip(ordered, continuation_actions)
                    ),
                )
            )
    return tuple(policies)


def evaluate_horizon_two_policy(
    policy: OraclePolicy,
    belief: Distribution,
    outcome_model: Callable[[dict, Action], Distribution],
    observation_model: Callable[[dict, Action], Distribution],
    target: Callable[[dict], bool],
    action_cost: Callable[[Action], float],
    *,
    goal_reward: float = 1.0,
    cost_weight: float = 1.0,
) -> OracleEvaluation:
    """Evaluate a fixed policy by direct path enumeration."""
    goal_mass = 0.0
    expected_cost = 0.0
    groups = {}

    for state, state_mass in _normalized(belief):
        state = dict(state)
        if target(state):
            goal_mass += state_mass
            continue
        expected_cost += state_mass * action_cost(policy.root_action)
        for updates, outcome_probability in _normalized(
            outcome_model(state, policy.root_action)
        ):
            following = dict(state)
            following.update(updates)
            outcome_mass = state_mass * outcome_probability
            if target(following):
                goal_mass += outcome_mass
                continue
            for observation, observation_probability in _normalized(
                observation_model(following, policy.root_action)
            ):
                branch_mass = outcome_mass * observation_probability
                key = _observation_key(observation)
                groups.setdefault(key, []).append((following, branch_mass))

    branch_goals = []
    for observation, states in groups.items():
        branch_mass = sum(mass for _, mass in states)
        action = dict(policy.continuations)[observation]
        expected_cost += branch_mass * action_cost(action)
        branch_goal_mass = 0.0
        for state, state_mass in states:
            for updates, probability in _normalized(
                outcome_model(dict(state), action)
            ):
                following = dict(state)
                following.update(updates)
                if target(following):
                    reached = state_mass * probability
                    goal_mass += reached
                    branch_goal_mass += reached
        branch_goals.append(
            (
                observation,
                0.0 if branch_mass == 0.0 else branch_goal_mass / branch_mass,
            )
        )

    return OracleEvaluation(
        policy=policy,
        goal_probability=goal_mass,
        expected_action_cost=expected_cost,
        expected_utility=goal_reward * goal_mass - cost_weight * expected_cost,
        branch_goal_probabilities=tuple(
            sorted(branch_goals, key=lambda item: repr(item[0]))
        ),
    )


def exhaustive_horizon_two(
    belief: Distribution,
    actions: Sequence[Action],
    outcome_model: Callable[[dict, Action], Distribution],
    observation_model: Callable[[dict, Action], Distribution],
    target: Callable[[dict], bool],
    action_cost: Callable[[Action], float],
    *,
    min_goal_probability: Optional[float] = None,
    min_branch_goal_probability: Optional[float] = None,
    goal_reward: float = 1.0,
    cost_weight: float = 1.0,
):
    """Return all evaluations and the oracle-selected feasible policy."""
    policies = enumerate_horizon_two_policies(
        belief,
        actions,
        (outcome_model,),
        observation_model,
        target,
    )
    evaluations = tuple(
        evaluate_horizon_two_policy(
            policy,
            belief,
            outcome_model,
            observation_model,
            target,
            action_cost,
            goal_reward=goal_reward,
            cost_weight=cost_weight,
        )
        for policy in policies
    )

    def feasible(evaluation):
        if (
            min_goal_probability is not None
            and evaluation.goal_probability < min_goal_probability - 1e-9
        ):
            return False
        if min_branch_goal_probability is not None:
            branch_floor = evaluation.minimum_branch_goal_probability
            if (
                branch_floor is None
                or branch_floor < min_branch_goal_probability - 1e-9
            ):
                return False
        return True

    eligible = [evaluation for evaluation in evaluations if feasible(evaluation)]
    if eligible:
        selected = max(
            eligible,
            key=lambda item: (
                item.expected_utility,
                item.goal_probability,
                repr(item.policy),
            ),
        )
    else:
        selected = max(
            evaluations,
            key=lambda item: (
                item.goal_probability,
                item.expected_utility,
                repr(item.policy),
            ),
        )
    return selected, evaluations


def exhaustive_robust_horizon_two(
    belief: Distribution,
    actions: Sequence[Action],
    outcome_models: Mapping[str, Callable[[dict, Action], Distribution]],
    observation_model: Callable[[dict, Action], Distribution],
    target: Callable[[dict], bool],
    action_cost: Callable[[Action], float],
    *,
    min_goal_probability: Optional[float] = None,
    min_branch_goal_probability: Optional[float] = None,
    goal_reward: float = 1.0,
    cost_weight: float = 1.0,
):
    """Enumerate common policies and select the exact tiny maximin result."""
    policies = enumerate_horizon_two_policies(
        belief,
        actions,
        tuple(outcome_models.values()),
        observation_model,
        target,
    )
    audited = []
    for policy in policies:
        by_scenario = {
            name: evaluate_horizon_two_policy(
                policy,
                belief,
                model,
                observation_model,
                target,
                action_cost,
                goal_reward=goal_reward,
                cost_weight=cost_weight,
            )
            for name, model in outcome_models.items()
        }
        audited.append((policy, by_scenario))

    def feasible(item):
        _, by_scenario = item
        for evaluation in by_scenario.values():
            if (
                min_goal_probability is not None
                and evaluation.goal_probability
                < min_goal_probability - 1e-9
            ):
                return False
            if min_branch_goal_probability is not None:
                branch_floor = evaluation.minimum_branch_goal_probability
                if (
                    branch_floor is None
                    or branch_floor < min_branch_goal_probability - 1e-9
                ):
                    return False
        return True

    eligible = [item for item in audited if feasible(item)]
    pool = eligible or audited
    selected = max(
        pool,
        key=lambda item: (
            min(
                evaluation.expected_utility
                for evaluation in item[1].values()
            ),
            min(
                evaluation.goal_probability
                for evaluation in item[1].values()
            ),
            repr(item[0]),
        ),
    )
    return selected, tuple(audited)
