import math

import pytest

import modenexus.diagnosis as diagnosis_module
from modenexus import (
    BeliefActionCertificate,
    ConditionalBeliefPolicyResult,
    ModeNexusInvariantError,
    Planner,
    SystemModel,
)

from planning_oracle import (
    OraclePolicy,
    enumerate_horizon_two_policies,
    evaluate_horizon_two_policy,
    exhaustive_horizon_two,
    exhaustive_robust_horizon_two,
)


ACTIONS = ("none", "probe", "fix_a", "fix_b", "fix_c")
BELIEF = (({"mode": "start"}, 1.0),)
COSTS = {
    "none": 0.0,
    "probe": 0.05,
    "fix_a": 0.05,
    "fix_b": 0.05,
    "fix_c": 0.05,
}


def _compiled_oracle_fixture():
    planner = Planner()
    planner.mode(
        "mode",
        ("start", "a", "b", "c", "goal"),
        priors=(1, 1, 1, 1, 1),
    )
    planner.command("action", ACTIONS)
    for mode in ("a", "b", "c"):
        planner.transition(
            "mode",
            "start",
            mode,
            command=("action", "probe"),
        )
        planner.transition(
            "mode",
            mode,
            "goal",
            command=("action", f"fix_{mode}"),
        )
    return planner.compile(2)


def _outcomes(success_rates):
    def outcome(state, action):
        if state["mode"] == "start" and action == "probe":
            return [
                ({"mode": "a"}, 1 / 3),
                ({"mode": "b"}, 1 / 3),
                ({"mode": "c"}, 1 / 3),
            ]
        mode = state["mode"]
        if mode in ("a", "b", "c") and action == f"fix_{mode}":
            probability = success_rates[mode]
            return [
                ({"mode": "goal"}, probability),
                ({}, 1.0 - probability),
            ]
        return [({}, 1.0)]

    return outcome


def _observations(state, action):
    return [({"signal": state["mode"]}, 1.0)]


def _target(state):
    return state["mode"] == "goal"


def _as_planner_outcomes(outcome):
    return lambda state, command: outcome(state, command["action"])


def _as_planner_observations(state, command):
    return _observations(state, command["action"])


def _oracle_policy_for_result(result, policies):
    root_action = result.action["action"]
    for candidate in policies:
        if candidate.root_action != root_action:
            continue
        matched = True
        for observation, action in candidate.continuations:
            routed = result.policy.route(dict(observation))
            if (
                routed.policy is None
                or routed.policy.action["action"] != action
            ):
                matched = False
                break
        if matched:
            return candidate
    raise AssertionError("planner policy was absent from exhaustive policy set")


def test_exact_conditional_planner_matches_exhaustive_oracle():
    outcome = _outcomes({"a": 1.0, "b": 1.0, "c": 1.0})
    selected, _ = exhaustive_horizon_two(
        BELIEF,
        ACTIONS,
        outcome,
        _observations,
        _target,
        lambda action: COSTS[action],
    )
    result = _compiled_oracle_fixture().plan_belief(
        BELIEF,
        {"mode": "goal"},
        outcome_model=_as_planner_outcomes(outcome),
        observation_model=_as_planner_observations,
        action_costs=COSTS,
    )
    assert isinstance(result, ConditionalBeliefPolicyResult)
    assert result.action["action"] == selected.policy.root_action
    assert result.expected_goal_probability == pytest.approx(
        selected.goal_probability
    )
    assert result.expected_utility == pytest.approx(
        selected.expected_utility
    )


def test_constraints_and_pruned_bounds_match_direct_oracle_evaluation():
    outcome = _outcomes({"a": 1.0, "b": 1.0, "c": 1.0})
    compiled = _compiled_oracle_fixture()
    selected, evaluations = exhaustive_horizon_two(
        BELIEF,
        ACTIONS,
        outcome,
        _observations,
        _target,
        lambda action: COSTS[action],
        min_goal_probability=0.9,
        min_branch_goal_probability=0.9,
    )
    policies = tuple(evaluation.policy for evaluation in evaluations)
    exact = compiled.plan_belief(
        BELIEF,
        {"mode": "goal"},
        outcome_model=_as_planner_outcomes(outcome),
        observation_model=_as_planner_observations,
        action_costs=COSTS,
        min_goal_probability=0.9,
        min_branch_goal_probability=0.9,
    )
    assert exact.expected_utility == pytest.approx(selected.expected_utility)
    assert exact.constraint_certification == "certified-feasible"

    pruned = compiled.plan_belief(
        BELIEF,
        {"mode": "goal"},
        outcome_model=_as_planner_outcomes(outcome),
        observation_model=_as_planner_observations,
        action_costs=COSTS,
        max_observations_per_node=1,
    )
    policy = _oracle_policy_for_result(pruned, policies)
    direct = evaluate_horizon_two_policy(
        policy,
        BELIEF,
        outcome,
        _observations,
        _target,
        lambda action: COSTS[action],
    )
    assert pruned.expected_goal_probability == pytest.approx(
        direct.goal_probability
    )
    assert pruned.expected_utility == pytest.approx(direct.expected_utility)
    assert pruned.utility_lower_bound <= selected.expected_utility
    assert pruned.optimal_utility_upper_bound >= selected.expected_utility


def test_robust_scenario_metrics_match_exhaustive_oracle():
    scenarios = {
        "left": _outcomes({"a": 1.0, "b": 0.6, "c": 0.2}),
        "right": _outcomes({"a": 0.2, "b": 0.6, "c": 1.0}),
    }
    (oracle_policy, oracle_scenarios), policies = (
        exhaustive_robust_horizon_two(
            BELIEF,
            ACTIONS,
            scenarios,
            _observations,
            _target,
            lambda action: COSTS[action],
        )
    )
    result = _compiled_oracle_fixture().plan_belief(
        BELIEF,
        {"mode": "goal"},
        outcome_scenarios={
            name: _as_planner_outcomes(model)
            for name, model in scenarios.items()
        },
        robust_objective="maximin",
        robust_weight_resolution=4,
        max_robust_candidates=10,
        observation_model=_as_planner_observations,
        action_costs=COSTS,
    )
    assert result.worst_case_expected_utility == pytest.approx(
        min(
            evaluation.expected_utility
            for evaluation in oracle_scenarios.values()
        )
    )
    assert result.worst_case_goal_probability == pytest.approx(
        min(
            evaluation.goal_probability
            for evaluation in oracle_scenarios.values()
        )
    )
    assert result.action["action"] == oracle_policy.root_action

    planner_policy = _oracle_policy_for_result(
        result,
        tuple(policy for policy, _ in policies),
    )
    reported = {
        item.scenario: item for item in result.robust_scenario_evaluations
    }
    for name, outcome in scenarios.items():
        direct = evaluate_horizon_two_policy(
            planner_policy,
            BELIEF,
            outcome,
            _observations,
            _target,
            lambda action: COSTS[action],
        )
        assert reported[name].expected_goal_probability == pytest.approx(
            direct.goal_probability
        )
        assert reported[name].expected_utility == pytest.approx(
            direct.expected_utility
        )


def test_work_estimate_telemetry_and_grouped_result_views():
    compiled = _compiled_oracle_fixture()
    estimate = compiled.estimate_belief_work(
        3,
        actions=ACTIONS,
        conditional=True,
        outcome_branching_hint=3,
        observation_branching_hint=1,
        max_policy_nodes=1000,
    )
    assert estimate.action_sequences == len(ACTIONS) ** 2
    assert estimate.estimated_policy_nodes == len(ACTIONS) * (1 + 3)
    assert estimate.estimated_outcome_branches == 180

    outcome = _outcomes({"a": 1.0, "b": 1.0, "c": 1.0})
    result = compiled.plan_belief(
        BELIEF,
        {"mode": "goal"},
        outcome_model=_as_planner_outcomes(outcome),
        observation_model=_as_planner_observations,
        action_costs=COSTS,
    )
    assert result.work is not None
    assert result.work.action_evaluations == result.policy_node_count
    assert result.work.goal_probability_queries > 0
    assert result.work.outcome_callback_calls > 0
    assert result.work.measured_work_units > 0
    assert result.work.seconds_per_work_unit >= 0.0
    assert result.certificate.scope == result.certificate_scope
    assert result.approximation_details.kind == result.approximation
    assert result.diagnostics.work is result.work


def test_postconditions_raise_dedicated_invariant_error(monkeypatch):
    model = SystemModel()
    model.mode("mode", ("ok", "bad"), (0.5, 0.5))
    compiled = model.compile()
    values = iter((0.0, math.log(9.0), -math.inf))
    monkeypatch.setattr(
        diagnosis_module.fd,
        "log_wmc",
        lambda circuit, weights: next(values),
    )
    with pytest.raises(ModeNexusInvariantError, match="posterior probability"):
        compiled.mode_posteriors({})

    with pytest.raises(ModeNexusInvariantError, match="lower bound"):
        BeliefActionCertificate(
            action={"action": "bad"},
            policy_utility_lower_bound=2.0,
            policy_utility_upper_bound=1.0,
            utility_lower_bound=0.0,
            utility_upper_bound=1.0,
        )
