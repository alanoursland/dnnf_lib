import math

import pytest

from modenexus import (
    PlanControl,
    Planner,
    PlanningBudgetExceeded,
    PlanningCancelled,
    TrackedBelief,
    iff,
)


def repair_planner(horizon=1):
    planner = Planner()
    planner.mode("mode", ("a", "b", "goal"), priors=(0.45, 0.45, 0.1))
    planner.command("action", ("none", "fix_a", "fix_b"))
    planner.observable("done")
    planner.behavior(
        lambda value: iff(value["done"] == True, value["mode"] == "goal")
    )
    planner.transition(
        "mode", "a", "goal", command=("action", "fix_a")
    )
    planner.transition(
        "mode", "b", "goal", command=("action", "fix_b")
    )
    return planner.compile(horizon)


def repair_outcomes(state, command):
    action = command["action"]
    if (
        (state["mode"] == "a" and action == "fix_a")
        or (state["mode"] == "b" and action == "fix_b")
    ):
        return [({"mode": "goal"}, 1.0)]
    return [({}, 1.0)]


def test_plan_belief_maximizes_expected_utility():
    planner = repair_planner()
    result = planner.plan_belief(
        belief=[({"mode": "a"}, 4.0), ({"mode": "b"}, 6.0)],
        target={"done": True},
        action_costs={"none": 0.0, "fix_a": 0.0, "fix_b": 0.2},
        outcome_model=repair_outcomes,
        cost_weight=0.5,
    )
    assert result.action == {"action": "fix_b"}
    assert result.commands == [{"action": "fix_b"}]
    assert result.expected_goal_probability == pytest.approx(0.6)
    assert result.expected_utility == pytest.approx(0.5)
    by_action = {
        item.action["action"]: item for item in result.evaluations
    }
    assert by_action["fix_a"].expected_goal_probability == pytest.approx(0.4)
    assert by_action["fix_b"].expected_goal_probability == pytest.approx(0.6)
    assert by_action["none"].expected_goal_probability == pytest.approx(0.0)


def test_plan_belief_preserves_joint_correlations():
    planner = Planner()
    planner.mode("left", ("off", "on"), priors=(0.5, 0.5))
    planner.mode("right", ("off", "on"), priors=(0.5, 0.5))
    planner.command("action", ("none", "flip_left"))
    planner.observable("same")
    planner.behavior(
        lambda value: iff(
            value["same"] == True,
            ((value["left"] == "off") & (value["right"] == "off"))
            | ((value["left"] == "on") & (value["right"] == "on")),
        )
    )
    planner.transition(
        "left", "off", "on", command=("action", "flip_left")
    )
    planner.transition(
        "left", "on", "off", command=("action", "flip_left")
    )
    compiled = planner.compile(1)

    def outcomes(state, command):
        if command["action"] == "flip_left":
            changed = "on" if state["left"] == "off" else "off"
            return [({"left": changed}, 1.0)]
        return [({}, 1.0)]

    result = compiled.plan_belief(
        belief=[
            ({"left": "off", "right": "off"}, 0.5),
            ({"left": "on", "right": "on"}, 0.5),
        ],
        target={"same": True},
        outcome_model=outcomes,
    )
    assert result.action == {"action": "none"}
    assert result.expected_goal_probability == pytest.approx(1.0)
    assert result.evaluations[1].expected_goal_probability == pytest.approx(0.0)


def test_plan_belief_uses_compiled_transition_weights_by_default():
    planner = Planner()
    planner.mode("mode", ("bad", "good"), priors=(0.9, 0.1))
    planner.command("action", ("none", "try"))
    planner.observable("done")
    planner.behavior(
        lambda value: iff(value["done"] == True, value["mode"] == "good")
    )
    # Relative outcome weight 0.25 against the implicit noop weight 1.0:
    # P(success | try, bad) = 0.25 / 1.25 = 0.2.
    planner.transition(
        "mode",
        "bad",
        "good",
        command=("action", "try"),
        cost=-math.log(0.25),
    )
    result = planner.compile(1).plan_belief(
        belief=[({"mode": "bad"}, 1.0)],
        target={"done": True},
        actions=["try"],
    )
    assert result.expected_goal_probability == pytest.approx(0.2)


def test_plan_belief_selects_multistep_prerequisite_sequence():
    planner = Planner()
    planner.mode("mode", ("start", "ready", "goal"), priors=(1, 1, 1))
    planner.command("action", ("none", "prepare", "finish"))
    planner.observable("done")
    planner.behavior(
        lambda value: iff(value["done"] == True, value["mode"] == "goal")
    )
    planner.transition(
        "mode", "start", "ready", command=("action", "prepare")
    )
    planner.transition(
        "mode", "ready", "goal", command=("action", "finish")
    )

    def outcomes(state, command):
        if state["mode"] == "start" and command["action"] == "prepare":
            return [({"mode": "ready"}, 1.0)]
        if state["mode"] == "ready" and command["action"] == "finish":
            return [({"mode": "goal"}, 1.0)]
        return [({}, 1.0)]

    result = planner.compile(horizon=2).plan_belief(
        belief=[({"mode": "start"}, 1.0)],
        target={"done": True},
        outcome_model=outcomes,
        action_costs={"none": 0.0, "prepare": 0.1, "finish": 0.1},
        cost_weight=0.1,
    )
    assert [command["action"] for command in result.commands] == [
        "prepare",
        "finish",
    ]
    assert result.action == {"action": "prepare"}
    assert result.expected_goal_probability == pytest.approx(1.0)
    assert result.expected_utility == pytest.approx(0.98)
    assert len(result.evaluations) == 9
    assert not result.observation_branching


def test_plan_belief_multistep_has_explicit_sequence_budget():
    with pytest.raises(ValueError, match="action sequences"):
        repair_planner(horizon=3).plan_belief(
            belief=[({"mode": "a"}, 1.0)],
            target={"done": True},
            max_action_sequences=10,
        )


def test_plan_belief_selects_observation_contingent_first_action():
    planner = Planner()
    planner.mode(
        "mode",
        ("a", "a_ready", "b", "goal"),
        priors=(1, 1, 1, 1),
    )
    planner.command("action", ("none", "prepare", "fix_a", "fix_b"))
    planner.observable("done")
    planner.behavior(
        lambda value: iff(value["done"] == True, value["mode"] == "goal")
    )
    planner.transition(
        "mode", "a", "a_ready", command=("action", "prepare")
    )
    planner.transition(
        "mode", "a_ready", "goal", command=("action", "fix_a")
    )
    planner.transition(
        "mode", "b", "goal", command=("action", "fix_b")
    )

    def outcomes(state, command):
        action = command["action"]
        if state["mode"] == "a" and action == "prepare":
            return [({"mode": "a_ready"}, 1.0)]
        if state["mode"] == "a_ready" and action == "fix_a":
            return [({"mode": "goal"}, 1.0)]
        if state["mode"] == "b" and action == "fix_b":
            return [({"mode": "goal"}, 1.0)]
        return [({}, 1.0)]

    result = planner.compile(horizon=2).plan_belief(
        belief=[({"mode": "a"}, 0.5), ({"mode": "b"}, 0.5)],
        target={"done": True},
        outcome_model=outcomes,
        observation_model=lambda state, command: {
            "mode": state["mode"]
        },
        action_costs={
            "none": 0.0,
            "prepare": 0.1,
            "fix_a": 0.1,
            "fix_b": 0.1,
        },
        cost_weight=0.1,
    )

    assert result.observation_branching
    assert result.action == {"action": "prepare"}
    assert result.expected_goal_probability == pytest.approx(1.0)
    assert result.action_cost == pytest.approx(0.2)
    assert result.expected_utility == pytest.approx(0.98)
    assert result.policy_node_count > 0
    assert result.outcome_branch_count > 0
    assert result.observation_branch_count > 0
    continuations = {
        branch.observation["mode"]: (
            None if branch.policy is None
            else branch.policy.action["action"]
        )
        for branch in result.policy.branches
    }
    assert continuations == {"a_ready": "fix_a", "b": "fix_b"}


def test_plan_belief_conditional_policy_has_explicit_node_budget():
    with pytest.raises(ValueError, match="max_policy_nodes"):
        repair_planner(horizon=2).plan_belief(
            belief=[({"mode": "a"}, 1.0)],
            target={"done": True},
            outcome_model=repair_outcomes,
            observation_model=lambda state, command: dict(state),
            max_policy_nodes=2,
        )


def test_plan_belief_conditional_policy_requires_outcome_model():
    with pytest.raises(ValueError, match="explicit outcome_model"):
        repair_planner(horizon=2).plan_belief(
            belief=[({"mode": "a"}, 1.0)],
            target={"done": True},
            observation_model=lambda state, command: dict(state),
        )


def test_plan_belief_prunes_rare_observations_into_fallback():
    planner = repair_planner(horizon=2)

    def observations(state, command):
        del command
        if state["mode"] == "goal":
            return {"signal": "goal"}
        expected = state["mode"]
        other = "b" if expected == "a" else "a"
        return [
            ({"signal": expected}, 0.99),
            ({"signal": other}, 0.01),
        ]

    result = planner.plan_belief(
        belief=[({"mode": "a"}, 0.5), ({"mode": "b"}, 0.5)],
        target={"done": True},
        outcome_model=repair_outcomes,
        observation_model=observations,
        action_costs={"none": 0.0, "fix_a": 0.0, "fix_b": 0.2},
        cost_weight=0.1,
        min_observation_probability=0.01,
    )

    assert result.action == {"action": "fix_a"}
    assert result.approximation == "observation-pruned"
    assert result.action_ranking == "certified"
    assert result.utility_is_lower_bound
    assert result.root_action_certified
    assert result.maximum_regret == pytest.approx(0.0)
    assert result.utility_lower_bound <= result.utility_upper_bound
    assert result.optimal_utility_upper_bound >= (
        result.utility_lower_bound
    )
    assert result.retained_observation_probability == pytest.approx(0.995)
    assert result.discarded_observation_probability == pytest.approx(0.005)
    assert result.pruned_observation_branch_count > 0
    assert result.generated_observation_branch_count >= (
        result.observation_branch_count
    )
    assert result.policy.fallback_policy is not None
    assert result.policy.continuation(
        {"signal": "a"}
    ) is result.policy.fallback_policy

    aggressive = planner.plan_belief(
        belief=[({"mode": "a"}, 0.5), ({"mode": "b"}, 0.5)],
        target={"done": True},
        outcome_model=repair_outcomes,
        observation_model=observations,
        action_costs={"none": 0.0, "fix_a": 0.0, "fix_b": 0.2},
        cost_weight=0.1,
        max_observations_per_node=1,
    )
    assert aggressive.action_ranking == "heuristic"
    assert not aggressive.root_action_certified
    assert aggressive.maximum_regret == pytest.approx(0.005)
    assert aggressive.optimal_utility_upper_bound == pytest.approx(1.0)

    tracked = planner.plan_belief(
        belief=TrackedBelief(
            [({"mode": "a"}, 0.5), ({"mode": "b"}, 0.5)],
            exact=False,
            retained_probability_mass=None,
        ),
        target={"done": True},
        outcome_model=repair_outcomes,
        observation_model=observations,
        action_costs={"none": 0.0, "fix_a": 0.0, "fix_b": 0.2},
        cost_weight=0.1,
        min_observation_probability=0.01,
    )
    assert tracked.policy_root_action_certified
    assert not tracked.root_action_certified
    assert tracked.action_ranking == "heuristic"
    assert tracked.maximum_regret > 0.0
    assert tracked.belief_exact is False
    assert tracked.belief_retained_probability_mass is None
    assert tracked.certificate_scope == "tracked-belief-mass-unknown"

    bounded = planner.plan_belief(
        belief=TrackedBelief(
            [({"mode": "a"}, 0.5), ({"mode": "b"}, 0.5)],
            exact=False,
            retained_probability_mass=0.99,
        ),
        target={"done": True},
        outcome_model=repair_outcomes,
        observation_model=observations,
        action_costs={"none": 0.0, "fix_a": 0.0, "fix_b": 0.2},
        cost_weight=0.1,
        min_observation_probability=0.01,
    )
    assert bounded.certificate_scope == "tracked-belief-mass-bound"
    assert bounded.belief_retained_probability_mass == pytest.approx(0.99)
    assert bounded.maximum_regret < tracked.maximum_regret


def test_plan_belief_skips_observations_after_final_action():
    calls = []

    def observations(state, command):
        calls.append((dict(state), dict(command)))
        return {"mode": state["mode"]}

    result = repair_planner(horizon=2).plan_belief(
        belief=TrackedBelief(
            [({"mode": "a"}, 0.5), ({"mode": "b"}, 0.5)],
            exact=True,
            retained_probability_mass=1.0,
        ),
        target={"done": True},
        outcome_model=repair_outcomes,
        observation_model=observations,
    )

    assert result.approximation == "exact"
    assert result.action_ranking == "exact"
    assert not result.utility_is_lower_bound
    assert result.root_action_certified
    assert result.maximum_regret == pytest.approx(0.0)
    assert result.utility_lower_bound == pytest.approx(
        result.utility_upper_bound
    )
    assert result.belief_exact is True
    assert result.belief_retained_probability_mass == pytest.approx(1.0)
    assert result.certificate_scope == "exact-tracker-belief"
    assert result.retained_observation_probability == pytest.approx(1.0)
    assert len(calls) == 6
    assert all(not branch.policy.branches for branch in result.policy.branches)


@pytest.mark.parametrize(
    "belief",
    [
        [],
        [({"mode": "a"}, -0.1)],
        [({"mode": "a"}, math.inf)],
        [({"unrelated": "value"}, 1.0)],
    ],
)
def test_plan_belief_validates_joint_belief(belief):
    with pytest.raises(ValueError):
        repair_planner().plan_belief(
            belief=belief,
            target={"done": True},
        )


def reliability_planner(horizon=1):
    planner = Planner()
    planner.mode("mode", ("bad", "goal"), priors=(0.9, 0.1))
    planner.command("action", ("none", "cheap", "sure"))
    planner.observable("done")
    planner.behavior(
        lambda value: iff(value["done"] == True, value["mode"] == "goal")
    )
    planner.transition("mode", "bad", "goal", command=("action", "cheap"))
    planner.transition("mode", "bad", "goal", command=("action", "sure"))
    return planner.compile(horizon)


def reliability_outcomes(state, command):
    action = command["action"]
    if state["mode"] == "bad" and action == "cheap":
        return [({"mode": "goal"}, 0.85), ({}, 0.15)]
    if state["mode"] == "bad" and action == "sure":
        return [({"mode": "goal"}, 0.95), ({}, 0.05)]
    return [({}, 1.0)]


RELIABILITY_COSTS = {"none": 0.0, "cheap": 0.0, "sure": 0.5}


def test_one_step_reliability_floor_overrides_utility():
    planner = reliability_planner()
    arguments = dict(
        belief=[({"mode": "bad"}, 1.0)],
        target={"done": True},
        outcome_model=reliability_outcomes,
        action_costs=RELIABILITY_COSTS,
        cost_weight=0.5,
    )
    unconstrained = planner.plan_belief(**arguments)
    assert unconstrained.action == {"action": "cheap"}
    assert unconstrained.feasible is None

    floored = planner.plan_belief(min_goal_probability=0.9, **arguments)
    assert floored.action == {"action": "sure"}
    assert floored.feasible
    assert floored.goal_probability_constraint == 0.9
    assert floored.expected_goal_probability == pytest.approx(0.95)

    impossible = planner.plan_belief(
        min_goal_probability=0.99, **arguments
    )
    assert not impossible.feasible
    assert impossible.action == {"action": "sure"}
    assert impossible.best_achievable_goal_probability == pytest.approx(
        0.95
    )


def test_open_loop_reliability_floor():
    planner = reliability_planner(horizon=2)
    arguments = dict(
        belief=[({"mode": "bad"}, 1.0)],
        target={"done": True},
        outcome_model=reliability_outcomes,
        action_costs=RELIABILITY_COSTS,
        cost_weight=0.5,
    )
    floored = planner.plan_belief(min_goal_probability=0.99, **arguments)
    assert floored.feasible
    assert floored.expected_goal_probability >= 0.99 - 1e-9

    impossible = planner.plan_belief(
        min_goal_probability=0.999, **arguments
    )
    assert not impossible.feasible
    assert impossible.best_achievable_goal_probability == pytest.approx(
        0.95 + 0.05 * 0.95
    )


def staged_repair_planner():
    """Fixes succeed only while ``fresh``; retries are worthless."""
    planner = Planner()
    planner.mode("mode", ("a", "b", "goal"), priors=(0.45, 0.45, 0.1))
    planner.mode("fresh", ("yes", "no"))
    planner.command(
        "action", ("none", "fix_cheap", "fix_sure", "fix_b")
    )
    planner.observable("done")
    planner.behavior(
        lambda value: iff(value["done"] == True, value["mode"] == "goal")
    )
    for fix in ("fix_cheap", "fix_sure", "fix_b"):
        planner.transition("fresh", "yes", "no", command=("action", fix))
    planner.transition(
        "mode", "a", "goal", command=("action", "fix_cheap")
    )
    planner.transition(
        "mode", "a", "goal", command=("action", "fix_sure")
    )
    planner.transition("mode", "b", "goal", command=("action", "fix_b"))
    return planner.compile(2)


def staged_outcomes(state, command):
    action = command["action"]
    if action == "none":
        return [({}, 1.0)]
    if state["fresh"] == "yes":
        if state["mode"] == "a" and action == "fix_cheap":
            return [
                ({"mode": "goal", "fresh": "no"}, 0.9),
                ({"fresh": "no"}, 0.1),
            ]
        if state["mode"] == "a" and action == "fix_sure":
            return [({"mode": "goal", "fresh": "no"}, 1.0)]
        if state["mode"] == "b" and action == "fix_b":
            return [
                ({"mode": "goal", "fresh": "no"}, 0.8),
                ({"fresh": "no"}, 0.2),
            ]
    return [({"fresh": "no"}, 1.0)]


STAGED_BELIEF = [
    ({"mode": "a", "fresh": "yes"}, 0.5),
    ({"mode": "b", "fresh": "yes"}, 0.5),
]
STAGED_COSTS = {
    "none": 0.0,
    "fix_cheap": 0.1,
    "fix_sure": 1.0,
    "fix_b": 0.1,
}


def staged_arguments(**overrides):
    arguments = dict(
        belief=STAGED_BELIEF,
        target={"done": True},
        outcome_model=staged_outcomes,
        observation_model=lambda state, command: {
            "mode": state["mode"]
        },
        action_costs=STAGED_COSTS,
        cost_weight=0.5,
    )
    arguments.update(overrides)
    return arguments


def branch_action(policy, mode):
    branch = next(
        branch for branch in policy.branches
        if branch.observation == {"mode": mode}
    )
    return branch.policy.action["action"]


def test_reliability_floor_composes_across_observation_branches():
    planner = staged_repair_planner()
    unconstrained = planner.plan_belief(**staged_arguments())
    # Utility alone picks the cheap repair in branch a: 0.85 overall.
    assert unconstrained.expected_goal_probability == pytest.approx(0.85)
    assert branch_action(unconstrained.policy, "a") == "fix_cheap"

    # Branch b's ceiling is 0.8, below the floor — a per-node threshold
    # would wrongly report infeasibility.  The whole-policy constraint is
    # met by buying reliability in branch a instead.
    floored = planner.plan_belief(
        min_goal_probability=0.9, **staged_arguments()
    )
    assert floored.feasible
    assert floored.expected_goal_probability == pytest.approx(0.9)
    assert branch_action(floored.policy, "a") == "fix_sure"
    assert branch_action(floored.policy, "b") == "fix_b"
    assert floored.goal_probability_constraint == 0.9
    assert floored.constraint_certification == "certified-feasible"
    assert floored.constraint_optimality == "exact"

    impossible = planner.plan_belief(
        min_goal_probability=0.95, **staged_arguments()
    )
    assert not impossible.feasible
    assert impossible.best_achievable_goal_probability == pytest.approx(
        0.9
    )
    assert impossible.constraint_certification == "certified-infeasible"
    # Infeasible results still return the most reliable policy.
    assert impossible.expected_goal_probability == pytest.approx(0.9)


def test_branch_reliability_floor_is_stricter():
    planner = staged_repair_planner()
    relaxed = planner.plan_belief(
        min_branch_goal_probability=0.75, **staged_arguments()
    )
    assert relaxed.feasible
    assert relaxed.constraint_certification == "certified-feasible"
    assert relaxed.branch_goal_probability_constraint == 0.75
    for branch in relaxed.policy.branches:
        assert branch.expected_goal_probability >= 0.75 - 1e-9

    # No continuation for branch b can reach 0.9, so the safety variant is
    # genuinely infeasible even though the root constraint above was not.
    strict = planner.plan_belief(
        min_branch_goal_probability=0.9, **staged_arguments()
    )
    assert not strict.feasible
    assert strict.constraint_certification == "certified-infeasible"
    assert strict.best_achievable_goal_probability == pytest.approx(0.9)


def test_frontier_truncation_keeps_feasibility_exact():
    planner = staged_repair_planner()
    snapshots = []
    result = planner.plan_belief(
        min_goal_probability=0.9,
        max_frontier_points=2,
        control=PlanControl(
            progress=snapshots.append,
            progress_interval=0.0,
        ),
        **staged_arguments(),
    )
    assert result.feasible
    assert result.expected_goal_probability >= 0.9 - 1e-9
    assert result.constraint_optimality == "frontier-truncated"
    assert result.generated_frontier_points >= (
        result.retained_frontier_points
    )
    assert result.maximum_frontier_size > 2
    assert result.truncated_frontier_nodes > 0
    assert result.frontier_saturation_by_depth
    assert result.frontier_saturated_root_actions
    assert result.constraint_utility_upper_bound >= (
        result.expected_utility
    )
    assert result.constraint_utility_optimality_gap == pytest.approx(
        result.constraint_utility_upper_bound - result.expected_utility
    )
    final = snapshots[-1]
    assert final.complete
    assert final.generated_frontier_points == (
        result.generated_frontier_points
    )
    assert final.retained_frontier_points == (
        result.retained_frontier_points
    )
    assert final.maximum_frontier_size == result.maximum_frontier_size
    assert final.truncated_frontier_nodes == (
        result.truncated_frontier_nodes
    )


def test_pruned_branch_floor_certificate_is_indeterminate():
    result = staged_repair_planner().plan_belief(
        min_branch_goal_probability=0.75,
        max_observations_per_node=1,
        **staged_arguments(),
    )
    assert result.pruned_observation_branch_count > 0
    assert result.constraint_certification == "indeterminate"


def test_best_achievable_probability_reports_coarsening_scope_and_bound():
    pruned = staged_repair_planner().plan_belief(
        min_goal_probability=0.99,
        max_observations_per_node=1,
        **staged_arguments(),
    )
    assert not pruned.feasible
    assert pruned.constraint_optimality == "exact"
    assert (
        pruned.best_achievable_goal_probability_scope
        == "selected-observation-coarsening"
    )
    assert pruned.observation_partition_optimality == "heuristic-pruned"
    assert pruned.best_achievable_goal_probability_upper_bound >= (
        pruned.best_achievable_goal_probability
    )

    unpruned = staged_repair_planner().plan_belief(
        min_goal_probability=0.99,
        **staged_arguments(),
    )
    assert (
        unpruned.best_achievable_goal_probability_scope
        == "full-observation-policy-space"
    )
    assert unpruned.observation_partition_optimality == "exact"


def test_scenario_robust_maximin_reports_per_scenario_metrics():
    def favorable(state, command):
        action = command["action"]
        if state["mode"] == "bad" and action in {"cheap", "sure"}:
            success = 0.99 if action == "cheap" else 0.60
            return [({"mode": "goal"}, success), ({}, 1 - success)]
        return [({}, 1.0)]

    def adverse(state, command):
        action = command["action"]
        if state["mode"] == "bad" and action in {"cheap", "sure"}:
            success = 0.10 if action == "cheap" else 0.80
            return [({"mode": "goal"}, success), ({}, 1 - success)]
        return [({}, 1.0)]

    result = reliability_planner(horizon=2).plan_belief(
        belief=[({"mode": "bad"}, 1.0)],
        target={"done": True},
        outcome_scenarios={
            "favorable": favorable,
            "adverse": adverse,
        },
        robust_objective="maximin",
        robust_weight_resolution=4,
        max_robust_candidates=10,
        observation_model=lambda state, command: {
            "mode": state["mode"]
        },
        action_costs=RELIABILITY_COSTS,
        cost_weight=0.5,
        min_goal_probability=0.75,
    )

    assert result.robust_objective == "maximin"
    assert result.robust_optimality == "weight-grid-heuristic"
    assert result.certificate_scope == "robust-scenario-weight-grid"
    assert result.inherited_metric_scope == (
        "scenario-weighted-candidate-generation"
    )
    assert dict(result.selected_scenario_weights) == pytest.approx(
        {"favorable": 0.25, "adverse": 0.75}
    )
    assert result.robust_candidate_count >= 1
    assert len(result.robust_scenario_evaluations) == 2
    assert result.worst_case_scenario in {"favorable", "adverse"}
    assert result.worst_case_expected_utility == pytest.approx(
        min(
            item.expected_utility
            for item in result.robust_scenario_evaluations
        )
    )
    assert result.worst_case_goal_probability == pytest.approx(
        min(
            item.expected_goal_probability
            for item in result.robust_scenario_evaluations
        )
    )
    scenario_metrics = {
        item.scenario: item
        for item in result.robust_scenario_evaluations
    }
    assert scenario_metrics["favorable"].expected_action_cost == (
        pytest.approx(0.005)
    )
    assert scenario_metrics["adverse"].expected_action_cost == (
        pytest.approx(0.45)
    )
    assert result.robust_feasible
    assert result.constraint_certification == "certified-feasible"
    assert not result.root_action_certified
    assert result.worst_case_goal_probability == pytest.approx(
        result.robust_best_achievable_min_goal_probability
    )


def test_scenario_robust_branch_floor_is_audited_per_scenario():
    def favorable(state, command):
        action = command["action"]
        if state["mode"] == "bad" and action in {"cheap", "sure"}:
            success = 0.99 if action == "cheap" else 0.60
            return [({"mode": "goal"}, success), ({}, 1 - success)]
        return [({}, 1.0)]

    def adverse(state, command):
        action = command["action"]
        if state["mode"] == "bad" and action in {"cheap", "sure"}:
            success = 0.10 if action == "cheap" else 0.80
            return [({"mode": "goal"}, success), ({}, 1 - success)]
        return [({}, 1.0)]

    result = reliability_planner(horizon=2).plan_belief(
        belief=[({"mode": "bad"}, 1.0)],
        target={"done": True},
        outcome_scenarios={
            "favorable": favorable,
            "adverse": adverse,
        },
        robust_objective="maximin",
        robust_weight_resolution=4,
        max_robust_candidates=10,
        observation_model=lambda state, command: {
            "mode": state["mode"]
        },
        action_costs=RELIABILITY_COSTS,
        cost_weight=0.5,
        min_goal_probability=0.75,
        min_branch_goal_probability=0.75,
    )

    assert not result.robust_feasible
    assert not result.robust_branch_feasible
    assert result.worst_case_branch_scenario == "favorable"
    assert result.worst_case_minimum_branch_goal_probability == (
        pytest.approx(0.60)
    )
    assert result.robust_best_achievable_min_branch_goal_probability == (
        pytest.approx(0.60)
    )
    assert result.worst_case_branch_observation_path == (
        {"mode": "bad"},
    )
    with pytest.raises(TypeError):
        result.worst_case_branch_observation_path[0]["mode"] = "goal"
    assert result.robust_branch_constraint_certification == "indeterminate"
    assert result.constraint_certification == "indeterminate"

    pruned = reliability_planner(horizon=2).plan_belief(
        belief=[({"mode": "bad"}, 1.0)],
        target={"done": True},
        outcome_scenarios={
            "favorable": favorable,
            "adverse": adverse,
        },
        robust_objective="maximin",
        robust_weight_resolution=4,
        max_robust_candidates=10,
        observation_model=lambda state, command: {
            "mode": state["mode"]
        },
        action_costs=RELIABILITY_COSTS,
        cost_weight=0.5,
        min_goal_probability=0.75,
        min_branch_goal_probability=0.50,
        max_observations_per_node=1,
    )
    assert pruned.robust_feasible
    assert pruned.robust_branch_feasible
    assert (
        pruned.robust_branch_constraint_certification
        == "indeterminate"
    )
    assert pruned.constraint_certification == "indeterminate"
    assert result.worst_case_goal_probability == pytest.approx(
        result.robust_best_achievable_min_goal_probability
    )
    assert result.worst_case_goal_probability == pytest.approx(
        result.robust_best_achievable_min_goal_probability
    )


def test_scenario_robust_validation():
    with pytest.raises(ValueError, match="either outcome_model"):
        reliability_planner(horizon=2).plan_belief(
            belief=[({"mode": "bad"}, 1.0)],
            target={"done": True},
            outcome_model=reliability_outcomes,
            outcome_scenarios={
                "a": reliability_outcomes,
                "b": reliability_outcomes,
            },
            robust_objective="maximin",
            observation_model=lambda state, command: dict(state),
        )


def test_policy_execution_updates_terminal_hidden_posterior():
    planner = Planner()
    planner.mode("battery", ("low", "ready"))
    planner.mode("regime", ("good", "bad"), priors=(0.5, 0.5))
    planner.command("action", ("none", "charge"))
    planner.observable("done")
    planner.behavior(
        lambda value: iff(
            value["done"] == True,
            value["battery"] == "ready",
        )
    )
    planner.transition(
        "battery", "low", "ready", command=("action", "charge")
    )
    compiled = planner.compile(2)

    def outcomes(state, command):
        success = 0.9 if state["regime"] == "good" else 0.2
        return [
            ({"battery": "ready"}, success),
            ({}, 1.0 - success),
        ]

    belief = [
        ({"battery": "low", "regime": "good"}, 0.5),
        ({"battery": "low", "regime": "bad"}, 0.5),
    ]
    result = compiled.plan_belief(
        belief=belief,
        target={"done": True},
        outcome_model=outcomes,
        observation_model=lambda state, command: {
            "meter": state["battery"]
        },
        actions=("charge",),
        action_costs={"charge": 0.1},
    )
    execution = result.execution()
    assert execution.action == {"action": "charge"}
    step = execution.advance(outcome={"battery": "ready"})

    assert step.goal_reached
    assert step.terminal
    assert step.route.kind == "terminal"
    assert step.accumulated_cost == pytest.approx(0.1)
    assert step.posterior_marginals()["regime"] == pytest.approx(
        {"good": 0.9 / 1.1, "bad": 0.2 / 1.1}
    )
    assert execution.action is None
    with pytest.raises(TypeError):
        step.posterior[0][0]["regime"] = "bad"


def test_one_step_execution_updates_terminal_hidden_posterior():
    planner = Planner()
    planner.mode("battery", ("low", "ready"))
    planner.mode("regime", ("good", "bad"), priors=(0.5, 0.5))
    planner.command("action", ("none", "charge"))
    planner.observable("done")
    planner.behavior(
        lambda value: iff(
            value["done"] == True,
            value["battery"] == "ready",
        )
    )
    planner.transition(
        "battery", "low", "ready", command=("action", "charge")
    )
    compiled = planner.compile(1)

    def outcomes(state, command):
        success = 0.9 if state["regime"] == "good" else 0.2
        return [
            ({"battery": "ready"}, success),
            ({}, 1.0 - success),
        ]

    result = compiled.plan_belief(
        belief=[
            ({"battery": "low", "regime": "good"}, 0.5),
            ({"battery": "low", "regime": "bad"}, 0.5),
        ],
        target={"done": True},
        outcome_model=outcomes,
        actions=("charge",),
        action_costs={"charge": 0.1},
    )
    execution = result.execution()
    assert execution.action == {"action": "charge"}
    assert not execution.requires_observation
    step = execution.advance(outcome={"battery": "ready"})

    assert step.goal_reached
    assert step.terminal
    assert step.route.kind == "terminal"
    assert step.accumulated_cost == pytest.approx(0.1)
    assert step.posterior_marginals()["regime"] == pytest.approx(
        {"good": 0.9 / 1.1, "bad": 0.2 / 1.1}
    )
    assert execution.action is None


def test_robust_policy_execution_requires_realized_scenario():
    result = reliability_planner(horizon=2).plan_belief(
        belief=[({"mode": "bad"}, 1.0)],
        target={"done": True},
        outcome_scenarios={
            "one": reliability_outcomes,
            "two": reliability_outcomes,
        },
        robust_objective="maximin",
        robust_weight_resolution=2,
        max_robust_candidates=1,
        observation_model=lambda state, command: dict(state),
        actions=("sure",),
    )
    with pytest.raises(ValueError, match="requires outcome_scenario"):
        result.execution()
    execution = result.execution(outcome_scenario="one")
    assert execution.action is not None


def test_reliability_certification_composes_tracker_scope():
    planner = staged_repair_planner()
    unknown = planner.plan_belief(
        min_goal_probability=0.5,
        **staged_arguments(
            belief=TrackedBelief(
                STAGED_BELIEF,
                exact=False,
                retained_probability_mass=None,
            )
        ),
    )
    # Feasible against the supplied normalized beam, but with unknown
    # tracker mass no end-to-end feasibility certificate is possible.
    assert unknown.feasible
    assert unknown.constraint_certification == "indeterminate"

    exact = planner.plan_belief(
        min_goal_probability=0.5,
        **staged_arguments(
            belief=TrackedBelief(
                STAGED_BELIEF,
                exact=True,
                retained_probability_mass=1.0,
            )
        ),
    )
    assert exact.feasible
    assert exact.constraint_certification == "certified-feasible"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("min_goal_probability", -0.1),
        ("min_goal_probability", 1.1),
        ("min_goal_probability", math.nan),
        ("min_branch_goal_probability", 1.1),
    ],
)
def test_reliability_constraints_validate_range(name, value):
    with pytest.raises(ValueError, match=name):
        staged_repair_planner().plan_belief(
            **staged_arguments(), **{name: value}
        )


def test_branch_floor_requires_observation_model():
    with pytest.raises(ValueError, match="observation_model"):
        reliability_planner().plan_belief(
            belief=[({"mode": "bad"}, 1.0)],
            target={"done": True},
            outcome_model=reliability_outcomes,
            min_branch_goal_probability=0.5,
        )


def test_max_frontier_points_validates():
    with pytest.raises(ValueError, match="max_frontier_points"):
        staged_repair_planner().plan_belief(
            max_frontier_points=1, **staged_arguments()
        )


@pytest.mark.parametrize(
    "probabilities",
    [
        (0.25, 0.25),
        (2.0, 2.0),
        (0.0, 0.0),
    ],
)
def test_plan_belief_rejects_unnormalized_outcome_probabilities(
    probabilities,
):
    def outcomes(state, command):
        del state, command
        return [({"mode": "goal"}, probabilities[0]), ({}, probabilities[1])]

    with pytest.raises(ValueError, match="must sum to 1"):
        repair_planner().plan_belief(
            belief=[({"mode": "a"}, 1.0)],
            target={"done": True},
            outcome_model=outcomes,
        )


def test_plan_belief_accepts_normalized_outcome_probabilities():
    def outcomes(state, command):
        del state, command
        return [({"mode": "goal"}, 0.25), ({}, 0.75)]

    result = repair_planner().plan_belief(
        belief=[({"mode": "a"}, 1.0)],
        target={"done": True},
        outcome_model=outcomes,
        actions=["fix_a"],
    )
    assert result.expected_goal_probability == pytest.approx(0.25)


@pytest.mark.parametrize(
    "probabilities",
    [
        (0.25, 0.25),
        (2.0, 2.0),
        (0.0, 0.0),
    ],
)
def test_plan_belief_rejects_unnormalized_observation_probabilities(
    probabilities,
):
    def observations(state, command):
        del command
        return [
            ({"reading": state["mode"]}, probabilities[0]),
            ({"reading": "other"}, probabilities[1]),
        ]

    with pytest.raises(ValueError, match="must sum to 1"):
        repair_planner(horizon=2).plan_belief(
            belief=[({"mode": "a"}, 1.0)],
            target={"done": True},
            outcome_model=repair_outcomes,
            observation_model=observations,
        )


def test_plan_belief_can_be_cancelled_cooperatively():
    with pytest.raises(PlanningCancelled) as exc:
        repair_planner(horizon=2).plan_belief(
            belief=[({"mode": "a"}, 1.0)],
            target={"done": True},
            outcome_model=repair_outcomes,
            observation_model=lambda state, command: dict(state),
            control=PlanControl(cancel=lambda: True),
        )
    assert not exc.value.stats.complete
    assert exc.value.stats.elapsed_seconds >= 0


def test_plan_belief_honors_time_budget():
    with pytest.raises(PlanningBudgetExceeded) as exc:
        repair_planner(horizon=2).plan_belief(
            belief=[({"mode": "a"}, 1.0)],
            target={"done": True},
            outcome_model=repair_outcomes,
            observation_model=lambda state, command: dict(state),
            control=PlanControl(timeout_seconds=0.0),
        )
    assert isinstance(exc.value, ValueError)
    assert not exc.value.stats.complete


def test_plan_belief_reports_progress_and_final_stats():
    snapshots = []
    result = repair_planner(horizon=2).plan_belief(
        belief=[({"mode": "a"}, 1.0)],
        target={"done": True},
        outcome_model=repair_outcomes,
        observation_model=lambda state, command: dict(state),
        control=PlanControl(
            progress=snapshots.append, progress_interval=0.0
        ),
    )
    assert result.action is not None
    final = snapshots[-1]
    assert final.complete
    assert final.policy_nodes > 0
    assert final.outcome_branches > 0
    assert final.best_action == result.action
    assert final.best_expected_utility == pytest.approx(
        max(
            evaluation.expected_utility
            for evaluation in result.evaluations
        )
    )


def test_policy_routing_projects_superset_and_falls_back_for_partial():
    def observations(state, command):
        del command
        if state["mode"] == "goal":
            return {"signal": "goal"}
        expected = state["mode"]
        other = "b" if expected == "a" else "a"
        return [
            ({"signal": expected}, 0.99),
            ({"signal": other}, 0.01),
        ]

    result = repair_planner(horizon=2).plan_belief(
        belief=[({"mode": "a"}, 0.5), ({"mode": "b"}, 0.5)],
        target={"done": True},
        outcome_model=repair_outcomes,
        observation_model=observations,
        min_observation_probability=0.01,
    )
    root = result.policy
    assert root.observation_schema == ("signal",)
    branch = root.branches[0]

    exact = root.route(branch.observation)
    assert exact.kind == "exact"
    assert exact.policy is branch.policy
    assert exact.branch is branch

    superset = dict(branch.observation)
    superset["unrelated_telemetry"] = 42.0
    routed = root.route(superset)
    assert routed.kind == "exact"
    assert routed.policy is branch.policy
    assert root.continuation(superset) is branch.policy

    partial = root.route({"unrelated_telemetry": 42.0})
    assert partial.kind == "fallback"
    assert root.continuation({}) is root.fallback_policy

    fallback = root.route({"signal": "not-a-known-reading"})
    assert fallback.kind == "fallback"
    assert fallback.policy is root.fallback_policy


def test_policy_routing_preserves_heterogeneous_observation_identity():
    def observations(state, command):
        del state, command
        return [
            ({"breaker_signal": True}, 0.5),
            ({"generator_signal": True}, 0.5),
        ]

    result = repair_planner(horizon=2).plan_belief(
        belief=[({"mode": "a"}, 1.0)],
        target={"done": True},
        outcome_model=repair_outcomes,
        observation_model=observations,
    )
    root = result.policy
    assert root.observation_schema == (
        "breaker_signal",
        "generator_signal",
    )
    for branch in root.branches:
        routed = root.route(branch.observation)
        assert routed.kind == "exact"
        assert routed.branch is branch
        with_extra = dict(branch.observation)
        with_extra["unrelated"] = 1
        assert root.route(with_extra).branch is branch
    assert root.route(
        {
            "breaker_signal": True,
            "generator_signal": None,
        }
    ).kind == "unmatched"


def test_certified_policy_tree_is_deeply_immutable():
    result = repair_planner(horizon=2).plan_belief(
        belief=[({"mode": "a"}, 0.5), ({"mode": "b"}, 0.5)],
        target={"done": True},
        outcome_model=repair_outcomes,
        observation_model=lambda state, command: dict(state),
    )
    branch = result.policy.branches[0]
    with pytest.raises(TypeError):
        result.policy.action["action"] = "wait"
    with pytest.raises(TypeError):
        branch.observation["mode"] = "other"
    with pytest.raises(TypeError):
        branch.posterior[0][0]["mode"] = "other"
    with pytest.raises(TypeError):
        result.action_certificates[0].action["action"] = "wait"

    action_copy = result.action
    action_copy["action"] = "wait"
    assert result.action != action_copy


def test_policy_routing_reports_terminal_and_unmatched():
    result = repair_planner(horizon=2).plan_belief(
        belief=[({"mode": "a"}, 1.0)],
        target={"done": True},
        outcome_model=repair_outcomes,
        observation_model=lambda state, command: dict(state),
    )
    leaf = result.policy.branches[0].policy
    terminal = leaf.route({"anything": 1})
    assert terminal.kind == "terminal"
    assert terminal.policy is None

    unpruned_root = result.policy
    assert unpruned_root.fallback_policy is None
    unmatched = unpruned_root.route(
        {key: "no-such-value" for key in unpruned_root.observation_schema}
    )
    assert unmatched.kind == "unmatched"
    assert unmatched.policy is None


def test_policy_branches_expose_their_posterior():
    result = repair_planner(horizon=2).plan_belief(
        belief=[({"mode": "a"}, 0.5), ({"mode": "b"}, 0.5)],
        target={"done": True},
        outcome_model=repair_outcomes,
        observation_model=lambda state, command: dict(state),
    )
    for branch in result.policy.branches:
        assert branch.posterior
        total = sum(probability for _, probability in branch.posterior)
        assert total == pytest.approx(1.0)
        marginals = branch.posterior_marginals()
        assert "mode" in marginals
        assert sum(marginals["mode"].values()) == pytest.approx(1.0)

    # Perfect observation of a two-state belief: each branch's posterior
    # collapses to the single observed state.
    branch = result.policy.branches[0]
    assert len(branch.posterior) == 1
    assert branch.posterior[0][1] == pytest.approx(1.0)
    assert branch.posterior[0][0]["mode"] == branch.observation["mode"]


def test_fallback_branch_exposes_aggregate_posterior():
    def observations(state, command):
        del command
        if state["mode"] == "goal":
            return {"signal": "goal"}
        expected = state["mode"]
        other = "b" if expected == "a" else "a"
        return [
            ({"signal": expected}, 0.99),
            ({"signal": other}, 0.01),
        ]

    result = repair_planner(horizon=2).plan_belief(
        belief=[({"mode": "a"}, 0.5), ({"mode": "b"}, 0.5)],
        target={"done": True},
        outcome_model=repair_outcomes,
        observation_model=observations,
        min_observation_probability=0.01,
    )
    root = result.policy
    fallback = root.fallback_branch
    assert fallback is not None
    assert fallback.policy is root.fallback_policy
    assert fallback.probability == pytest.approx(
        root.discarded_observation_probability
    )
    assert fallback.posterior
    assert sum(
        probability for _, probability in fallback.posterior
    ) == pytest.approx(1.0)
    assert fallback.contributing_observations
    for observation in fallback.contributing_observations:
        assert "signal" in observation

    routed = root.route({"signal": "not-a-known-reading"})
    assert routed.kind == "fallback"
    assert routed.branch is fallback


def test_plan_belief_budget_error_carries_partial_stats():
    with pytest.raises(PlanningBudgetExceeded) as exc:
        repair_planner(horizon=2).plan_belief(
            belief=[({"mode": "a"}, 1.0)],
            target={"done": True},
            outcome_model=repair_outcomes,
            observation_model=lambda state, command: dict(state),
            max_policy_nodes=2,
        )
    assert isinstance(exc.value, ValueError)
    stats = exc.value.stats
    assert stats.policy_nodes > 2
    assert not stats.complete
