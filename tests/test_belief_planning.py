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


def test_policy_routing_projects_superset_and_rejects_partial():
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

    with pytest.raises(KeyError, match="missing planner schema keys"):
        root.route({"unrelated_telemetry": 42.0})
    with pytest.raises(KeyError, match="missing planner schema keys"):
        root.continuation({})

    fallback = root.route({"signal": "not-a-known-reading"})
    assert fallback.kind == "fallback"
    assert fallback.policy is root.fallback_policy


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
