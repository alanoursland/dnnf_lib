import math

import pytest

from modenexus import Planner, iff


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
