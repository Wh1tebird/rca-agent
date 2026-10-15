"""Pure, deterministic v4 Inquiry menu projection and admission tests."""
import json

import pytest
from test_batched_inquiry_state import planning_step
from test_dynamic_investigation import registry
from test_engineering_alternatives import DiscoveryV7
from test_engineering_independent_regressions import EmptyPlanner, PublicEnvironment, output

from rca_agent.investigation.inquiry_contract import (
    action_fingerprint,
    measurement_target,
    project_inquiry_menu,
    public_query,
)
from rca_agent.investigation.inquiry_loop import (
    LoopState,
    accept_loop_step,
    loop_attempt_digest,
    normalize_loop_step,
)
from rca_agent.models import CallToolAction
from rca_agent.runner import EpisodeRunner
from rca_agent.tools.fingerprint import query_fingerprint


def _menu():
    env = PublicEnvironment()
    bootstrap = env.reset()
    from rca_agent.langchain_acquisition import query_options

    options = query_options(
        {"bootstrap": bootstrap.model_dump(mode="json")},
        [bootstrap.alert_entity.model_dump(mode="json")],
    )
    return {
        ref: {**item, "action": item["action"].model_dump(mode="json"), "query_cost": 1}
        for ref, item in options.items()
    }


def test_projection_is_subset_and_preserves_actions_and_fingerprints():
    menu = _menu()
    sample = next(item for item in menu.values()
                  if measurement_target(CallToolAction.model_validate(item["action"]))["measurement_kind"] == "series")
    action = CallToolAction.model_validate(sample["action"])
    target = measurement_target(action)
    need = {"entity_ref": target["entity_ref"], "observable_signals": (target["signal"],),
            "measurement_kind": target["measurement_kind"], "windows": target["windows"]}
    projection = project_inquiry_menu(menu, [need], budget_remaining=10)
    assert set(projection["focused"]) <= set(menu)
    assert projection["focused"]
    for ref, item in projection["focused"].items():
        assert item["action"] == menu[ref]["action"]
        assert query_fingerprint(
            CallToolAction.model_validate(item["action"]).tool_name,
            CallToolAction.model_validate(item["action"]).request,
        ) == query_fingerprint(
            CallToolAction.model_validate(menu[ref]["action"]).tool_name,
            CallToolAction.model_validate(menu[ref]["action"]).request,
        )
        target_item = measurement_target(CallToolAction.model_validate(item["action"]))
        assert target_item["entity_ref"] == need["entity_ref"]
        assert target_item["signal"] in need["observable_signals"]
        assert target_item["measurement_kind"] == need["measurement_kind"]


@pytest.mark.parametrize("field", ["entity_ref", "observable_signals", "measurement_kind", "windows"])
def test_need_field_mismatch_is_excluded_from_focused_menu(field):
    menu = _menu()
    sample = next(iter(menu.values()))
    target = measurement_target(CallToolAction.model_validate(sample["action"]))
    need = {"entity_ref": target["entity_ref"], "observable_signals": (target["signal"],),
            "measurement_kind": target["measurement_kind"], "windows": target["windows"]}
    if field == "entity_ref":
        need[field] = "ent-other"
    elif field == "observable_signals":
        need[field] = ("signal-that-is-not-offered",)
    elif field == "measurement_kind":
        need[field] = "entity_exploration"
    else:
        need[field] = {"series": {"start": "2030-01-01T00:00:00Z", "end": "2030-01-01T00:01:00Z"}}
    projection = project_inquiry_menu(menu, [need], budget_remaining=10)
    assert all(ref not in projection["focused"] for ref in menu)


def test_no_match_uses_small_explicit_exploration_layer():
    menu = _menu()
    projection = project_inquiry_menu(menu, [{"entity_ref": "ent-missing",
        "observable_signals": ("missing",), "measurement_kind": "series", "windows": {}}],
        budget_remaining=10)
    assert projection["focused"] == {}
    assert projection["mode"] == "exploration"
    assert 0 < len(projection["exploration"]) <= 8
    assert set(projection["exposed"]) == set(projection["exploration"])


def test_projection_honors_existing_fingerprints_and_budget_without_repricing():
    menu = _menu()
    ref, item = next(iter(menu.items()))
    action = CallToolAction.model_validate(item["action"])
    fingerprint = query_fingerprint(action.tool_name, action.request)
    item["query_cost"] = 2
    projection = project_inquiry_menu(menu, [], existing_query_fingerprints=(fingerprint,),
                                      budget_remaining=1)
    assert ref not in projection["focused"]
    assert ref not in projection["exploration"]
    assert all(entry.get("query_cost") == 1 for entry in projection["exposed"].values())


def test_v4_aliases_are_stable_and_hidden_action_cannot_bypass_admission():
    runner = EpisodeRunner(PublicEnvironment(), evidence_registry=registry(),
        discovery=DiscoveryV7(),
        verification_planner=EmptyPlanner(), investigation_workspace_mode="shadow",
        inquiry_loop_mode="batched")
    runner.start()
    current = runner.inquiry_loop_view()
    public = current["view"]
    aliases = public["menu_aliases"]
    assert aliases
    assert all(item["menu_ref"].startswith("aq-") for item in aliases.values())
    assert all(item["action_fingerprint"].startswith("act-") for item in aliases.values())
    assert all(item["action_fingerprint"] == action_fingerprint(
        CallToolAction.model_validate(current["menu"][item["menu_ref"]]["action"]))
        for item in aliases.values())
    state = LoopState.model_validate(runner.investigation_workspace_snapshot.inquiry_loop)
    offered_ref = next(iter(public["offered_menu_refs"]))
    hidden_ref = next(ref for ref, item in current["menu"].items()
                      if ref != offered_ref and item["action"].get("tool_name") == "get_metric_series")
    hidden = public_query({**current["menu"][hidden_ref],
                           "action": CallToolAction.model_validate(current["menu"][hidden_ref]["action"])})
    draft = output({"context": json.dumps({"queries": {hidden_ref: hidden}})}, (("n1", hidden_ref),))
    step = normalize_loop_step(planning_step(state, draft, hidden_ref),
                               current["menu"], menu_aliases=aliases)
    before = state.model_dump(mode="json")
    with pytest.raises(ValueError, match="projected Inquiry menu"):
        accept_loop_step(state, step, menu=current["menu"], batch_limit=2,
                         query_limit=10, offered_refs={offered_ref})
    assert state.model_dump(mode="json") == before and state.verify_digest()


def test_projection_digest_and_attempt_digest_bind_public_projection_and_state():
    runner = EpisodeRunner(PublicEnvironment(), evidence_registry=registry(),
        discovery=DiscoveryV7(),
        verification_planner=EmptyPlanner(), investigation_workspace_mode="shadow",
        inquiry_loop_mode="batched")
    runner.start()
    bundle = runner.inquiry_loop_view()
    state_digest = bundle["view"]["based_on_digest"]
    projection_digest = bundle["view"]["menu_projection_digest"]
    attempt = loop_attempt_digest(bundle)
    changed = json.loads(json.dumps(bundle))
    changed["view"]["menu_projection"]["mode"] = "tampered"
    assert changed["view"]["based_on_digest"] == state_digest
    assert changed["view"]["menu_projection_digest"] == projection_digest
    assert loop_attempt_digest(changed) != attempt
