import pytest
from pydantic import ValidationError

from rca_agent.core.v3.catalog import mechanism_catalog
from rca_agent.core.v3.models import Candidate, MechanismFamily


def test_catalog_covers_four_families_and_eleven_candidates():
    catalog = mechanism_catalog("frontend")

    assert len(catalog) == 11
    assert {item.family for item in catalog} == set(MechanismFamily)
    assert len({item.candidate_id for item in catalog}) == 11
    assert all(item.positive_evidence_requirements for item in catalog)
    assert all(item.falsification_conditions for item in catalog)
    assert all(item.confounders for item in catalog)
    assert all(item.required_capabilities for item in catalog)
    assert all(item.knowledge_provenance for item in catalog)


@pytest.mark.parametrize(
    "missing_field",
    [
        "positive_evidence_requirements",
        "falsification_conditions",
        "confounders",
        "required_capabilities",
        "conditions",
        "knowledge_provenance",
    ],
)
def test_candidate_contract_rejects_missing_safety_fields(missing_field: str):
    raw = mechanism_catalog("frontend")[0].model_dump()
    raw[missing_field] = [] if missing_field != "knowledge_provenance" else ""

    with pytest.raises(ValidationError):
        Candidate.model_validate(raw)


def test_positive_requirement_must_reference_owned_condition():
    raw = mechanism_catalog("frontend")[0].model_dump()
    raw["positive_evidence_requirements"] = ["other-condition"]

    with pytest.raises(ValidationError, match="positive requirements"):
        Candidate.model_validate(raw)
