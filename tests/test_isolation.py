from pathlib import Path


def test_inference_package_has_no_evaluator_loader_or_evaluator_path_dependency():
    source_root = Path(__file__).parents[1] / "src" / "rca_agent"
    forbidden_imports = (
        "rca_core.ground_truth",
        "evaluation.ground_truth",
        "FlatGroundTruthRepository",
    )
    forbidden_paths = ("answer_key/", "/ground_truth/")

    for path in source_root.rglob("*.py"):
        text = path.read_text()
        assert all(item not in text for item in forbidden_imports), path
        assert all(item not in text for item in forbidden_paths), path


def test_evaluator_loader_exists_only_outside_inference_package():
    project_root = Path(__file__).parents[1]
    inference_files = list((project_root / "src" / "rca_agent").rglob("*.py"))
    evaluator_file = project_root / "evaluation" / "evaluator.py"

    assert evaluator_file.is_file()
    assert "FlatGroundTruthRepository" in evaluator_file.read_text()
    assert all("FlatGroundTruthRepository" not in path.read_text() for path in inference_files)
