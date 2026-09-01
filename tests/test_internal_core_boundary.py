from pathlib import Path


def test_package_has_no_rca_core_import_or_dependency():
    project_root = Path(__file__).parents[1]
    python_files = [
        *project_root.joinpath("src").rglob("*.py"),
        *project_root.joinpath("evaluation").rglob("*.py"),
    ]
    assert all("rca_core" not in path.read_text() for path in python_files)

    pyproject = project_root.joinpath("pyproject.toml").read_text()
    assert "rca-core" not in pyproject


def test_ground_truth_loader_stays_outside_inference_package():
    project_root = Path(__file__).parents[1]
    assert project_root.joinpath("evaluation", "ground_truth.py").is_file()
    assert not project_root.joinpath("src", "rca_agent", "core", "ground_truth.py").exists()


def test_research_modules_are_not_internalized_into_runtime_core():
    core_root = Path(__file__).parents[1] / "src" / "rca_agent" / "core"
    excluded = {
        "automatic.py",
        "database_oracle.py",
        "evaluation.py",
        "experiment.py",
        "ground_truth.py",
        "node_traffic_holdout.py",
        "oracle.py",
        "scoring.py",
    }
    assert excluded.isdisjoint({path.name for path in core_root.rglob("*.py")})
