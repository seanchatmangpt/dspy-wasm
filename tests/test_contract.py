from pathlib import Path


WIT = Path("wit/dspy.wit")


def test_contract_declares_expected_world() -> None:
    text = WIT.read_text()
    assert "world dspy" in text
    assert "export component-version: func() -> string;" in text
    assert "export runtime-info: func() -> string;" in text
    assert "export dspy-version: func() -> string;" in text


def test_bootstrap_and_dspy_implement_same_contract() -> None:
    bootstrap = Path("bootstrap.py").read_text()
    app = Path("app.py").read_text()
    for method in ("component_version", "runtime_info", "dspy_version"):
        assert f"def {method}" in bootstrap
        assert f"def {method}" in app
