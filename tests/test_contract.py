from pathlib import Path


WIT = Path("wit/dspy.wit")


def test_contract_declares_bootstrap_and_operational_worlds() -> None:
    text = WIT.read_text()
    assert "world bootstrap" in text
    assert "world dspy" in text
    assert "import lm;" in text
    assert "export run-self-tests: func() -> string;" in text
    assert "export predict: func(signature: string, inputs-json: string) -> string;" in text
    assert "import tools;" in text
    assert "call: func(name: string, args-json: string) -> string;" in text
    for export in ("capabilities", "run", "render", "evaluate", "compile"):
        assert f"export {export}: func(" in text


def test_component_implements_operational_exports() -> None:
    app = Path("app.py").read_text()
    for method in (
        "component_version",
        "runtime_info",
        "dspy_version",
        "run_self_tests",
        "predict",
        "capabilities",
        "run",
        "render",
        "evaluate",
        "compile",
    ):
        assert f"def {method}" in app


def test_wasi_lock_covers_every_native_recipe() -> None:
    lock = {
        line.split("==")[0].lower().replace("_", "-")
        for line in Path("wasi/requirements.lock").read_text().splitlines()
        if line and not line.startswith("#")
    }
    native = {
        line.split("==")[0].lower().replace("_", "-")
        for line in Path("wasi/native.txt").read_text().split()
    }
    assert native <= lock, native - lock
    assert {"numpy", "optuna", "litellm", "dspy"} <= lock
