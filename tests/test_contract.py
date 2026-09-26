from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


WIT = Path("wit/dspy.wit")


def test_contract_declares_bootstrap_and_operational_worlds() -> None:
    text = WIT.read_text()
    assert "world bootstrap" in text
    assert "world dspy" in text
    assert "import lm;" in text
    assert "export run-self-tests: func() -> string;" in text
    assert "export predict: func(signature: string, inputs-json: string) -> string;" in text


def test_component_implements_operational_exports() -> None:
    app = Path("app.py").read_text()
    for method in (
        "component_version",
        "runtime_info",
        "dspy_version",
        "run_self_tests",
        "predict",
    ):
        assert f"def {method}" in app


def test_orjson_projection_covers_dspy_eager_surface() -> None:
    path = Path("wasm_compat/orjson.py")
    spec = spec_from_file_location("wasm_orjson", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    data = module.dumps(
        {"b": 2, "a": 1},
        option=module.OPT_SORT_KEYS | module.OPT_APPEND_NEWLINE,
    )
    assert data == b'{"a":1,"b":2}\n'
    assert module.loads(data) == {"a": 1, "b": 2}
