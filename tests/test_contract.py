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


def _load(path: str, name: str):
    spec = spec_from_file_location(name, Path(path))
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rpds_projection_is_persistent() -> None:
    rpds = _load("wasm_compat/rpds.py", "wasm_rpds")

    empty = rpds.HashTrieMap()
    one = empty.insert("a", 1)
    assert dict(empty) == {} and dict(one) == {"a": 1}
    assert one.update({"b": 2}).remove("a") == {"b": 2}
    assert one.discard("missing") is one
    assert rpds.HashTrieMap.convert({"x": 1}) == {"x": 1}
    assert hash(one) == hash(rpds.HashTrieMap(a=1))

    s = rpds.HashTrieSet().insert(1).update([2, 3])
    assert set(s) == {1, 2, 3} and set(s.discard(2)) == {1, 3} and 2 in s

    previous = rpds.List()
    pushed = previous.push_front("b").push_front("a")
    assert list(previous) == [] and list(pushed) == ["a", "b"]
    assert pushed.first == "a" and list(pushed.rest) == ["b"]


def test_jiter_projection_parses_complete_documents_only() -> None:
    jiter = _load("wasm_compat/jiter.py", "wasm_jiter")
    assert jiter.from_json(b'{"a": [1, 2]}') == {"a": [1, 2]}
    try:
        jiter.from_json(b'{"a": ', partial_mode="trailing-strings")
    except NotImplementedError:
        pass
    else:
        raise AssertionError("partial parsing must trap")
