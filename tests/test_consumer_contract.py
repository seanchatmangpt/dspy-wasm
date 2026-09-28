import json
import re
from pathlib import Path

CONTRACT = json.loads(Path("consumer/contract.json").read_text())
WIT = Path("wit/dspy.wit").read_text()
APP = Path("app.py").read_text()


def test_contract_identity_matches_wit_and_component() -> None:
    assert CONTRACT["component"] == "dspy-wasm"
    assert CONTRACT["target"] == "wasm32-wasip2"
    assert f"package {CONTRACT['wit_package']};" in WIT
    assert f"world {CONTRACT['world']} {{" in WIT

    version_match = re.search(
        r"def component_version\(self\) -> str:\s+return \"([^\"]+)\"",
        APP,
    )
    assert version_match
    assert version_match.group(1) == CONTRACT["component_version"]


def test_contract_imports_match_wit() -> None:
    assert CONTRACT["imports"] == {
        "chatman:dspy/lm@0.1.0": ["complete"],
        "chatman:dspy/tools@0.1.0": ["call"],
    }

    assert "import lm;" in WIT
    assert "complete: func(request-json: string) -> string;" in WIT
    assert "import tools;" in WIT
    assert "call: func(name: string, args-json: string) -> string;" in WIT


def test_contract_exports_are_present_in_dspy_world() -> None:
    exports = {
        match.group(1)
        for match in re.finditer(r"^\s*export ([a-z0-9-]+): func\(", WIT, re.MULTILINE)
    }
    assert set(CONTRACT["exports"]) <= exports


def test_json_request_exports_stay_stable() -> None:
    for export in ("run", "render", "evaluate", "compile"):
        assert export in CONTRACT["exports"]
        assert f"export {export}: func(request-json: string) -> string;" in WIT

    assert CONTRACT["payload_encoding"] == "json-string"
    assert CONTRACT["success_state"] == "ALIVE"
