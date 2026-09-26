"""Wasmtime host for dspy-wasm.

The host owns provider/network authority. The WASM component owns DSPy program
semantics and calls this host only through the typed WIT LM interface.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path
from typing import Any

from wasmtime import Config, Engine, Store
from wasmtime.component import Component, Linker


DEFAULT_RESPONSE = "[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]"


class CompletionProvider:
    def __init__(
        self,
        *,
        static_response: str | None,
        base_url: str | None,
        api_key: str | None,
        upstream_model: str | None,
    ) -> None:
        self.static_response = static_response
        self.base_url = base_url
        self.api_key = api_key
        self.upstream_model = upstream_model

    def complete(self, _store, request_json: str) -> str:
        request = json.loads(request_json)
        if self.static_response is not None:
            return json.dumps(
                {
                    "text": self.static_response,
                    "model": self.upstream_model or request.get("model", "static"),
                    "finish_reason": "stop",
                }
            )
        return json.dumps(self._http_complete(request))

    def _http_complete(self, request: dict[str, Any]) -> dict[str, Any]:
        if not self.base_url:
            raise RuntimeError("host LM requires --response or --base-url/OPENAI_BASE_URL")

        model = self.upstream_model or request.get("model")
        if not model or model == "wasm-host":
            raise RuntimeError("real provider mode requires --upstream-model")

        messages: list[dict[str, str]] = []
        if request.get("system"):
            messages.append({"role": "system", "content": request["system"]})
        for message in request.get("messages", []):
            messages.append(
                {
                    "role": message.get("role", "user"),
                    "content": message.get("text", ""),
                }
            )

        body = json.dumps({"model": model, "messages": messages}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = self.base_url.rstrip("/") + "/chat/completions"
        http_request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(http_request, timeout=120) as response:
            payload = json.loads(response.read())

        choice = payload["choices"][0]
        return {
            "id": payload.get("id"),
            "model": payload.get("model", model),
            "text": choice["message"]["content"],
            "finish_reason": choice.get("finish_reason", "stop"),
        }


def instantiate(component_path: Path, provider: CompletionProvider):
    config = Config()
    config.cache = True
    engine = Engine(config)
    store = Store(engine)
    component = Component.from_file(engine, str(component_path))
    linker = Linker(engine)

    with linker.root() as root:
        with root.add_instance("chatman:dspy/lm@0.1.0") as lm:
            lm.add_func("complete", provider.complete)

    instance = linker.instantiate(store, component)
    return store, instance


def call_json(store, instance, export_name: str, *args: str) -> dict[str, Any]:
    func = instance.get_func(store, export_name)
    if func is None:
        raise RuntimeError(f"missing export: {export_name}")
    result = func(store, *args)
    if not isinstance(result, str):
        raise TypeError(f"{export_name} returned non-string result")
    return json.loads(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("component", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--predict-signature")
    parser.add_argument("--inputs", default="{}")
    parser.add_argument("--response")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--upstream-model", default=os.environ.get("DSPY_WASM_MODEL"))
    args = parser.parse_args()

    # Deterministic static response is the default only for tests/inspection.
    # Explicit real-provider configuration disables it.
    static_response = args.response
    if static_response is None and not args.base_url:
        static_response = DEFAULT_RESPONSE

    provider = CompletionProvider(
        static_response=static_response,
        base_url=args.base_url,
        api_key=args.api_key,
        upstream_model=args.upstream_model,
    )
    store, instance = instantiate(args.component, provider)

    if args.self_test:
        report = call_json(store, instance, "run-self-tests")
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(0 if report.get("state") == "ALIVE" else 1)

    if args.predict_signature:
        report = call_json(
            store,
            instance,
            "predict",
            args.predict_signature,
            args.inputs,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(0 if report.get("state") == "ALIVE" else 1)

    for export_name in ("component-version", "runtime-info", "dspy-version"):
        func = instance.get_func(store, export_name)
        if func is None:
            raise RuntimeError(f"missing export: {export_name}")
        print(f"{export_name}: {func(store)}")


if __name__ == "__main__":
    main()
