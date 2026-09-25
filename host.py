"""Minimal Wasmtime host for the dspy-wasm component."""

from __future__ import annotations

import argparse
from pathlib import Path

from wasmtime import Config, Engine, Store
from wasmtime.component import Component, Linker


def invoke(component_path: Path) -> None:
    config = Config()
    config.cache = True
    engine = Engine(config)
    store = Store(engine)
    component = Component.from_file(engine, str(component_path))
    linker = Linker(engine)
    instance = linker.instantiate(store, component)

    for export_name in ("component-version", "runtime-info", "dspy-version"):
        func = instance.get_func(store, export_name)
        if func is None:
            raise RuntimeError(f"missing export: {export_name}")
        print(f"{export_name}: {func(store)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("component", type=Path)
    args = parser.parse_args()
    invoke(args.component)


if __name__ == "__main__":
    main()
