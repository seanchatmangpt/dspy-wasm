.PHONY: install install-dspy bindings wasi-deps bootstrap dspy host-bootstrap host-dspy wasm-test test clean

install:
	python -m pip install -e ".[dev]"

install-dspy:
	python -m pip install -e ".[dev,dspy]"

bindings:
	rm -rf build/bindings
	mkdir -p build/bindings
	componentize-py -d wit -w dspy --world-module dspy_bindings \
		--import-interface-name chatman:dspy/lm@0.1.0=host_lm \
		bindings build/bindings

wasi-deps:
	rm -rf build/wasi_deps
	mkdir -p build/wasi_deps
	python -m pip install \
		--target build/wasi_deps \
		--platform any \
		--platform wasi_0_0_0_wasm32 \
		--python-version "3.12" \
		--only-binary :all: \
		--index-url https://benbrandt.github.io/wasi-wheels/ \
		--extra-index-url https://pypi.org/simple \
		--upgrade \
		"pydantic>=2.11.0" "regex>=2023.10.3"

bootstrap:
	mkdir -p dist
	componentize-py -d wit -w bootstrap --world-module dspy_bindings componentize --stub-wasi -p . bootstrap -o dist/bootstrap.wasm

dspy: wasi-deps
	mkdir -p dist
	componentize-py -d wit -w dspy \
		--import-interface-name chatman:dspy/lm@0.1.0=host_lm \
		componentize --stub-wasi \
		-p wasm_compat -p . -p build/wasi_deps \
		app -o dist/dspy.wasm

host-bootstrap:
	python host.py dist/bootstrap.wasm

host-dspy:
	python host.py dist/dspy.wasm

wasm-test:
	python host.py dist/dspy.wasm --self-test

test:
	pytest -q

clean:
	rm -rf build dist
