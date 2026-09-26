.PHONY: install install-dspy bindings wasi-toolchain wasi-wheels wasi-deps bootstrap dspy host-bootstrap host-dspy wasm-test test bench clean

# The DSPy component is built with componentize-py 0.25.1 compiled from its
# published crate with a raised build-time hostcall budget (wasi/toolchain.sh).
# The bootstrap court uses the stock PyPI release.
COMPONENTIZE_PY ?= build/wasi/bin/componentize-py

install:
	python -m pip install -e ".[dev]"

install-dspy:
	python -m pip install -e ".[dev,dspy]"

bindings:
	rm -rf build/bindings
	mkdir -p build/bindings
	componentize-py -d wit -w dspy --world-module dspy_bindings \
		--import-interface-name chatman:dspy/lm@0.1.0=host_lm \
		--import-interface-name chatman:dspy/tools@0.1.0=host_tools \
		bindings build/bindings

# wasi-sdk 33, CPython 3.14 for wasm32-wasip2, zlib, libyaml, OpenSSL and the
# componentize-py build.
wasi-toolchain:
	bash wasi/toolchain.sh

# Every native dependency cross-compiled from source to cp314 WASI wheels.
wasi-wheels: wasi-toolchain
	python wasi/build_wheels.py fetch
	python wasi/build_wheels.py build --missing

# Full locked dependency closure (wasi/requirements.lock) in build/wasi_deps.
wasi-deps: wasi-wheels
	python wasi/build_wheels.py install

bootstrap:
	mkdir -p dist
	componentize-py -d wit -w bootstrap --world-module dspy_bindings componentize --stub-wasi -p . bootstrap_app -o dist/bootstrap.wasm

dspy:
	test -d build/wasi_deps || $(MAKE) wasi-deps
	mkdir -p dist
	$(COMPONENTIZE_PY) -d wit -w dspy --world-module dspy_bindings \
		--import-interface-name chatman:dspy/lm@0.1.0=host_lm \
		--import-interface-name chatman:dspy/tools@0.1.0=host_tools \
		componentize \
		-p . -p build/wasi_deps \
		app -o dist/dspy.wasm

host-bootstrap:
	python host.py dist/bootstrap.wasm

host-dspy:
	python host.py dist/dspy.wasm

wasm-test:
	python host.py dist/dspy.wasm --self-test

test:
	python -m pytest -q

# Timing receipt for the capability boundary (and bootstrap.wasm when built);
# medians are bounded by bench/bench_capabilities.py BOUNDS_MS.
bench:
	python bench/bench_capabilities.py --write bench/receipt.json

clean:
	rm -rf build dist
