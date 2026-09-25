.PHONY: install install-dspy bindings bootstrap dspy host-bootstrap host-dspy test clean

install:
	python -m pip install -e ".[dev]"

install-dspy:
	python -m pip install -e ".[dev,dspy]"

bindings:
	rm -rf build/bindings
	mkdir -p build/bindings
	componentize-py -d wit -w dspy bindings build/bindings

bootstrap:
	mkdir -p dist
	componentize-py -d wit -w dspy componentize --stub-wasi bootstrap -o dist/bootstrap.wasm

dspy:
	mkdir -p dist
	componentize-py -d wit -w dspy componentize --stub-wasi app -o dist/dspy.wasm

host-bootstrap:
	python host.py dist/bootstrap.wasm

host-dspy:
	python host.py dist/dspy.wasm

test:
	pytest -q

clean:
	rm -rf build dist
