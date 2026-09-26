.PHONY: install install-n2v test verify demo benchmark clean

PY ?= python

install:
	$(PY) -m pip install --only-binary=:all: -r requirements.txt

install-n2v:
	$(PY) -m pip install --only-binary=:all: node2vec gensim

test:
	$(PY) -m pytest -q -W ignore tests --basetemp=data/.pytest-tmp -p no:cacheprovider

verify:
	$(PY) scripts/verify.py --fast

verify-full:
	$(PY) scripts/verify.py

demo:
	$(PY) examples/run_demo.py --fast

benchmark:
	$(PY) -c "from heteroforge.pipeline.benchmark import run_benchmark; from heteroforge.eval.report import save_benchmark; p = run_benchmark(enforce_gate=True); save_benchmark(p, 'benchmark.json'); print('[GATE]', p['haar_gate'])"

scan:
	$(PY) scripts/scan_chars.py

clean:
	$(PY) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').glob('__pycache__') if False]"
