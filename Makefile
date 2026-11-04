# simple makefile to simplify repetetive build env management tasks under posix

# caution: testing won't work on windows, see README

PYTHON ?= python
PYTEST ?= pytest

all: clean inplace test

clean-pyc:
	find tract_querier -name "*.pyc" | xargs rm -f

clean-build:
	rm -rf build

clean-doc:
	rm -rf doc/_build

clean: clean-build clean-pyc clean-doc

in: inplace # just a shortcut
inplace:
	$(PYTHON) -m pip install -e .

test-code: in
	$(PYTEST) tract_querier
test-doc:
	$(PYTEST) --doctest-glob='*.rst' doc/

test-coverage:
	rm -rf coverage .coverage
	coverage run --source tract_querier -m $(PYTEST) tract_querier
	coverage html -d coverage

test: test-code

doc: inplace
	make -C doc html
