# simple makefile to simplify repetetive build env management tasks under posix

# caution: testing won't work on windows, see README

PYTHON ?= python
CYTHON ?= cython
NOSETESTS ?= nosetests
CTAGS ?= ctags

all: clean inplace test

clean-pyc:
	find tract_querier -name "*.pyc" | xargs rm -f

clean-so:
	find tract_querier -name "*.so" | xargs rm -f
	find tract_querier -name "*.pyd" | xargs rm -f

clean-build:
	rm -rf build

clean-ctags:
	rm -f tags

clean-doc:
	rm -rf doc/_build

clean: clean-build clean-pyc clean-so clean-ctags clean-doc

in: inplace # just a shortcut
inplace:
	$(PYTHON) setup.py build_ext -i

test-code: in
	$(NOSETESTS) -s tract_querier
test-doc:
	$(NOSETESTS) -s --with-doctest --doctest-tests --doctest-extension=rst \
	--doctest-extension=inc --doctest-fixtures=_fixture doc/ 

test-coverage:
	rm -rf coverage .coverage
	$(NOSETESTS) -s --with-coverage --cover-html --cover-html-dir=coverage \
	--cover-package=tract_querier tract_querier

test: test-code

trailing-spaces:
	find tract_querier -name "*.py" | xargs perl -pi -e 's/[ \t]*$$//'

cython:
	find tract_querier -name "*.pyx" | xargs $(CYTHON)

ctags:
	# make tags for symbol based navigation in emacs and vim
	# Install with: sudo apt-get install exuberant-ctags
	$(CTAGS) -R *

doc: inplace
	make -C doc html

doc-noplot: inplace
	make -C doc html-noplot
