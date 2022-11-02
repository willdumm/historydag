# Taken from gctree makefile https://github.com/matsengrp/gctree/blob/master/Makefile
default:

install:
	pip install --use-pep517 -r requirements.txt
	pip install -e .

test:
	pytest tests/

format:
	black historydag --exclude historydag/dag_pb2.py
	black tests
	docformatter --in-place historydag/*.py

lint:
	# stop the build if there are Python syntax errors or undefined names
	flake8 historydag --count --select=E9,F63,F7,F82 --show-source --statistics --exclude historydag/dag_pb2.py
	flake8 tests --count --select=E9,F63,F7,F82 --show-source --statistics
	# exit-zero treats all errors as warnings. The GitHub editor is 127 chars wide
	flake8 historydag --count --max-complexity=30 --max-line-length=127 --statistics --ignore=E203,W503 --exclude historydag/dag_pb2.py
	flake8 tests --count --max-complexity=30 --max-line-length=127 --statistics --ignore=E203,W503

docs:
	make -C docs html

.PHONY: install test format lint deploy docs
