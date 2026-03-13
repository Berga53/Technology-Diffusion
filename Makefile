PYTHON ?= python3
VENV_PYTHON := .venv/bin/python

.PHONY: setup setup-exact td-exp td-exp-help

setup:
	$(PYTHON) -m venv .venv
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -r requirements.txt

setup-exact:
	$(PYTHON) -m venv .venv
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -r requirements-exp.txt

td-exp:
	$(VENV_PYTHON) scripts/TD_exp.py

td-exp-help:
	$(VENV_PYTHON) scripts/TD_exp.py --help