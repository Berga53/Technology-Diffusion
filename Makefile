PYTHON ?= python3
VENV_PYTHON := .venv/bin/python

.PHONY: setup run-technology-diffusion run-technology-diffusion-help

setup:
	$(PYTHON) -m venv .venv
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -r requirements.txt

run-technology-diffusion:
	$(VENV_PYTHON) scripts/run_technology_diffusion.py

run-technology-diffusion-help:
	$(VENV_PYTHON) scripts/run_technology_diffusion.py --help