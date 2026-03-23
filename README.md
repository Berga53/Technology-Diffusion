# technology diffusion

bla bla bla

## Structure

- `src/technology_diffusion/`: core modules (`helpers`, `heuristics`, `ip_problems`, `ns`)
- `experiments/`: experimental outputs and data summaries
- `notebooks/`: Jupyter notebooks
- `tests/`: test suite

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducible Environment

This project is currently running on Python 3.13.7.

For an exact recreation of the current environment, create a fresh virtual environment with the same Python version and install the pinned packages:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements-exp.txt
```

Do not copy the `.venv/` folder between machines. Virtual environments contain machine-specific paths and native artifacts, so recreating them from the lock file is safer and more portable.

If you prefer a one-command setup from the repository root:

```bash
make setup-exact PYTHON=python3.13
```

## Run Technology Diffusion Script

```bash
source .venv/bin/activate
python scripts/run_technology_diffusion.py
```

Or, without manually activating the environment:

```bash
make run-technology-diffusion
```