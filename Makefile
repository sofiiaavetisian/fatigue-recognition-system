.PHONY: setup test run tree

setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

test:
	pytest

run:
	python -m src.app --config configs/base.yaml

tree:
	find . -maxdepth 3 -type d | sort
