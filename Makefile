SHELL := /bin/zsh
ENV := connect4-zero
PY := conda run -n $(ENV) python

.PHONY: env install-web dev api web test test-core train-60 build

env:
	conda env create -f environment.yml

install-web:
	npm --prefix web install

api:
	$(PY) -m connect4_zero.cli serve

web:
	npm --prefix web run dev

dev:
	@echo "Run 'make api' and 'make web' in separate terminals."

test-core:
	$(PY) -m pytest tests

test: test-core
	npm --prefix web run test

train-60:
	$(PY) -m connect4_zero.cli train --minutes 60

build:
	npm --prefix web run build

