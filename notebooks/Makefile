SHELL := /bin/bash

help:	## Show this help.
	@egrep -h '\s##\s' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m  %-30s\033[0m %s\n", $$1, $$2}'

venv: ## Create virtualenv.
	@python3 -m virtualenv ../.venv-notebook
	@echo "\n--------------------"
	@echo "Please manually run source ../.venv-notebook/bin/activate"

install:	## Install requirements.
	@pip install -r requirements.txt