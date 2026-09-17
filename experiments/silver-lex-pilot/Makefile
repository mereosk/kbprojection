# Silver-LEX pilot pipeline (sample -> baseline -> silver LEX -> augmented
# proving -> evaluate). Run steps individually, or `make all` for the full
# chain. All scripts are resumable (id-based skip logic), so re-running a
# target after a partial failure only redoes missing rows.
#
# Requires: ../../.venv (see `make setup`), and for generate-silver-lex* an
# API key in .env (OPENROUTER_API_KEY by default).

REPO_ROOT := $(abspath ../..)
VENV := $(REPO_ROOT)/.venv
PYTHON := $(VENV)/bin/python3
SCRIPTS := scripts
DATA := data

.PHONY: setup sample baseline silver-lex silver-lex-claude augmented augmented-claude evaluate all help

help:
	@echo "Targets:"
	@echo "  setup               create .venv and install kbprojection + pilot deps"
	@echo "  sample              sample e-SNLI pairs -> data/esnli_sample.csv"
	@echo "  baseline            baseline WordNet-only proving -> data/baseline.csv"
	@echo "  silver-lex          silver LEX via Gemini 3.1 Flash-Lite -> data/silver_lex_gemini-3.1-flash-lite.csv"
	@echo "  silver-lex-claude   silver LEX via Claude Sonnet 5 -> data/silver_lex_claude-sonnet-5.csv"
	@echo "  augmented           re-prove with Gemini's silver KB -> data/augmented_gemini-3.1-flash-lite.csv"
	@echo "  augmented-claude    re-prove with Claude's silver KB -> data/augmented_claude-sonnet-5.csv"
	@echo "  evaluate            evaluate all runs -> data/evaluation/{summary.csv,flips.csv,report.md}"
	@echo "  all                 run the full pipeline in order"

setup:
	test -d $(VENV) || python3 -m venv $(VENV)
	$(PYTHON) -m pip install -q --upgrade pip
	$(PYTHON) -m pip install -q -e $(REPO_ROOT)
	$(PYTHON) -m pip install -q datasets python-dotenv

sample:
	$(PYTHON) $(SCRIPTS)/sample_esnli.py

baseline:
	$(PYTHON) $(SCRIPTS)/prove_baseline.py

silver-lex:
	$(PYTHON) $(SCRIPTS)/generate_silver_lex.py

silver-lex-claude:
	$(PYTHON) $(SCRIPTS)/generate_silver_lex.py --model anthropic/claude-sonnet-5

augmented:
	$(PYTHON) $(SCRIPTS)/prove_augmented.py

augmented-claude:
	$(PYTHON) $(SCRIPTS)/prove_augmented.py --silver-csv $(DATA)/silver_lex_claude-sonnet-5.csv

evaluate:
	$(PYTHON) $(SCRIPTS)/evaluate.py

all: sample baseline silver-lex silver-lex-claude augmented augmented-claude evaluate
