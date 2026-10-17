# ==============================================================================
# Makefile: Credit Card Fraud Detection & Transaction Risk Analysis
# Automation commands for local development, model training, and testing
# ==============================================================================

PYTHON = python

.PHONY: help setup data train eval sql powerbi test app clean

help:
	@echo "Available commands:"
	@echo "  make setup     - Install Python dependencies"
	@echo "  make data      - Generate realistic sample datasets"
	@echo "  make train     - Train Logistic Regression & Random Forest models"
	@echo "  make eval      - Generate evaluation curves & publication figures"
	@echo "  make sql       - Run automated SQL analytics & create database"
	@echo "  make powerbi   - Export Star Schema tables for Power BI"
	@echo "  make test      - Run automated pytest unit test suite"
	@echo "  make app       - Launch interactive Streamlit web dashboard"
	@echo "  make clean     - Remove temporary caches and logs"

setup:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) data/generate_sample_data.py --samples 50000 --fraud-ratio 0.003 --output data/raw/creditcard.csv
	$(PYTHON) data/generate_sample_data.py --samples 5000 --fraud-ratio 0.005 --output data/raw/sample_creditcard.csv

train:
	$(PYTHON) -m src.models --data-path data/raw/creditcard.csv

eval:
	$(PYTHON) -m src.evaluation

sql:
	$(PYTHON) sql/run_sql_analysis.py --data-path data/raw/creditcard.csv

powerbi:
	$(PYTHON) power_bi/export_powerbi_data.py

test:
	$(PYTHON) -m pytest -v tests/

app:
	$(PYTHON) -m streamlit run app/app.py

clean:
	rm -rf __pycache__ .pytest_cache *.log
