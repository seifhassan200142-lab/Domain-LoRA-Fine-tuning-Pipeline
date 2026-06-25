.PHONY: install validate split baseline train evaluate infer api ui test clean

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

validate:
	python scripts/validate_dataset.py --data data/raw/sample_support_dataset.jsonl --report outputs/dataset_validation_report.json

split:
	python scripts/split_dataset.py --data data/raw/sample_support_dataset.jsonl --train-out data/splits/train.jsonl --validation-out data/splits/validation.jsonl

baseline:
	python scripts/run_baseline.py --config configs/training_config.yaml --limit 5

train:
	python scripts/train_lora.py --config configs/training_config.yaml

evaluate:
	python scripts/evaluate_model.py --config configs/training_config.yaml --limit 5

infer:
	python scripts/run_inference.py --text "My order arrived damaged and I need a replacement."

api:
	uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000

ui:
	streamlit run ui/streamlit_app.py

test:
	pytest -q

clean:
	rm -rf outputs/* data/splits/*.jsonl .pytest_cache __pycache__
	touch outputs/.gitkeep data/splits/.gitkeep
