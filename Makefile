install:
	pip install -e .

retrain:
	python sspo/grid_search.py
