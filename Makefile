EVAL=1
TRAIN=0

download_mc_maze_small:
	dandi download https://dandiarchive.org/dandiset/000140

download_mc_maze_medium:
	dandi download https://dandiarchive.org/dandiset/000139

download_mc_maze_large:
	dandi download https://dandiarchive.org/dandiset/000138

download_mc_maze:
	dandi download https://dandiarchive.org/dandiset/000128

download_mc_rtt:
	dandi download https://dandiarchive.org/dandiset/000129

download_dmfc:
	dandi download https://dandiarchive.org/dandiset/000127

download_area2bump:
	dandi download https://dandiarchive.org/dandiset/000130

download_all: 
	download_mc_maze_small 
	download_mc_maze_medium 
	download_mc_maze_large 
	download_mc_maze 
	download_mc_rtt 
	download_dmfc 
	download_area2bump

# Run main.py file to 
run_eval:
	python3 -m src.main $(EVAL)

# Run main.py file to train the model
run_training:
	python3 -m src.main $(TRAIN)

# Setup Environment
setup:
	pip install -r requirements.txt

# Test dataloader.py
test_dataloader:
	python3 -m unittest tests/src/test_dataloader.py 

# Test train.py
test_train:
	python3 -m unittest tests/src/test_train.py 

# Test bayes_tuning.py
test_tuning:
	python3 -m unittest tests/src/test_bayes_tuning.py

# Test load_model.py
test_load_model:
	python3 -m unittest tests/src/test_load_model.py 

# Test Models/*.py
test_models:
	python3 -m unittest tests/src/test_models.py

# Test all
test_all: test_train test_tuning test_load_model test_models test_dataloader

# Get rid of pycache inside tests folder directory
clean_tests:
	find tests/src | grep -E "(__pycache__|\.pyc|\.pyo$))" | xargs rm -rf

# Get rid of pycache inside tests/Transformer directory
clean_tests_models:
	find tests/src/models | grep -E "(__pycache__|\.pyc|\.pyo$))" | xargs rm -rf

# Get rid of pycache inside Transformer directory
clean_src:
	find src | grep -E "(__pycache__|\.pyc|\.pyo$))" | xargs rm -rf

# Get rid of pycache inside Transformer directory
clean_src_models:
	find src | grep -E "(__pycache__|\.pyc|\.pyo$))" | xargs rm -rf

# Get rid of pycache
clean: clean_tests clean_tests_models clean_src clean_src_models
