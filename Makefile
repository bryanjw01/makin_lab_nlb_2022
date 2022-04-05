# Download mc_maze_small dataset from dandi
download_mc_maze_small:
	dandi download https://dandiarchive.org/dandiset/000140
	mv 000140 data

# Download mc_maze_medium dataset from dandi
download_mc_maze_medium:
	dandi download https://dandiarchive.org/dandiset/000139
	mv 000139 data

# Download mc_maze_large dataset from dandi
download_mc_maze_large:
	dandi download https://dandiarchive.org/dandiset/000138
	mv 000138 data

# Download mc_maze dataset from dandi
download_mc_maze:
	dandi download https://dandiarchive.org/dandiset/000128
	mv 000128 data

# Download mc_rtt dataset from dandi
download_mc_rtt:
	dandi download https://dandiarchive.org/dandiset/000129
	mv 000129 data

# Download dmfc dataset from dandi
download_dmfc:
	dandi download https://dandiarchive.org/dandiset/000127
	mv 000127 data

# Download area2bump dataset from dandi
download_area2bump:
	dandi download https://dandiarchive.org/dandiset/000130
	mv 000130 data

# Download all datasets from dandi
download_all: download_mc_maze_small download_mc_maze_medium download_mc_maze_large download_mc_maze download_mc_rtt download_dmfc download_area2bump

# Run main.py file to 
run:
	python3 -m src.main 

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

# Get rid of files inside of data
clean_data:
rm -r -f data/*

# Get rid of all h5 files from result directory
clean_results:
rm -r -f results/*.h5

# Get rid of all ckpt files from checkpoint directory
clean_checkpoint:
rm -r -f src/checkpoint/*.ckpt

# Get rid of all csv files from log directory
clean_log:
rm -r -f src/log/*.csv

# Get rid of pycache
clean: clean_tests clean_tests_models clean_src clean_src_models

