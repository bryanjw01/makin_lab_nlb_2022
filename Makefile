
# Shows all make commands with descriptions
help:
	@echo "----------------------------------------------------------------"
	@echo "Environment Setup:"
	@echo
	@echo "  setup   - pip installs all dependencies from requirements.txt"
	@echo "            file by using pip -r command"
	@echo
	@echo "  test   - Runs unittest which verifies all paths and dependencies"
	@echo "            are correct"
	@echo
	@echo
	@echo "Download Data:"
	@echo
	@echo "  download_all			- downloads all of the dataset using dandi and"
	@echo "               			  moves the downloaded folder inside of the data"
	@echo "               			  folder"
	@echo
	@echo "  download_mc_maze_small	- downloads the mc_maze_small dataset using dandi"
	@echo "               			  and moves the downloaded folder inside of the"
	@echo "               			  data folder"
	@echo
	@echo "  download_mc_maze_medium	- downloads the mc_maze_medium dataset using dandi"
	@echo "               			  and moves the downloaded folder inside of the"
	@echo "               			  data folder"
	@echo
	@echo "  download_mc_maze_large	- downloads the mc_maze_large dataset using dandi"
	@echo "               			  and moves the downloaded folder inside of the"
	@echo "               			  data folder"
	@echo
	@echo "  download_mc_maze		- downloads the mc_maze dataset using dandi and"
	@echo "               			  moves the downloaded folder inside of the data"
	@echo "               			  folder"
	@echo
	@echo "  download_mc_rtt		- downloads the mc_rtt dataset using dandi and"
	@echo "               			  moves the downloaded folder inside of the data"
	@echo "               			  folder"
	@echo
	@echo "  download_dmfc			- downloads the dmfc_rsg dataset using dandi and"
	@echo "               			  moves the downloaded folder inside of the data"
	@echo "               			  folder"
	@echo
	@echo "  download_area2bump		- downloads the area2bump dataset using dandi and "
	@echo "                    		  moves the downloaded folder inside of the data"
	@echo "                    		  folder"
	@echo
	@echo
	@echo "Running Code:"
	@echo
	@echo "  run		- Runs main.py function"
	@echo 
	@echo
	@echo "Cleaning:"
	@echo
	@echo "  clean         - removes pycache from test and src directories"
	@echo
	@echo "  veryclean     - removes all h5, csv, and ckpt files as well"
	@echo "----------------------------------------------------------------"

# Setup Environment
setup:
	pip install -r requirements.txt

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

# Test config.py
test_config:
	python3 -m unittest tests.src.test_config

# Test all
test: test_config

# Run main.py file to 
run:
	python3 src/main.py 

# Get rid of pycache inside tests folder directory
clean_tests:
	find tests | grep -E "(__pycache__|\.pyc|\.pyo$))" | xargs rm -rf

# Get rid of pycache inside src directory
clean_src:
	find src | grep -E "(__pycache__|\.pyc|\.pyo$))" | xargs rm -rf

# Get rid of files inside of data
clean_data:
	rm -r -f data/*

# Get rid of all h5 files from result directory
clean_results:
	rm -r -f results/*.h5

# Get rid of all ckpt files from checkpoint directory
clean_checkpoint:
	rm -r -f checkpoint/*.ckpt

# Get rid of all csv files from log directory
clean_log:
	rm -r -f log/*.csv

# Get rid of pycache
clean: clean_tests clean_src

# get rid of pycache, csv, h5, ckpt
very_clean: clean_tests clean_src clean_log clean_checkpoint clean_results
