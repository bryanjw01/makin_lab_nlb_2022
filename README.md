# makin_lab_nlb_2022
Neural Latent Workshop Submission

## Requirements
   * python 3.7+
   * installed dependencies from `requirements.txt` 
   * at least 30GB of free disk space
   * at least 32GB RAM (all datasets except `mc_maze` can be validated on 16GB)
   * dandi only supported by **native** or **conda** Python
## Setup

1. Download dependencies from requirements.txt:
   ```
   make setup
   ```
2. download datasets from dandi:
   * If using **conda** then:
      ```
      conda install -c conda-forge dandi
      ```
      ```
      make download_mc_maze_small
      ```
   * Otherwise:
      ```
      make download_mc_maze_small 
      ```
3. Edit makin_lab_nlb_2022/src/config.py:
   * Change paths (LOG_PATH, RESULT_PATH, CHECKPOINT_PATH, DATA_PATH_DICT)
   * Choose dataset + model

4. Running 
   ```
   make run
   ```
  
