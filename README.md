# makin_lab_nlb_2022
Neural Latent Workshop Submission. Top score for area2bump and mc_rtt.

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
    ```
    make download_mc_maze_small 
    ```
3. Run config unittest:
    ```
    make test
    ```
4. Edit `src/config.py` File (optional)
  * Can Change:
    * number of epochs
    * dataset
    * model
    * etc
5. Running 
   ```
   make run
   ```
  
