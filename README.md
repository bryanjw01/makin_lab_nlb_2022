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
4. Edit `src/config.py` file
5. Running 
   ```
   make run
   ```
  
# Method Description

Transformers have been proven to work well for language modeling and time-series forecasting. They capture long-term and short-term correlations across time using self-attention modules. 

But when modeling Neural data, it is observed that the temporal correlation is more localized. This is different from language modeling where information aggreagtion from  long temporal distances become significant. 

We use an RNN as our base module since it captures local dependencies across time well. 

We used a Gated-Recurrent Neural Network (GRU) coupled with a feed-forward layer as our base model and maximized our performance on mc_rtt (phase="val") dataset. 

The following result which was 2nd on the leaderboard for mc_rtt was obtained using a 2-layer bi-directional GRU :

| Dataset | Model        | Phase | co-bps | velR2  | fp-bps |
|---------|--------------|-------|--------|--------|--------|
| mc_rtt  | GRU->FF->exp | val   | 0.2050 | 0.5813 | 0.1122 |
| mc_rtt  | GRU->FF->exp | test  | 0.2119 | 0.6133 | 0.1148 |
