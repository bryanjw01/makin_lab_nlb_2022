# Makin Lab NLB 2022
[Neural Latent Workshop](https://neurallatents.github.io/) Submission. Top score on the [eval.ai](https://eval.ai/) leaderboard for [Neural Latents Benchmark '21](https://eval.ai/web/challenges/challenge-page/1256/overview) on [MC_RTT](https://eval.ai/web/challenges/challenge-page/1256/leaderboard/3187) and [area2bump](https://eval.ai/web/challenges/challenge-page/1256/leaderboard/3186).

# Introduction

Transformers have been proven to work well for language modeling and time-series forecasting. They capture long-term and short-term correlations across time using self-attention modules. Phase 1  winners of the Neural Latents Benchmark used Neural Data Transformers (NDT) for their winning model in all 4 dataset categories ([MC_Maze](https://github.com/neurallatents/neurallatents.github.io/blob/master/notebooks/mc_maze.ipynb), [MC_RTT](https://github.com/neurallatents/neurallatents.github.io/blob/master/notebooks/mc_rtt.ipynb), [Area2_Bump](https://github.com/neurallatents/neurallatents.github.io/blob/master/notebooks/area2_bump.ipynb), [DMFC_RSG](https://github.com/neurallatents/neurallatents.github.io/blob/master/notebooks/dmfc_rsg.ipynb)) . 

In the field of Neuroscience, the amount of data available is significantly smaller in comparison to the massive datasets used in NLP and Computer Vision. This is illustrated by the following table:

<!-- | Field           | Model            | Parameters    | Training Data | Ratio = Data / Parameters |
|-----------------|------------------|---------------|---------------|---------------------------|
| Neuroscience    | Neural r-RoBERTa | ~0.47 Million | ~51 MB        | 0.1085                    |
| NLP             | RoBERTa base     | ~125 Million  | ~160 GB       | 1.2800                    |
| Computer Vision | EfficientNet-B7  | ~ 66 Million  | ~150 GB       | 2.2727                    | -->

| Field           | Model           | Parameters    | Training Data        | Ratio = Data / Parameters |
|-----------------|-----------------|---------------|----------------------|---------------------------|
| Neuroscience    | NDT-2           | ~0.48 Million | ~ 51 MB (MC_RTT)     | 0.1062                    |
| NLP             | RoBERTa base    | ~125 Million  | ~496 GB (Books+Wiki) | 3.9680                    |
| Computer Vision | EfficientNet-B7 | ~ 66 Million  | ~150 GB (Imagenet)   | 2.2727                    |


# Method

Based on our hypothesis, when modeling Neural data, the temporal correlation is more localized. This makes it different from language modeling where information aggregation from long temporal distances becomes significant. Given the limited training data, this assumption allows us to reduce the model complexity. Hence, we use an RNN as our base model since it captures local dependencies across time well. 

We used Gated Recurrent Units (GRU) coupled with a feed-forward layer and maximized our performance on mc_rtt dataset (phase="val"). The following result which was 2nd on the leaderboard (higher co-bps than NDT) for mc_rtt was obtained using a 2-layer bi-directional GRU (~0.15 Million parameters, Ratio = 0.34):

| Dataset | Model        | Phase | co-bps | velR2  | fp-bps |
|---------|--------------|-------|--------|--------|--------|
| MC_RTT  | GRU->FF->exp | val   | 0.2050 | 0.5813 | 0.1122 |
| MC_RTT  | GRU->FF->exp | test  | 0.2119 | 0.6133 | 0.1148 |

However, we found that NDT achieves better scores for the metrics vel R2 and fp-bps. Hence we experimented with the following variants on the **MC_RTT** dataset on **phase="val"**:


| Model Architecture                          | co-bps | vel R2 | fp-bps |
| ------------------------------------------- | ------ | ------ | ------ |
| GRU(2) -> FF -> Conv -> FF -> exp           | 0.1987 | 0.5569 | 0.1225 |
| GRU(2) -> FF -> Deconv -> FF -> exp         | 0.1953 | 0.5280 | 0.1212 |
| GRU(2) -> FF -> Conv(smooth) -> FF -> exp   | 0.1916 | 0.5311 | 0.1129 |
| Conv -> GRU(2) -> FF -> Deconv -> FF -> exp | 0.1901 | 0.5383 | 0.0973 |
| GRU(2) -> FF -> FF -> exp                   | 0.1880 | 0.5941 | 0.1142 |
| GRU(2) -> FF -> FF -> Sigmoid               | 0.1844 | 0.5345 | 0.0860 |
| Conv -> GRU(1) -> FF -> Sigmoid             | 0.1808 | 0.5375 | 0.1107 |
| Conv -> Trans -> Deconv -> Trans -> exp     | 0.1753 | 0.4431 | 0.0655 |
| Conv -> Trans -> Deconv -> Trans -> Sigmoid | 0.1750 | 0.4634 | 0.0565 |
| Conv -> Trans -> Deconv -> Sigmoid          | 0.1726 | 0.3882 | 0.0706 |
| Conv -> Trans -> Deconv -> FF -> Sigmoid    | 0.1704 | 0.3936 | 0.0653 |
| Conv -> Trans -> FF -> Sigmoid              | 0.1555 | 0.3332 | 0.0601 |
| Deconv -> Trans -> FF -> Sigmoid            | 0.1521 | 0.3609 | 0.0664 |
| Deconv -> GRU(1) -> FF -> Sigmoid           | 0.1503 | 0.4233 | 0.0918 |

Based on the performance in the validation phase, we re-trained the top two models (in the table above) to get the following metrics on the **Test data** (phase="test") (on the leaderboard) :

| Model Architecture                     | co-bps | vel R2 | fp-bps |
| -------------------------------------- | ------ | ------ | ------ |
| GRU(2) -> FF -> RoBERTa(1) -> FF ->exp | 0.2074 | 0.6410 | 0.1279 |
| GRU(2) -> FF -> Conv -> FF -> exp      | 0.1959 | 0.6155 | 0.1065 |

# Quickstart

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
  