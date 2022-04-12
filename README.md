# Makin Lab NLB 2022
[Neural Latent Workshop](https://neurallatents.github.io/) Submission. Top score for [mc_rtt](https://eval.ai/web/challenges/challenge-page/1256/leaderboard/3187) and [area2bump](https://eval.ai/web/challenges/challenge-page/1256/leaderboard/3186).

# Introduction

Transformers have been proven to work well for language modeling and time-series forecasting. They capture long-term and short-term correlations across time using self-attention modules. Phase 1  winners of the Neural Latents Benchmark used Neural Data Transformers (NDT) for their winning model in all 4 dataset categories. 

In the field of Neuroscience, the amount of data available is significantly smaller in comparison to the massive datasets in NLP and Computer Vision. 

| Field           | Model            | Parameters    | Training Data | Ratio = Data / Parameters |
|-----------------|------------------|---------------|---------------|---------------------------|
| Neuroscience    | Neural r-RoBERTa | ~0.35 Million | ~51 MB        | 0.1457                    |
| NLP             | RoBERTa base     | ~125 Million  | ~160 GB       | 1.2800                    |
| Computer Vision | EfficientNet-B7  | ~ 66 Million  | ~150 GB       | 2.2727                    |

Based on our hypothesis, when modeling Neural data, the temporal correlation is more localized. This makes it different from language modeling where information aggregation from long temporal distances becomes significant. Given the limited training data, this assumption allows us to reduce the model complexity.

Hence, we use an RNN as our base model since it captures local dependencies across time well. 

We used a Gated-Recurrent Neural Network (GRU) coupled with a feed-forward layer and maximized our performance on mc_rtt dataset.

The following result which was 2nd on the leaderboard (higher co-bps than NDT) for mc_rtt was obtained using a 2-layer bi-directional GRU:

| Dataset | Model        | Phase | co-bps | velR2  | fp-bps |
|---------|--------------|-------|--------|--------|--------|
| mc_rtt  | GRU->FF->exp | val   | 0.2050 | 0.5813 | 0.1122 |
| mc_rtt  | GRU->FF->exp | test  | 0.2119 | 0.6133 | 0.1148 |


# Instructions

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
  