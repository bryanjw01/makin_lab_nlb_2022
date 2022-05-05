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

<!--| Field           | Model           | Parameters | Training Data (GB)  | Ratio = Data / Parameters |
|-----------------|-----------------|------------|---------------------|---------------------------|
| Neuroscience    | NDT-2           | ~0.48 M    | ~0.051 (MC_RTT)     | 0.1062                    |
| NLP             | RoBERTa base    | ~125  M    | ~496   (Books+Wiki) | 3.9680                    |
| Computer Vision | EfficientNet-B7 | ~ 66  M    | ~150   (Imagenet)   | 2.2727                    | -->

| Field           | Model           | Parameters | Training Data (GB)  | Ratio = Data / Parameters |
|-----------------|-----------------|------------|---------------------|---------------------------|
| Neuroscience    | NDT-2           | ~0.07 M    | ~0.016 (area2_bump) | 0.2286                    |
| NLP             | RoBERTa base    | ~125  M    | ~496   (Books+Wiki) | 3.9680                    |
| Computer Vision | EfficientNet-B7 | ~ 66  M    | ~150   (Imagenet)   | 2.2727                    | 


# Method

Based on our hypothesis, when modeling Neural data, the temporal correlation is more localized. This makes it different from language modeling where information aggregation from long temporal distances becomes significant. Given the limited training data, this assumption allows us to reduce the model complexity. Hence, we use an RNN as our base model since it captures local dependencies across time well. 

We used Gated Recurrent Units (GRU) coupled with a feed-forward layer and maximized our performance on mc_rtt dataset (phase="val"). The following result which was 2nd on the leaderboard (higher co-bps than NDT) for mc_rtt was obtained using a 2-layer bi-directional GRU (~0.15 Million parameters, Ratio = 0.34):

| Dataset | Model        | Phase | co-bps | velR2  | fp-bps |
|---------|--------------|-------|--------|--------|--------|
| MC_RTT  | GRU->FF->exp | val   | 0.2050 | 0.5813 | 0.1122 |
| MC_RTT  | GRU->FF->exp | test  | 0.2119 | 0.6133 | 0.1148 |

However, we found that NDT achieves better scores for the metrics vel R2 and fp-bps. In the above model, we're using the RNN to predict forward in time and the feed forward layer to make the predictions in space (held_in + held_out neurons). Since our initial hypothesis was based on RNN's ability to make predictions in time, we experimented with the variants of the base model that would be a better estimator for space. The best results of these variants on on the **MC_RTT** dataset (**phase="val"**) is displayed below :


<!-- | Model Architecture                          | co-bps | vel R2 | fp-bps | -->
<!-- | ------------------------------------------- | ------ | ------ | ------ | -->
<!-- | GRU(2) -> FF -> RoBERTa(1) -> FF ->exp	    | 0.2027 | 0.5311	| 0.1257 | -->
<!-- | GRU(2) -> FF -> Conv -> FF -> exp           | 0.1987 | 0.5569 | 0.1225 | -->
<!-- | GRU(2) -> FF -> Deconv -> FF -> exp         | 0.1953 | 0.5280 | 0.1212 | -->
<!-- | GRU(2) -> FF -> Conv(smooth) -> FF -> exp   | 0.1916 | 0.5311 | 0.1129 | -->
<!-- | Conv -> GRU(2) -> FF -> Deconv -> FF -> exp | 0.1901 | 0.5383 | 0.0973 | -->
<!-- | GRU(2) -> FF -> FF -> exp                   | 0.1880 | 0.5941 | 0.1142 | -->
<!-- | GRU(2) -> FF -> FF -> Sigmoid               | 0.1844 | 0.5345 | 0.0860 | -->
<!-- | Conv -> GRU(1) -> FF -> Sigmoid             | 0.1808 | 0.5375 | 0.1107 | -->
<!-- | Conv -> Trans -> Deconv -> Trans -> exp     | 0.1753 | 0.4431 | 0.0655 | -->
<!-- | Conv -> Trans -> Deconv -> Trans -> Sigmoid | 0.1750 | 0.4634 | 0.0565 | -->
<!-- | Conv -> Trans -> Deconv -> Sigmoid          | 0.1726 | 0.3882 | 0.0706 | -->
<!-- | Conv -> Trans -> Deconv -> FF -> Sigmoid    | 0.1704 | 0.3936 | 0.0653 | -->
<!-- | Conv -> Trans -> FF -> Sigmoid              | 0.1555 | 0.3332 | 0.0601 | -->
<!-- | Deconv -> Trans -> FF -> Sigmoid            | 0.1521 | 0.3609 | 0.0664 | -->
<!-- | Deconv -> GRU(1) -> FF -> Sigmoid           | 0.1503 | 0.4233 | 0.0918 | -->


| Model Architecture                          | co-bps | vel R2 | fp-bps |
| ------------------------------------------- | ------ | ------ | ------ |
| GRU(2) -> FF -> RoBERTa(1) -> FF ->exp	    | 0.2027 | 0.5311	| 0.1257 |
| GRU(2) -> FF -> Conv -> FF -> exp           | 0.1987 | 0.5569 | 0.1225 |
| GRU(2) -> FF -> Conv(smooth) -> FF -> exp   | 0.1916 | 0.5311 | 0.1129 |
| GRU(2) -> FF -> FF -> exp                   | 0.1880 | 0.5941 | 0.1142 |
| GRU(2) -> FF -> FF -> Sigmoid               | 0.1844 | 0.5345 | 0.0860 |
| Conv -> GRU(1) -> FF -> Sigmoid             | 0.1808 | 0.5375 | 0.1107 |
| Conv -> Trans -> FF -> Sigmoid              | 0.1555 | 0.3332 | 0.0601 |

Sigmoid activation at the output in place of exp was noticed to converge faster. (Note : For the MC_RTT dataset, the maximum spikes per bin was identified to be 1). Based on the performance in the validation phase, we re-trained the top two models (in the table above) to get the following metrics on the **Test data** (phase="test") (on the leaderboard) :

| Model Architecture                     | co-bps | vel R2 | fp-bps |
| -------------------------------------- | ------ | ------ | ------ |
| GRU(2) -> FF -> RoBERTa(1) -> FF ->exp | 0.2074 | 0.6410 | 0.1279 |
| GRU(2) -> FF -> Conv -> FF -> exp      | 0.1959 | 0.6155 | 0.1065 |

As seen from the above table, the RoBERTa based model achieves better vel R2 and fp-bps scores on the Test data compared to the GRU. (Note: for co-bps and fp-bps, it does better than the NDT variants). This model was found to be more stable during training. The RNN half of the network predicts forward in time, the feed-forward expands in space and the transformer makes corrections in space to provide better results for vel R2 and fp-bps. Adding a Transformer layer on top of the RNN layer aggregates infomation from across time-steps and neuron channels to generate relatively more stable predictions.

In order to improve all three metrics (co-bps, vel R2, fp-bps), we combined the predictions of the GRU model along with the RoBERTa variant to achieve the top score on MC_RTT. 

# Model Architecture

| RNNf | Neural RoBERTa | Neural r-RoBERTa |
|------|----------------|------------------|
| ![RNNf](/images/RNNf.png)  |  ![Neural_RoBERTa](/images/Neural_RoBERTa.png) |  ![Neural_r_RoBERTa](/images/Neural_r_RoBERTa.png) |

# Results
All 3 model submissions for **MC_RTT** on the **Leaderboard** have been summarised below :

| Rank | Model Architecture | co-bps | vel R2 | fp-bps | Parameters | Training Data | Ratio = Data / Parameters |
| ---- | ------------------ | ------ | ------ | ------ | ---------- | ------------- | ------------------------- |
| 1    | Neural r-RoBERTa   | 0.2168 | 0.6489 | 0.1341 | ~0.47 M    | ~0.051 GB     | 0.1085                    |
| 2    | RNNf               | 0.2119 | 0.6133 | 0.1148 | ~0.15 M    | ~0.051 GB     | 0.3400                    |
| 3    | Neural RoBERTa     | 0.2074 | 0.6410 | 0.1279 | ~0.32 M    | ~0.051 GB     | 0.1594                    |

For the other 3 datasets, we didn't tune the models and submitted the results using the same hyper-parameters as used for MC_RTT dataset. This is especially true for MC_Maze where the training was interrupted midway for the submission. Hence we refrain from reporting results or claims about our model performance on these datasets.

# Training
For the *RNN* and *Neural RoBERTa* base models, we performed hyper-parameters semi-manually (Combination of optimizing manually + grid search):

| Parameter       | Values                                                 |
|-----------------|--------------------------------------------------------|
| lr              | [0.0005, 0.001, 0.005, 0.01, 0.015]                    |
| dropout         | [0.46, 0.5, 0.6]                                       |
| l2_weight       | [0, 5e-7, 5e-8]                                        |
| hidden          | [32, 40, 64]                                           |
| num_layers      | [1, 2, 3]                                              |
| bi-directional  | [True, False]                                          |
| input_size      | [DT, dT]                                               |
| mask_ratio      | [0.25, 0.27, 0.30]                                     |
| mask_variants   | [dot line, dot strip line, strip line, dot L, strip L] |

** Please note that not all combinations from this were tested (exhaustively).

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
  
