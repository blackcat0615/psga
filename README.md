# PSGA: Propio State Guided Attention To Resist VLA Attack

## Overview
This project aims to explore the impact of image salt-and-pepper noise and language noise on VLA model performance. At the same time, to mitigate the effects of noise, we have introduced the ** Propio State Guided Attention (PSGA) module ** , which reconstructs VLM features based on propio state to enhance the VLA model's resistance to noise.

Why use ontological information? There are two reasons for this. On one hand, when there is noise in visual and language features, we assume that propio state is reliable. On the other hand, there is an inherent correlation between propio state and visual information. To some extent, the movement of the robot changes in visual information, and propio state contains some temporal information, which is beneficial for improving VLA model performance.

The core algorithm enhances VLM features by incorporating proprioceptive state (e.g., robot joint angles, end-effector position) through a ** cross-attention mechanism **. The updated VLM feature is computed as the average of the original VLM feature and the cross-attended feature.

```python
vlm_feature = 0.5 * (vlm_feature + cross_att(q=vlm_feature, k=propio_state, v=propio_state))

```

This repo is based on [SimVLA](https://github.com/LUOyk1999/SimVLA/), thanks for their excellent work and useful repo.

## Hardware

GPU: 2 RTX 4090, CPU: 28 core 100G. CUDA: 12.8


## Start
To make it easier for everyone to set up the environment，this repo offers the requirement.

```bash
cd psga
conda create -n psga python=3.10 -y
conda activate psga

pip install -r requirements.txt

# install flash-attn, 28 core, set max_jobs=6, you can adjust max_jobs based on cpu resource.
MAX_JOBS=6 pip install flash-attn==2.5.6 --no-build-isolation -v

```

> Important: Use `transformers>=4.57.0`.

if you want to generate the similiar task desc of your own dataset, you can use the tool.
```bash
cd tools

python generate_similiar_task_desc.py

```

## Training (LIBERO Dataset)

### 1. Prepare LIBERO Dataset

Download [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) dataset, and place it in `./datasets/metas/`.

For more convenient data downloads, we provide a small tool.
```bash
grep -q "export HF_ENDPOINT=" ~/.bashrc || echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc && source ~/.bashrc

cd tools

python download_libero.py


```
### 2. Create Training Metadata

```bash
python create_libero_meta.py \
    --data_dir ./datasets/metas \
    --subsets libero_10 libero_goal libero_object libero_spatial \
    --output ./datasets/metas/libero_train.json
```

### 3. Compute Normalization Statistics

```bash
python compute_libero_norm_stats.py \
    --data_dir ./datasets/metas \
    --subsets libero_10 libero_goal libero_object libero_spatial \
    --output ./norm_stats/libero_norm.json
```

### 4. Start Training

**Small Model Configuration:**
```bash
bash train_smolvlm_small.sh
```

**Large Model Configuration:**
```bash
bash train_smolvlm_large.sh
```

### 5. Evaluation

```bash
cd evaluation/libero
```


### 6. Model Architecture

- **Vision-Language Backbone**: SmolVLM-500M-Instruct (576 hidden dim)
- **Action Transformer**: Configurable depth and width
  - Small: 768 hidden, 12 layers, 12 heads
  - Large: 1024 hidden, 24 layers, 16 heads


## Results and Analysis
This repo chooses the Libero object dataset for testing, the metric is success rate(SR)

### Results

[baseline model checkpoint] todo
[PSGA model checkpoint] todo

| experiments setting | baseline(simvla) | PSGA |
|:-------:|:--------:|:-------:|
| origin  |  86%   |   95% |
| origin+img_noise(0.02)  |  82%   |   97% |
| origin+img_noise(0.05)  |  81%   |   94% |
| origin+img_noise(0.1)  |  77%   |   95% |
| origin+img_noise(0.2)  |  48%   |   47% |
| origin+language  |  82%   |   99% |

### Analysis
From the experimental results, it can be observed that after incorporating the PSGA module, the model achieves a 9% improvement over the baseline in the absence of noise, which stems from the temporal information embedded in the proprio state. When salt-and-pepper noise is added to the images, under low-noise conditions (≤ 0.1), the model with the PSGA module exhibits reduced sensitivity to noise.



## Others

If you are interested in PSGA, you can try to use PSGA to your model, and if you have any questions or suggestions, you can raise an issue.



## Acknowledgements

- [SimVLA](https://github.com/LUOyk1999/SimVLA/)

- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
