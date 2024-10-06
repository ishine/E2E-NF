# End-to-End Neural Formant Synthesis Using Low-Dimensional Acoustic Parameters

This repo provides official PyTorch implementation of [E2E-NF/+, E2E-SiFi-NF](https://misumisumi.github.io/myprojects/end-to-end-neural-formant-synthesis),
generating high-quality speech waveform with controllable acoustic parameters from low-dimensional representations.
Also includes an unofficial implementation of [Neural Formant Synthesis](https://arxiv.org/pdf/2306.01957).
For more information, please see out [DEMO](https://misumisumi.github.io/myprojects/end-to-end-neural-formant-synthesis).

## Environment Setup

```sh
cd E2E-NF
pip install -e .
# or use flake
nix develop default
```

## Run

```sh
# 1. Extract acoustic features
e2enf-extract_features audio=data/scp/jvs.scp
# 2. Compute statistics of training data
e2enf-compute_statistics  feats=data/scp/jvs/train.list stats=data/stats/jvs_train.joblib
# 3. Train model
e2enf-train
# 4. Inference from acoustic parameters
e2enf-decode = "e2enf.bin.decode:main"

# For previous study
nf-train = "e2enf.bin.neuralformants.train:main"
nf-decode = "e2enf.bin.neuralformants.decode:main"
```
