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

## Data Structures

- Data dirs

```sh
data
├── scp # Text of file list
│   └── jvs
│       ├── train.scp # train set
│       ├── valid.scp # validation set
│       └── test.scp  # test set
└── wav # WAV Files
    └── jvs
        ├── jvs001
        ├── jvs002
        ...
```

- Data list file

```txt
data/wav/jvs/jvs001/VOICEACTRESS100_005.wav
data/wav/jvs/jvs001/VOICEACTRESS100_006.wav
data/wav/jvs/jvs001/VOICEACTRESS100_007.wav
...
```

## Run

```sh
# 1. Extract acoustic features (output to data/hdf5/...)
e2enf-extract_features file_list=data/scp/jvs/train.scp
# 2. Compute statistics of training data
e2enf-compute_statistics  feats=data/scp/jvs/train.list stats=data/stats/jvs_no_dev.joblib
# 3. Train model
e2enf-train out_dir=exp/jvs.e2enf
# 4. Inference from acoustic parameters
e2enf-decode out_dir=exp/jvs.e2enf checkpoint_steps=400000
# 5. Inference from wav file
e2enf-anasyn in_dir=data/wav/path/to/wav_files/ out_dir=exp/foo/bar stats=data/stats/jvs_no_dev.joblib checkpoint_path=exp/path/to/checkpoint.pth

# For previous study
nf-train out_dir=exp/jvs.neuralformants
nf-decode out_dir=exp/jvs.neuralformants checkpoint_steps=99000
```
