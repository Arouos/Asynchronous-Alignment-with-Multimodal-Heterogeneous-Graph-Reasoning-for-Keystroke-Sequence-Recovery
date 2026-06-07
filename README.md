<!-- Author: Haoyu Qiang -->

# AAM-HG Reference Implementation for Multimodal Keystroke Sequence Recovery

## 1. Project Overview

This project is a reference implementation of a sparse heterogeneous graph reasoning method for multimodal side-channel leakage on electronic password locks. The method is named AAM-HG. It targets the recovery of complete password sequences from acoustic, vibration, and electromagnetic modalities under asynchronous, overlapping, and partially missing conditions.
The pipeline covers multi-scale action candidate construction, sparse asynchronous alignment on a heterogeneous graph, action-level graph coarsening, and relation-enhanced ordered decoding. The CTC loss supports weakly supervised recovery of password symbol sequences.
The training objective takes the joint form $\mathcal{L} = \mathcal{L}_{\mathrm{ctc}} + \lambda_1 \mathcal{L}_{\mathrm{sparse}} + \lambda_2 \mathcal{L}_{\mathrm{deg}} + \lambda_3 \mathcal{L}_{\mathrm{tri}}$, which combines three regularizers for sparse edges, soft degree, and triangle consistency. Cross-modal asynchrony is handled by the modality-pair delay prior $\kappa_{m,n}$ and the dynamic temporal gate $g_{ij}^{(l)}$.

## 2. Directory Layout

```
PaperCode/
├── configs/
│   ├── default.yaml
│   ├── model/
│   ├── dataset/
│   └── experiment/
├── src/
│   ├── data/
│   ├── models/
│   ├── losses/
│   ├── metrics/
│   ├── training/
│   ├── evaluation/
│   ├── experiments/
│   └── utils/
├── scripts/
├── tests/
├── README.md
├── requirements.txt
├── setup.cfg
└── pyproject.toml
```

The repository root holds four subdirectories along with several packaging metadata files. Source code sits under `src/`. Configurations sit under `configs/`. Run entries sit under `scripts/`. Unit tests sit under `tests/`.

## 3. Environment and Dependencies

The project requires Python 3.10 or later. A conda virtual environment is recommended to avoid interference with the system Python. The commands below create and activate an isolated environment named `aamhg`.

```bash
conda create -n aamhg python=3.10 -y
conda activate aamhg
```

Inside the environment, pip handles the editable install of the project and the third-party dependencies in two steps. The first step registers `src` as an importable package. The second step installs runtime libraries that include PyTorch, PyTorch Geometric, torchaudio, numpy, scipy, scikit-learn, librosa, PyYAML, tqdm, requests, and einops.

```bash
pip install -e .
pip install -r requirements.txt
```

## 4. Dataset Acquisition

The ten public datasets are downloaded on demand at runtime through the GitHub links registered in `src/data/dataset_registry.py`. The download results land in a user-level cache directory rather than inside the repository, which simplifies reuse across machines. Before downloading, run the environment check script. It verifies the Python version, the required dependencies, and the writable state of the default cache directory.

```bash
python scripts/prepare_environment.py
```

After the check, each archive is fetched by dataset name and extracted into the cache directory. The dataset name takes values from `cmu`, `bbmas`, `aalto`, `clarkson2`, `buffalo`, `greyc`, `vuagnoux_pasini`, `zzt_acoustic`, `asonov_agrawal`, and `spiphone`.

```bash
python scripts/fetch_dataset.py --dataset <dataset_name>
```

Once the download finishes, the index builder scans the cached samples and produces a unified index for the training stage to consume.

```bash
python scripts/build_data_index.py --dataset <dataset_name>
```

## 5. Training Procedure

Training relies on three YAML files that work together. They specify the model structure, the dataset source, and the experiment settings. Model configurations sit in `configs/model/`. Dataset configurations sit in `configs/dataset/`. Experiment configurations sit in `configs/experiment/`. The command below points to all three files and fixes a random seed.

```bash
python scripts/train_runner.py --model-config configs/model/aam_hg.yaml --dataset-config configs/dataset/bbmas.yaml --experiment-config configs/experiment/main_comparison.yaml --seed <seed>
```

To reproduce the means and standard deviations reported in the tables, the seeds take values from $\{11, 23, 47, 91, 137\}$ and the run is repeated five times. Whenever the validation character error rate reaches a new low, the best weights are written to `checkpoints/best.pt` for later evaluation.

## 6. Evaluation Procedure

The evaluation stage loads the best checkpoint produced during training. It reports six metrics on the target dataset, namely the character error rate $\mathrm{CER}$, the sequence recovery accuracy $\mathrm{SRA}$, the key-level $\mathrm{Top}\text{-}1$, the key-level $\mathrm{Top}\text{-}3$, the key-level $\mathrm{F1}$, and the mean time offset $\Delta t$.

```bash
python scripts/evaluate_runner.py --model-config configs/model/aam_hg.yaml --dataset-config configs/dataset/bbmas.yaml --experiment-config configs/experiment/main_comparison.yaml --checkpoint <checkpoint_path>
```

Multi-seed results are aggregated into mean and standard deviation columns by `src/evaluation/result_aggregator.py`. The output format matches the table style used in the paper.

## 7. Experiment Reproduction

Six experiments share `scripts/experiment_dispatch.py` as a unified dispatcher. They cover the main comparison, SNR robustness, asynchronous offset robustness, modality missing, cross-dataset generalization, and ablation analysis. The dispatcher only selects the inner runner by experiment name. The actual data preparation, training, and evaluation are still handled by the scripts described in earlier sections.

```bash
python scripts/experiment_dispatch.py --experiment <experiment_name>
```

The experiment name takes values from `main_comparison`, `robustness_snr`, `robustness_offset`, `modality_missing`, `cross_dataset`, and `ablation`. Each value matches a YAML file with the same name under `configs/experiment/`.

## 8. Common Issues and Troubleshooting

PyTorch Geometric on Windows must match the local CUDA version. When `pip install` fails during compilation, install the prebuilt wheels for the matching CUDA version as suggested by the official documentation. This route bypasses the local build toolchain.

CUDA out-of-memory errors usually appear when both the node dimension and the retained cross-modal degree are set large. In the model configuration, lower `node_dim` and `cross_modal_topk` to a moderate level. Reduce the batch size and enable gradient accumulation. The effective optimization batch is preserved while the peak memory drops.

When GitHub download links are slow, the `github_url` field in the dataset configuration can be overridden by a mirror address. Archives can also be downloaded manually into the cache directory. After that, the download step is skipped and the index builder is run directly.