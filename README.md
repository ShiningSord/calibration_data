<div align="center">
<h1>Calibration Data</h1>
</div>

This repo contains the code for "Beware of Calibration Data for Pruning Large Language Models" ([https://arxiv.org/abs/2410.17711](https://arxiv.org/abs/2410.17711))

## Quick start
### Installation
Create and activate the conda environment:
```
conda env create -f environment.yml
conda activate calibration_data
```
This environment installs PyTorch 2.1.1 for CUDA 12.1 and uses NumPy 1.13.x.

### Empirical study
```
cd Wanda
bash run_dclm_wanda.sh
```

### Self-generating synthetic data
```
bash run_gendata.sh
python sample_ppl.py
```

### Evaluation
```
cd Wanda
bash run_dclm_wanda_sample.sh
```
