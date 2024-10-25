<div align="center">
<h1>Calibration Data</h1>
</div>

This repo contains the code for "Beware of Calibration Data for Pruning Large Language Models" ([https://arxiv.org/abs/2410.17711](https://arxiv.org/abs/2410.17711))

## Quick start
### Installation
```
pip install -r requirement.txt
```

### Empirical study
```
cd Wanda
bash run_dclm_wanda.sh
```

### Self-generating synthetic data
```
bash run_generate.sh
python sample_ppl
```

### Evaluation
```
cd Wanda
bash run_dclm_wanda_sample.sh
```
