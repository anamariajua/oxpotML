# ML-based Predictions of Oxidation Potential of π-Conjugated Molecules: A Cautionary Tale

This repository contains all code and data needed to reproduce the results from our study **“ML-based predictions of oxidation potential of π-conjugated molecules: A cautionary tale.”**  
It covers:

* Datasets used in this work (.csv) (`/datasets`)
* "Classical" ML models development and optimization (`/classical_ML`)
* GCN model development and optimization (`/GCN`)
* Multiple strategies for defining the **Applicability Domain** (AD) (`/ad_methods`)
* Helper functions (`/docs`)
* Splits used to reproduce the results (`/splits`)

---

## Quick Start
```bash
# clone the repo
git clone https://github.com/your-username/oxidation-potential-ml.git
cd oxidation-potential-ml

# create the conda environment
conda env create -f environment.yml
conda activate oxpot-ml
