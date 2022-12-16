# Linear vs blackbox

This repository contains code and data for reproducing *Interpretable models for extrapolation in scientific machine learning*. No ongoing maintenance or support of this content should be expected.

## Steps for reproduction
1. download this folder and open a terminal inside it
1. create a new python environment: `python3 -m venv venv`
1. activate the python environment: `. venv/bin/activate`
1. update pip in the python environment: `python -m pip install -U pip`
1. install dependencies in the python environment: `pip install -r requirements.txt`
1. start the jupyter server: `jupyter lab`
1. open `workflow.ipynb` in jupyter and run it

## Description of files
* `workflow.ipynb`: juypter notebook which reproduces the paper
* `utils.py`: python functions imported by `workflow.ipynb`
* `data`
    * `dataset_config.csv`: specifies the name, target property, size, and source of each dataset
    * `fig`: figures generated by `workflow.ipynb` and used in the paper
    * `raw`: raw data files in CSV format




> **Distribution A**
> Approved for Public Release, Distribution Unlimited