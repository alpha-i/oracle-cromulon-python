# oracle crocubot [![CircleCI](https://circleci.com/gh/alpha-i/oracle-crocubot-python.svg?style=svg&circle-token=f6a7198d3b32ae0fb56dfec1daee167a930445eb)](https://circleci.com/gh/alpha-i/oracle-crocubot-python)


A library for predicting returns using Bayes by Backprop


![Crocubot](docs/crocubot.jpg "Crocubot")

## Setup Development Environment

### Create conda environment
```bash
$ conda create -n crocubot-env python=3.6
$ source activate crocubot-env
```

### Install dependencies

```bash
$ pip install -u setuptools --ignore-installed --no-cache-dir
$ pip install -r requirements.txt --src $CONDA_PREFIX
$ pip install -r requirements_alphai.txt --src $CONDA_PREFIX
$ pip install -r requirements_dev.txt
```

### Running the test suite
```bash
$ PYTHONPATH=. python -m pytest tests/
```


## Installation on Windows
We need to install `scipy` and `scikit-learn` using `conda` as it requires some external libraries which are not easy to build in Windows.
```commandline
conda install scipy==0.18.1
conda install scikit-learn==0.18.1
```
We will need to set the variables for the GSL in the environment first. 
```commandline
set GSL_INCLUDE_DIR=C:\Users\sree\Documents\Software\gsl\include
set GSL_LIBRARY_DIR=C:\Users\sree\Documents\Software\gsl\lib
```
Now install the requirements using `pip`.
```commandline
pip install -U setuptools --ignore-installed --no-cache-dir
pip install -r requirements.txt --src %CONDA_PREFIX%
pip install -r requirements_alphai.txt --src %CONDA_PREFIX%
pip install -r requirements_dev.txt
```

