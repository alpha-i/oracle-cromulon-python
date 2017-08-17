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
$ pip install -r requirements_dev.txt
```

### Running the test suite
```bash
$ PYTHONPATH=. python -m pytest tests/
```
