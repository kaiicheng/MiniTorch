<img src="https://minitorch.github.io/minitorch.svg" width="50%">

# MiniTorch

*from minitorch page*

* Docs: https://minitorch.github.io/

<!-- * Overview: https://minitorch.github.io/module4.html -->

This project is an re-implemented version of PyTorch at the Machine Learning Engineering Course project [minitorch](https://minitorch.github.io/). 

Instructed by Professor Sasha Rush at Cornell Tech.

## Descriptions 

- Constructed a deep learning system using Python, including auto-differentiation, backpropagation, and tensor matrix operations.

- Implemented parallel computing with Numba and CUDA.

- Visualized with Streamlit and tested functions using pytest and Flake8.

## Overview

This module requires `fast_ops.py`, `cuda_ops.py`, `scalar.py`, `tensor_functions.py`, `tensor_data.py`, `tensor_ops.py`, `operators.py`, `module.py`, and `autodiff.py` from Module 3.


Additionally you will need to install and download the MNist library.

(On Mac, this may require installing the `wget` command)

```
pip install python-mnist
mnist_get_data.sh
```


* Tests:

```
python run_tests.py
```

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```
