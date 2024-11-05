## NetGen-Py: A package for accelerating kinetic model generation using deep learning models
### Description
`NetGen-Py` is a package developed for kinetic model generation of gas-phase reaction systems.
### Introduction
Kinetic models are fundamental for reactor design and process optimization. Over the years, 
various methods and software packages have been developed to explore complex reaction systems
and generate kinetic models. These methods typically enumerate all possible reactions in the
large reaction space and estimate reaction rates using empirical approaches and quantum mechanics (QM).
However, the template-based nature of empirical methods and the high computational cost of QM
calculations hinder their application to complex systems.   
Deep learning, with its fast inference speed and template-free nature, has emerged as a promising
method for predicting reaction rates. Nonetheless, the accuracy of deep learning models is 
constrained by the quantity and quality of existing kinetic databases, posing a significant 
challenge to their robust application in kinetic model generation.  
We propose a framework for the robust application of deep learning models to accelerate kinetic
model generation. This framework filters out kinetically unfavorable reactions using deep learning
models, thereby narrowing the reaction space. `NetGen-Py` integrates deep learning models,
empirical methods, and QM techniques for efficient kinetic model generation.
### Installation
#### 1. Installing the conda package manager
#### 2. Installing the RMG-databse  
We utilized the [RMG-database](https://github.com/ReactionMechanismGenerator/RMG-database) estimated reaction rates
as the benchmark reaction rates. To install the RMG-database, please use following
commands to create the new environment named 'netgen':  
```conda create -c defaults -c rmg -c rdkit -c cantera -c pytorch -c conda-forge --name netgen rmg rmgdatabase```  
#### 3. Removing the chemprop package  
We utilized [chemprop](https://github.com/chemprop/chemprop/tree/v1.7.1) package to predict reaction rates. This package
is included in our codes. Therefore, the default chemprop package should be removed by:  
```conda remove --force chemprop```

#### 4. Checking the chemprop dependency
Installing the pytorch gpu version by pip if running the chemprop on gpu device. 
Checking the [chemprop](https://github.com/chemprop/chemprop/tree/v1.7.1) environment and installing the dependency if necessary  
You might use following command:
```
pip install typed-argument-parser  
conda uninstall pytorch-cpu  
```
We strongly recommend user using the `pytorch-gpu` version for faster employing the ML estimator  
Note that RMG-database is coupled with RMG-Py which can only be installed based on Python 3.7

#### 5. Change the RMG-database cache:  
If you get error when loading the RMG database. Go to  
```netgen/lib/python3.7/site-packages/rmgpy/data/kinetics/family.py```
line 771, adding following code:  
```
if name == '__pycache__':
    continue
```

#### 6. Installing pygraphviz and rdkit using pip for generating flux diagram
```
pip install pygraphviz # 1.7 was tested
pip install rdkit  # 2023.3.2 was tested
```
#### Note:
The code for kinetic rate calculation using Transition State Theory was provided in `netgen-qm` repository.
`netgen-qm` should be placed in your remote servers for automatically doing transition state search, 
thermodynamic property calculation, and reaction rate estimation. We do not provide the scripts for 
controlling your remote servers.

### Running NetGen codes
We will release `NetGen-Py` as a python package in the future. Now, these codes are developed for the coupling of machine learning and 
other method for kinetic model generation. Try our demo `n-C5H12_pyrolysis/crns_test.ipynb` on jupyter notebook.
