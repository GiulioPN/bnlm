bnplm
====

`bnplm` is a C++/python library for nonparametric Bayesian modeling with Pitman-Yor process priors. It is an exthension of the the C++ library [cpyp](https://github.com/redpony/cpyp). 

## Structure 
The package consists in two main phases: 

- Preprocessing: using the ```preprocessing/preprocessCorpus.py```. Row Data goes to ```preprocessing/rawData/``` folder.
- Analysis: using the function in ```utlis.py``` for the training a teting of the model. The results (evaluation probabilities and performance) are saved in ```output/``` folder. 

## Installation
From terminal, in the main directory ```$ bash compile```

## Data 

```data/train.txt``` and ```data/test.txt``` are used for an example of simulation. 

## Features
- Memory-efficient histogram-based sampling scheme proposed by [Blunsom et al. (2009)](http://www.clg.ox.ac.uk/blunsom/pubs/blunsom-acl09-short.pdf)
- Full range of PYP hyperparameters (0 ≤ discount < 1, strength > -discount, etc.)
- Beta priors on discount hyperparameter
- (Conditional, given discount) Gamma prior on strength hyperparameter
- Tied hyperparameters
- Slice sampling for hyperparameter inference

## System Requirements
This library should work with any C++ compiler that implements the [C++11 standard](http://en.wikipedia.org/wiki/C%2B%2B11). No other libraries are required.

