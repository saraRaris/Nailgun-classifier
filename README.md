## READ ME

This is a small project that aims to use several python libraries to explore and gain insight about a small dataset. The library HyperOptThis can be found at: <https://github.com/hyperopt/hyperopt>. This dataset has two categories of nailguns: `good` and `bad`. The goal of this repository is to build a classifier to distinguish among theese classes.

## Instructions

The following libraries should be installed:

* scikit-learn == '0.21.3'
* hyperopt=='0.2.3'
* opencv=='4.1.2'
* sklearn=='0.22'

To retrain the model the file train.py should be ran. To test on the current model `model.pkl` run test.py. To visualize all of the image processing steps performed for the segmentation of the nail please run:

	python debug_extract_nails.py

	

## Overview

The first steps aim to detect the nail in the image, extract it from the background and copy it to a black background. The result is a new image with the nail in the center and black background. This is achieved using image processing techniques.

Once the nail has been extracted and the background removed, each of the pixels in this new image are fed as features into the HyperOpt library which chooses the best estimator and the best parameters. 