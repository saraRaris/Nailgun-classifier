from src.utils import list_images, split_and_get_labels, distribute_paths
from src.image_preproc import extract_nail
from sklearn.preprocessing import LabelBinarizer
from hpsklearn import HyperoptEstimator, any_classifier
from hpsklearn.components import any_preprocessing
from sklearn import metrics
from hyperopt import tpe
import numpy as np
import pickle, tqdm
import argparse

def main():
	# Construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--path", default='nailgun', help="path to nailgun folder")
	ap.add_argument("-m", "--model", required= True, help="name of the model file to save the model")
	ap.add_argument("-cs", "--csize", default=80, help="paramter to crop the image around the nailgun")
	ap.add_argument("-ex", "--ext", type=str, default='.jpeg', help="extension of the images")
	args = vars(ap.parse_args())

	# Load paramters
	crop_size = args['csize']
	path_to_images = args['path']
	filename = args['model']
	ext = args['ext']

	split_factor = 0.75

	# List all of the images
	paths, labels = list_images(path_to_images, ext)

	# Get paths correctly distibuted good/bad
	n_paths = distribute_paths(paths)

	# Split and generate labels
	(x_train_paths, y_train_str), (x_test_paths, y_test_str) = split_and_get_labels(n_paths, split_factor)

	print('--- Split ---')
	print('Train: '+str(len(x_train_paths))+', Test: '+str(len(x_test_paths)))

	# Load object for label binarizer
	lb = LabelBinarizer()
	lb.fit(y_train_str)	
	
	n_feats = crop_size**2 + 2
	x_train = np.zeros((len(x_train_paths), n_feats), np.uint8)
	y_train = np.zeros((len(y_train_str), 1), np.int32)

	print('---- Extracting Train samples ----')
	progress = tqdm.tqdm(total=len(x_train_paths))

	for idx, path in enumerate(x_train_paths):
		x_train[idx, :] = extract_nail(path)
		y_train[idx] = lb.transform([path.split("_")[-1].split(".")[0]])
		progress.update(1)

	y_train = np.ravel(y_train)

	print('---- Extracting Test samples ----')
	progress = tqdm.tqdm(total=len(x_test_paths))

	x_test = np.zeros((len(x_test_paths), n_feats), np.float)
	y_test = np.zeros((len(y_test_str), 1), np.int32)
	for idx, path in enumerate(x_test_paths):
		x_test[idx, :] = extract_nail(path)
		y_test[idx] = lb.transform([path.split("_")[-1].split(".")[0]])
		progress.update(1)

	y_test = np.ravel(y_test)

	# Define HyperoptEstimator
	estim = HyperoptEstimator(classifier=any_classifier('clf'), preprocessing=any_preprocessing('pp'), algo=tpe.suggest, trial_timeout=30)
	estim.fit(x_train, y_train)

	print('---- BEST SCORE (acc) ----')
	print( estim.score( x_test, y_test ) )

	print('---- BEST MODEL ----')
	print( estim.best_model() )

	pkl_filename = 'model/'+filename+'.pkl'
	with open(pkl_filename, 'wb') as file:
		pickle.dump(estim.best_model(), file)

	print('--- Correctly saved! ---')

if __name__ == '__main__':
	main()

