import pickle, argparse
from sklearn.preprocessing import LabelBinarizer
from sklearn import preprocessing
from src.utils import list_images, split_and_get_labels, distribute_paths
from src.image_preproc import extract_nail

from hpsklearn import HyperoptEstimator
import numpy as np
import tqdm
import pdb

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

	# Check valid model
	if not filename.endswith(".pkl"):
		print("Not a valid model. Please load a .pkl model.")
		return False

	# List all of the images
	paths, labels = list_images(path_to_images, ext)

	# Get paths correctly distibuted good/bad
	n_paths = distribute_paths(paths)

	# Binarize labels
	lb = LabelBinarizer()
	lb.fit(labels)	

	# Set number of feats
	n_feats = crop_size**2 + 2

	# Create vectors to process the data
	x_train = np.zeros((len(n_paths), n_feats), np.uint8)
	y_train = np.zeros((len(n_paths), 1), np.int32)

	# Create progress bar
	progress = tqdm.tqdm(total=len(n_paths))

	# Fill the vectors with nails and 
	print("Extracting nails ...")
	for idx, path in enumerate(n_paths):
			x_train[idx, :] = extract_nail(path)
			y_train[idx] = lb.transform([path.split("_")[-1].split(".")[0]])
			progress.update(1)
	y_train = np.ravel(y_train)

	pkl_filename = 'model/'+filename
	# Load from file
	with open(pkl_filename, 'rb') as file:
	    pickle_model = pickle.load(file)

	print("Loading model")
	# Load model and preprocessing 
	learner = pickle_model['learner']
	preproc = pickle_model['preprocs']
	ex_preprocs = pickle_model['ex_preprocs']

	print("Model: {}".format(learner))
	print("Preprocessing: {}".format(preproc))

	estim = HyperoptEstimator()
	estim._best_learner = learner
	estim._best_preprocs = preproc
	estim._best_ex_preprocs = ex_preprocs

	print("Calculating accuracy score and predicting target values...")
	# Calculate the accuracy score and predict target values
	score = estim.score(x_train, y_train)
	print("Test score: {0:.2f} %".format(100 * score))

if __name__ == '__main__':
	main()
