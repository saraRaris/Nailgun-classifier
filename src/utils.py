import os, random, glob
import numpy as np


def list_images(path_to_images, ext):
	'''
	List all of the images inside a path and return the label
	@params:
		path_to_images: path to the images. Inside this directory there should be one
						folder for each class.
		ext: type of extension. E.g. .jpeg
	@return:
		paths: list of paths to each of the images. Size: n_classes x n_imgs
		labels: each of the label for the pictures
	'''
	paths = []
	labels = []

	for root, dirs, files in os.walk(path_to_images):
		for dir_ in dirs:
			paths.append(glob.glob(os.path.join(root, dir_, '*'+ext)))
			labels.append(dir_)

	return paths, labels

def distribute_paths(paths):
	'''
	Distribute paths in a vector corresponding to the weight of each class. For example,
	if there are two classes equally distributed the vector would look like: [0,1,0,0,1, ...] but,
	if there are twice the numbers of samples in class 0, then it would look like: [0,0,1,0,0,1,..]
	@params
		paths: list of paths to each of the images. Size: n_classes x n_img
	@return
		n_paths: row vector containing the path to each of the images balanced
	'''
	total_files = sum([len(p) for p in paths])
	n_paths = []

	for idx_c, class_ in enumerate(paths):
		factor = float(total_files / len(class_))
		random.shuffle(class_)

		for idx_f, file_ in enumerate(class_):
			n_paths.insert(int(idx_f * factor)+ idx_c, file_)

	return n_paths

def split_and_get_labels(paths, split_factor):
	'''
	Function to split a row string vector containing balanced distribution of the classes
	@params:
		paths: row vector containing string paths to each of the images
		split_factor: value [0,1] indicating the number of samples going to the training. 1-split_factor
					go to the test.
	@return:
		(x_train_paths, y_train), (x_test_paths, y_test): paths to train/test images and their corresponding label.
	'''

	x_train_paths = paths[0:int(split_factor * len(paths))]
	random.shuffle(x_train_paths)

	x_test_paths = paths[int((split_factor) * len(paths)):]
	random.shuffle(x_test_paths)

	y_train = [name.split("_")[-1].split(".")[0] for name in x_train_paths]
	y_test = [name.split("_")[-1].split(".")[0] for name in x_test_paths]

	return (x_train_paths, y_train), (x_test_paths, y_test)